
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch.utils.data import DataLoader

from dataset_sim_physical import MC2DFIRSimDataset
from ioperator import MatlabLikeISAROp
from model_sim_physical import StageWiseFIRGEMNetSim


# ============================================================
# paths
# ============================================================
TRAIN_ROOT = r"D:\document\yanyi\电子书\论文\贝叶斯算法的寻优\2DFIRclaude\仿真数据训练\train"
VAL_ROOT = r"D:\document\yanyi\电子书\论文\贝叶斯算法的寻优\2DFIRclaude\仿真数据训练\val"
SAVE_DIR = r"D:\document\yanyi\电子书\论文\贝叶斯算法的寻优\2DFIRclaude\仿真数据训练\代码\checkpoints_sim_physical"

PRETRAIN_INIT_PATH = None
RESUME_MIGRATE_PATH = None

# ============================================================
# training config
# ============================================================
BATCH_SIZE = 4
NUM_EPOCHS = 100
USE_X0_INPUT = True

T_STAGES = 8
INIT_PRIOR_SCALE = 0.02

LR = 1e-4
WEIGHT_DECAY = 1e-6
EARLY_STOP_PATIENCE = 20

LOSS_W_COMPLEX = 0.70
LOSS_W_MAG = 0.30
LOSS_W_MID = 0.10
LOSS_W_DC = 0.01

INPUT_KEY = "Y_input_norm"
TARGET_KEY = "X_gt_raw_norm"     # 推荐第一版先学 Img_Label
FALLBACK_TARGET_KEY = "X_gt_norm"


def complex_rmse_loss(x_hat, x_gt, eps: float = 1e-8):
    num = torch.sqrt(torch.sum(torch.abs(x_hat - x_gt) ** 2, dim=(-2, -1)) + eps)
    den = torch.sqrt(torch.sum(torch.abs(x_gt) ** 2, dim=(-2, -1)) + eps)
    return (num / den).mean()


def mag_rmse_loss(x_hat, x_gt, eps: float = 1e-8):
    a = x_hat.abs()
    b = x_gt.abs()
    num = torch.sqrt(torch.sum((a - b) ** 2, dim=(-2, -1)) + eps)
    den = torch.sqrt(torch.sum(b ** 2, dim=(-2, -1)) + eps)
    return (num / den).mean()


def data_consistency_loss(x_hat, y, op, eps: float = 1e-8):
    resid = op.forward(x_hat) - y
    return resid.abs().pow(2).mean() / (y.abs().pow(2).mean() + eps)


def batch_data_consistency_loss(x_hat, y, mask_batch, op_if_same=None, eps: float = 1e-8):
    if op_if_same is not None:
        return data_consistency_loss(x_hat, y, op_if_same, eps=eps)

    if mask_batch.dim() >= 4 and mask_batch.shape[1] == 1:
        mask_batch = mask_batch[:, 0]

    vals = []
    for i in range(y.shape[0]):
        op_i = MatlabLikeISAROp(mask_batch[i])
        vals.append(data_consistency_loss(x_hat[i:i + 1], y[i:i + 1], op_i, eps=eps))
    return torch.stack(vals).mean()


def load_pretrained_partially(model, ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    src_state = ckpt["model_state_dict"]
    dst_state = model.state_dict()

    skip_keywords = [
        "cond_encoder",
        "alpha0_mod",
        "gamma_scale_mod",
        "lam_mod",
        "eta_mod",
        "rho_mod",
    ]

    keep_state = {}
    skipped_shape = []
    skipped_name = []

    for k, v in src_state.items():
        if k not in dst_state:
            skipped_name.append((k, "missing_in_current_model"))
            continue
        if dst_state[k].shape != v.shape:
            skipped_shape.append((k, tuple(v.shape), tuple(dst_state[k].shape)))
            continue
        if any(s in k for s in skip_keywords):
            skipped_name.append((k, "skip_condition_related"))
            continue
        keep_state[k] = v

    msg = model.load_state_dict(keep_state, strict=False)

    print(f"[Partial Init] loaded tensor count = {len(keep_state)}")
    if skipped_shape:
        print("[Partial Init] shape-mismatch keys:")
        for item in skipped_shape[:20]:
            print(" ", item)
    if skipped_name:
        print("[Partial Init] skipped keys:")
        for item in skipped_name[:30]:
            print(" ", item)
    print("[Partial Init] load_state_dict message =", msg)
    return ckpt


def run_model_with_batch_masks(model, y, x0, mask_batch, use_x0_input=True, mask_ratio=None, snr_est=None):
    if mask_batch.dim() >= 4 and mask_batch.shape[1] == 1:
        mask_batch = mask_batch[:, 0]

    same_mask = bool(torch.all(mask_batch == mask_batch[0:1]).item())
    if same_mask:
        op = MatlabLikeISAROp(mask_batch[0])
        x_hat, stage_outputs, aux = model(
            y,
            op,
            X0=x0 if use_x0_input else None,
            return_all=True,
            mask_ratio=mask_ratio,
            snr_est=snr_est,
        )
        return x_hat, stage_outputs, aux, op

    x_hat_all = []
    aux_all = []
    stage_outputs_all = None

    for i in range(y.shape[0]):
        op_i = MatlabLikeISAROp(mask_batch[i])
        mr_i = mask_ratio[i:i + 1] if mask_ratio is not None else None
        snr_i = snr_est[i:i + 1] if snr_est is not None else None

        x_hat_i, stage_outputs_i, aux_i = model(
            y[i:i + 1],
            op_i,
            X0=x0[i:i + 1] if use_x0_input else None,
            return_all=True,
            mask_ratio=mr_i,
            snr_est=snr_i,
        )
        x_hat_all.append(x_hat_i)
        aux_all.append(aux_i)

        if stage_outputs_all is None:
            stage_outputs_all = [[] for _ in range(len(stage_outputs_i))]
        for t, xs in enumerate(stage_outputs_i):
            stage_outputs_all[t].append(xs)

    x_hat = torch.cat(x_hat_all, dim=0)
    stage_outputs = [torch.cat(v, dim=0) for v in stage_outputs_all] if stage_outputs_all is not None else []
    return x_hat, stage_outputs, aux_all, None


def evaluate(model, loader, device: str, use_x0_input=True):
    model.eval()
    stats = {
        "loss_total": 0.0,
        "loss_cpx": 0.0,
        "loss_mag": 0.0,
        "loss_mid": 0.0,
        "loss_dc": 0.0,
        "x0_mag": 0.0,
    }
    count = 0

    with torch.no_grad():
        for batch in loader:
            y = batch["Y"].to(device)
            x0 = batch["X0"].to(device)
            x_gt = batch["X_gt"].to(device)
            mask_batch = batch["mask"].to(device)
            mask_ratio, snr_est = extract_condition_info(batch, device)

            x_hat, stage_outputs, _, op_same = run_model_with_batch_masks(
                model, y, x0, mask_batch,
                use_x0_input=use_x0_input,
                mask_ratio=mask_ratio,
                snr_est=snr_est,
            )

            loss_cpx = complex_rmse_loss(x_hat, x_gt)
            loss_mag = mag_rmse_loss(x_hat, x_gt)

            if stage_outputs:
                mid_idx = len(stage_outputs) // 2
                loss_mid = LOSS_W_MID * mag_rmse_loss(stage_outputs[mid_idx], x_gt)
            else:
                loss_mid = torch.tensor(0.0, device=device)

            loss_dc = LOSS_W_DC * batch_data_consistency_loss(
                x_hat, y, mask_batch, op_if_same=op_same
            )
            loss_total = LOSS_W_COMPLEX * loss_cpx + LOSS_W_MAG * loss_mag + loss_mid + loss_dc

            stats["loss_total"] += loss_total.item()
            stats["loss_cpx"] += loss_cpx.item()
            stats["loss_mag"] += loss_mag.item()
            stats["loss_mid"] += loss_mid.item()
            stats["loss_dc"] += loss_dc.item()
            stats["x0_mag"] += mag_rmse_loss(x0, x_gt).item()
            count += 1

    for k in stats:
        stats[k] /= max(count, 1)
    return stats


def set_trainable_params(model):
    for p in model.parameters():
        p.requires_grad = True


def build_optimizer(model):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)


def build_scheduler(optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=LR * 0.01,
    )


def print_trainable_parameters(model, title=""):
    print(f"\n[Trainable Parameters @ {title}]")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(" ", name, tuple(p.shape))


def print_stage_summary(model):
    with torch.no_grad():
        print("stage params summary (base):")
        print(model.get_params_summary())
        for mr_s, snr_s in [(0.25, 3.0), (0.35, 8.0), (0.50, 12.0)]:
            print(f"params @ mr={mr_s}, snr={snr_s}dB:")
            print(model.get_params_summary(mask_ratio_sample=mr_s, snr_sample=snr_s))


def print_batch_summary(batch, save_dir):
    if "snr_db" in batch:
        snr_db = batch["snr_db"].float()
        print(
            "batch summary: "
            f"snr[min/mean/max]=({snr_db.min().item():.2f}, "
            f"{snr_db.mean().item():.2f}, {snr_db.max().item():.2f})"
        )
    print(f"sample path example: {batch['path'][0]}")
    print(f"target_key_used example: {batch['target_key_used'][0]}")
    print(f"mask keep ratio actual: {batch['mask'][0].float().mean().item():.4f}")
    print(f"save_dir = {Path(save_dir).resolve()}")


def extract_condition_info(batch, device):
    if "mask_ratio" in batch:
        mask_ratio = batch["mask_ratio"].float().to(device)
    elif "keep_ratio" in batch:
        mask_ratio = batch["keep_ratio"].float().to(device)
    else:
        mask = batch["mask"]
        if mask.dim() >= 4 and mask.shape[1] == 1:
            mask = mask[:, 0]
        mask_ratio = mask.float().mean(dim=(-2, -1)).to(device)

    if "snr_db" in batch:
        snr_est = batch["snr_db"].float().to(device)
    else:
        raise KeyError("sim conditioned training 需要 snr_db，但 batch 中没有这个字段")

    return mask_ratio, snr_est


def build_model_cfg_from_init(init_ckpt: Optional[Dict[str, Any]] = None):
    if init_ckpt is not None and "config" in init_ckpt:
        cfg = dict(init_ckpt["config"])
        cfg["T"] = cfg.get("T", T_STAGES)
        cfg["use_cnn_prior"] = True
        cfg["prior_hidden_ch"] = cfg.get("prior_hidden_ch", 8)
        cfg["init_prior_scale"] = cfg.get("init_prior_scale", INIT_PRIOR_SCALE)
        return cfg

    return {
        "T": 8,
        "init_lambda": 0.89,
        "init_eta": 0.5,
        "init_rho": 8e-3,
        "init_alpha0": 10.0,
        "init_gamma_scale": 0.35,
        "a": 1.01,
        "b": 1e-6,
        "c": 1.0,
        "d": 1e-6,
        "gamma_ema": 0.3,
        "gamma_min": 1e-8,
        "gamma_max": 1e8,
        "share_params": False,
        "use_cnn_prior": True,
        "prior_hidden_ch": 8,
        "eta_var": 1.0,
        "tau_unc": 0.15,
        "max_neigh_sum": 1.2,
        "cond_dim": 16,
        "cond_max_shift": 0.5,
        "alpha_nf_scale": 1.4826,
        "alpha_snr_scale": 10.0,
        "alpha_min": 0.01,
        "alpha_max": 50.0,
        "gamma_init_clamp_max": 10.0,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_set = MC2DFIRSimDataset(
        TRAIN_ROOT,
        input_key=INPUT_KEY,
        target_key=TARGET_KEY,
        fallback_target_key=FALLBACK_TARGET_KEY,
    )
    val_set = MC2DFIRSimDataset(
        VAL_ROOT,
        input_key=INPUT_KEY,
        target_key=TARGET_KEY,
        fallback_target_key=FALLBACK_TARGET_KEY,
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    init_ckpt = None
    if PRETRAIN_INIT_PATH is not None and os.path.exists(PRETRAIN_INIT_PATH):
        print(f"[Init] found pretrained checkpoint: {PRETRAIN_INIT_PATH}")
        init_ckpt = torch.load(PRETRAIN_INIT_PATH, map_location=device)

    model_cfg = build_model_cfg_from_init(init_ckpt)
    model = StageWiseFIRGEMNetSim(**model_cfg).to(device)

    if PRETRAIN_INIT_PATH is not None and os.path.exists(PRETRAIN_INIT_PATH):
        load_pretrained_partially(model, PRETRAIN_INIT_PATH, device=device)

    set_trainable_params(model)
    print_trainable_parameters(model, title="full")
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)

    start_epoch = 1
    best_score = float("inf")
    best_mag = float("inf")
    no_improve_count = 0

    if RESUME_MIGRATE_PATH is not None and os.path.exists(RESUME_MIGRATE_PATH):
        print(f"[Resume] loading: {RESUME_MIGRATE_PATH}")
        ckpt = torch.load(RESUME_MIGRATE_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_score = float(ckpt.get("best_score", best_score))
        best_mag = float(ckpt.get("best_mag", best_mag))
        no_improve_count = int(ckpt.get("no_improve_count", 0))

    first_batch = next(iter(train_loader))
    print_batch_summary(first_batch, save_dir)
    print_stage_summary(model)

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_cpx = 0.0
        train_mag = 0.0
        train_mid = 0.0
        train_dc = 0.0
        count = 0

        for batch in train_loader:
            y = batch["Y"].to(device)
            x0 = batch["X0"].to(device)
            x_gt = batch["X_gt"].to(device)
            mask_batch = batch["mask"].to(device)
            mask_ratio, snr_est = extract_condition_info(batch, device)

            x_hat, stage_outputs, _, op_same = run_model_with_batch_masks(
                model, y, x0, mask_batch,
                use_x0_input=USE_X0_INPUT,
                mask_ratio=mask_ratio,
                snr_est=snr_est,
            )

            loss_cpx = complex_rmse_loss(x_hat, x_gt)
            loss_mag = mag_rmse_loss(x_hat, x_gt)

            if stage_outputs:
                mid_idx = len(stage_outputs) // 2
                loss_mid = LOSS_W_MID * mag_rmse_loss(stage_outputs[mid_idx], x_gt)
            else:
                loss_mid = torch.tensor(0.0, device=device)

            loss_dc = LOSS_W_DC * batch_data_consistency_loss(
                x_hat, y, mask_batch, op_if_same=op_same
            )
            loss_total = LOSS_W_COMPLEX * loss_cpx + LOSS_W_MAG * loss_mag + loss_mid + loss_dc

            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss_total.item()
            train_cpx += loss_cpx.item()
            train_mag += loss_mag.item()
            train_mid += loss_mid.item()
            train_dc += loss_dc.item()
            count += 1

        scheduler.step()

        train_loss /= max(count, 1)
        train_cpx /= max(count, 1)
        train_mag /= max(count, 1)
        train_mid /= max(count, 1)
        train_dc /= max(count, 1)

        val_stats = evaluate(model, val_loader, device, use_x0_input=USE_X0_INPUT)
        improve_over_x0_mag = val_stats["x0_mag"] - val_stats["loss_mag"]
        score = val_stats["loss_total"]

        print(
            f"\n[Epoch {epoch:03d}] "
            f"train_total={train_loss:.6f} "
            f"train_cpx={train_cpx:.6f} "
            f"train_mag={train_mag:.6f} "
            f"train_mid={train_mid:.6f} "
            f"train_dc={train_dc:.6f} | "
            f"val_total={val_stats['loss_total']:.6f} "
            f"val_mag={val_stats['loss_mag']:.6f} "
            f"improve_over_x0_mag={improve_over_x0_mag:.6f}"
        )
        print_stage_summary(model)

        MIN_DELTA = 1e-4
        improved_score = score < (best_score - MIN_DELTA)
        improved_mag = val_stats["loss_mag"] < (best_mag - MIN_DELTA)
        any_improved = improved_score or improved_mag

        if improved_score:
            best_score = score
        if improved_mag:
            best_mag = val_stats["loss_mag"]

        if any_improved:
            no_improve_count = 0
        else:
            no_improve_count += 1

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": model_cfg,
            "use_x0_input": USE_X0_INPUT,
            "best_score": best_score,
            "best_mag": best_mag,
            "no_improve_count": no_improve_count,
            "train_root": TRAIN_ROOT,
            "val_root": VAL_ROOT,
            "pretrain_init_path": PRETRAIN_INIT_PATH,
            "resume_migrate_path": RESUME_MIGRATE_PATH,
            "input_key": INPUT_KEY,
            "target_key": TARGET_KEY,
            "fallback_target_key": FALLBACK_TARGET_KEY,
            "loss_note": {
                "LOSS_W_COMPLEX": LOSS_W_COMPLEX,
                "LOSS_W_MAG": LOSS_W_MAG,
                "LOSS_W_MID": LOSS_W_MID,
                "LOSS_W_DC": LOSS_W_DC,
            },
            "note": "sim-physical training; defaults aligned to physical FIR-GEM initialization",
        }

        if improved_score:
            torch.save(ckpt, save_dir / "best_score.pth")
            torch.save(ckpt, save_dir / "best.pth")
            print("saving best_score to:", (save_dir / "best_score.pth").resolve())

        if improved_mag:
            torch.save(ckpt, save_dir / "best_mag.pth")
            print("saving best_mag to:", (save_dir / "best_mag.pth").resolve())

        torch.save(ckpt, save_dir / "last.pth")
        print("saving last to:", (save_dir / "last.pth").resolve())

        summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_cpx": train_cpx,
            "train_mag": train_mag,
            "train_mid": train_mid,
            "train_dc": train_dc,
            "val": val_stats,
            "score": score,
            "improve_over_x0_mag": improve_over_x0_mag,
            "best_score": best_score,
            "best_mag": best_mag,
        }
        with open(save_dir / "last_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        if no_improve_count >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break


if __name__ == "__main__":
    main()
