from pathlib import Path
from scipy.io import loadmat
import numpy as np
import torch
from torch.utils.data import Dataset


def to_numpy_complex(x):
    x = np.asarray(x)
    if np.iscomplexobj(x):
        return x.astype(np.complex64, copy=False)
    return x.astype(np.complex64, copy=False)


def to_torch_complex(x):
    x = to_numpy_complex(x)
    return torch.from_numpy(np.ascontiguousarray(x))


def squeeze_2d(x):
    x = np.asarray(x)
    x = np.squeeze(x)
    if x.ndim != 2:
        raise ValueError(f"期望二维数组，实际 shape={x.shape}")
    return x


def rescale_complex_by_max(ref, target, eps=1e-12):
    ref = np.asarray(ref)
    target = np.asarray(target)

    ref_max = np.max(np.abs(ref)) + eps
    tgt_max = np.max(np.abs(target)) + eps
    target = target / tgt_max * ref_max
    return target


def get_optional_field(mat, key, default=None):
    if key in mat:
        return mat[key]
    return default


def safe_scalar_from_mat(mat, key, default=None):
    if key not in mat:
        return default
    arr = np.asarray(mat[key]).squeeze()
    if arr.size == 0:
        return default
    try:
        return float(arr)
    except Exception:
        return default


def safe_string_from_mat(mat, key, default=""):
    if key not in mat:
        return default

    arr = mat[key]

    try:
        # MATLAB 字符串常见情况
        if isinstance(arr, np.ndarray):
            arr = np.squeeze(arr)
            if arr.dtype.kind in ("U", "S"):
                return str(arr.tolist())
            if arr.dtype == object and arr.size == 1:
                return str(arr.item())
        return str(arr)
    except Exception:
        return default


class MC2DFIRDataset(Dataset):
    def __init__(self, root, input_key="Y_input_norm", gt_rescale_mode="none"):
        self.files = sorted(Path(root).glob("*.mat"))
        self.input_key = input_key
        self.gt_rescale_mode = gt_rescale_mode

        if len(self.files) == 0:
            raise RuntimeError(f"在 {root} 没找到 .mat 文件")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mat_path = self.files[idx]
        mat = loadmat(mat_path)

        # ------------------------------
        # 必需字段
        # ------------------------------
        if self.input_key not in mat:
            raise KeyError(f"{mat_path} 中缺少输入字段: {self.input_key}")
        if "X0_norm" not in mat:
            raise KeyError(f"{mat_path} 中缺少字段: X0_norm")
        if "X_gt_norm" not in mat:
            raise KeyError(f"{mat_path} 中缺少字段: X_gt_norm")
        if "Mask_single" not in mat:
            raise KeyError(f"{mat_path} 中缺少字段: Mask_single")

        Y_np = squeeze_2d(mat[self.input_key])
        X0_np = squeeze_2d(mat["X0_norm"])
        X_gt_np = squeeze_2d(mat["X_gt_norm"])
        mask_np = squeeze_2d(mat["Mask_single"]).astype(np.float32, copy=False)

        # ------------------------------
        # 可选 teacher
        # 兼容 generate_teacher = false 的情况
        # ------------------------------
        X_teacher_raw = get_optional_field(mat, "X_teacher_norm", default=None)
        if X_teacher_raw is None:
            X_teacher_np = np.zeros_like(X0_np, dtype=np.complex64)
            has_teacher = 0.0
        else:
            X_teacher_arr = np.asarray(X_teacher_raw)
            if X_teacher_arr.size == 0:
                X_teacher_np = np.zeros_like(X0_np, dtype=np.complex64)
                has_teacher = 0.0
            else:
                X_teacher_np = squeeze_2d(X_teacher_arr)
                has_teacher = 1.0

        # ------------------------------
        # GT 重标定
        # ------------------------------
        if self.gt_rescale_mode == "match_x0":
            X_gt_np = rescale_complex_by_max(X0_np, X_gt_np)
        elif self.gt_rescale_mode == "self_norm":
            X_gt_np = rescale_complex_by_max(np.ones_like(X_gt_np), X_gt_np)
        elif self.gt_rescale_mode == "none":
            pass
        else:
            raise ValueError(f"unknown gt_rescale_mode: {self.gt_rescale_mode}")

        # ------------------------------
        # 转 torch
        # ------------------------------
        Y = to_torch_complex(Y_np)
        X0 = to_torch_complex(X0_np)
        X_teacher = to_torch_complex(X_teacher_np)
        X_gt = to_torch_complex(X_gt_np)
        mask = torch.from_numpy(np.ascontiguousarray(mask_np)).float()

        sample = {
            "Y": Y,
            "X0": X0,
            "X_teacher": X_teacher,
            "X_gt": X_gt,
            "mask": mask,
            "has_teacher": torch.tensor(has_teacher, dtype=torch.float32),
            "path": str(mat_path),
        }

        # ------------------------------
        # 可选附加字段
        # ------------------------------
        snr_db = safe_scalar_from_mat(mat, "snr_db", default=None)
        if snr_db is not None:
            sample["snr_db"] = torch.tensor(snr_db, dtype=torch.float32)

        scale = safe_scalar_from_mat(mat, "scale", default=None)
        if scale is not None:
            sample["scale"] = torch.tensor(scale, dtype=torch.float32)

        keep_ratio = safe_scalar_from_mat(mat, "keep_ratio", default=None)
        if keep_ratio is not None:
            keep_ratio_t = torch.tensor(keep_ratio, dtype=torch.float32)
        else:
            keep_ratio_t = torch.tensor(float(mask.mean().item()), dtype=torch.float32)

        sample["keep_ratio"] = keep_ratio_t
        sample["mask_ratio"] = keep_ratio_t

        # 调试字段：方便观察 GT / X0 尺度关系
        xgt_peak = np.max(np.abs(np.asarray(mat["X_gt_norm"]))) + 1e-12
        x0_peak = np.max(np.abs(np.asarray(mat["X0_norm"]))) + 1e-12
        sample["gt_x0_ratio_before_fix"] = torch.tensor(
            float(xgt_peak / x0_peak), dtype=torch.float32
        )

        return sample