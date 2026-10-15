import math
import torch
import torch.nn as nn
import torch.nn.functional as F



def conv_same_real(x, k):
    return F.conv2d(x, k, padding=(k.shape[-2] // 2, k.shape[-1] // 2))


def estimate_base_gamma(Y, nobs, rho_gamma=3.0, eps=1e-8):
    power_sum = Y.abs().pow(2).sum(dim=(-2, -1), keepdim=True)
    nobs = max(float(nobs), 1.0)
    power_mean_obs = power_sum / nobs
    return rho_gamma / (power_mean_obs + eps)


def build_learned_kappa_from_raw(raw_kappa, max_neigh_sum=1.2, eps=1e-8):
    k = F.softplus(raw_kappa) + eps
    k = torch.clamp(k, min=1e-6)   # 新增：防止 kappa 过度衰减

    neigh = k.clone()
    neigh[..., 1, 1] = 0.0

    neigh_sum = neigh.sum(dim=(-2, -1), keepdim=True)
    scale = torch.clamp(max_neigh_sum / (neigh_sum + eps), max=1.0)
    neigh = neigh * scale

    center = torch.zeros_like(neigh)
    center[..., 1, 1] = 1.0
    return center + neigh


def init_8n_kernel(diag_ratio=0.3):
    k = torch.tensor(
        [[diag_ratio, 1.0,        diag_ratio],
         [1.0,        1.0,        1.0],
         [diag_ratio, 1.0,        diag_ratio]],
        dtype=torch.float32,
    )
    return k.view(1, 1, 3, 3)


def inv_softplus(x):
    x = torch.as_tensor(x, dtype=torch.float32)
    return torch.log(torch.expm1(x).clamp_min(1e-20))


class TinyDeltaPrior(nn.Module):
    def __init__(self, in_ch=2, hidden_ch=8, out_ch=1, cond_dim=0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_ch, out_ch, kernel_size=3, padding=1)

        self.use_film = cond_dim > 0
        if self.use_film:
            self.film1 = nn.Linear(cond_dim, hidden_ch * 2)
            self.film2 = nn.Linear(cond_dim, hidden_ch * 2)

            nn.init.zeros_(self.film1.weight)
            self.film1.bias.data[:hidden_ch] = 1.0
            self.film1.bias.data[hidden_ch:] = 0.0

            nn.init.zeros_(self.film2.weight)
            self.film2.bias.data[:hidden_ch] = 1.0
            self.film2.bias.data[hidden_ch:] = 0.0

    def forward(self, x, cond_vec=None):
        h = self.conv1(x)
        if self.use_film and cond_vec is not None:
            p1 = self.film1(cond_vec)
            g1, b1 = p1.chunk(2, dim=-1)
            h = g1.unsqueeze(-1).unsqueeze(-1) * h + b1.unsqueeze(-1).unsqueeze(-1)
        h = F.relu(h, inplace=True)

        h = self.conv2(h)
        if self.use_film and cond_vec is not None:
            p2 = self.film2(cond_vec)
            g2, b2 = p2.chunk(2, dim=-1)
            h = g2.unsqueeze(-1).unsqueeze(-1) * h + b2.unsqueeze(-1).unsqueeze(-1)
        h = F.relu(h, inplace=True)

        return self.conv3(h)


class ConditionEncoder(nn.Module):
    def __init__(self, cond_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, cond_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cond_dim, cond_dim),
            nn.ReLU(inplace=True),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, mask_ratio, snr_est):
        mr = mask_ratio.unsqueeze(-1).float()
        snr = snr_est.unsqueeze(-1).float() / 15.0
        cond_in = torch.cat([mr, snr], dim=-1)
        return self.net(cond_in)

class ConditionModulator(nn.Module):
    def __init__(self, cond_dim, max_shift=1.0):
        super().__init__()
        self.linear = nn.Linear(cond_dim, 1, bias=True)
        self.max_shift = max_shift
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, cond_vec):
        return self.max_shift * torch.tanh(self.linear(cond_vec).squeeze(-1))


class FIRGEMStage(nn.Module):
    """一个 stage = 一轮 2D-FIR-GEM 更新 + 一个很浅的 CNN Delta 先验。"""
    def __init__(
        self,
        init_lambda=0.90,
        init_eta=1.0,
        init_rho=1e-2,
        init_kappa=None,
        eps=1e-8,
        use_cnn_prior=False,
        prior_hidden_ch=8,
        init_prior_scale=0.02,
        eta_var=1.0,
        tau_unc=0.15,
        max_neigh_sum=1.2,
        cond_dim=0,
        cond_max_shift=1.0,
    ):
        super().__init__()
        self.eps = eps
        self.eta_var = float(eta_var)
        self.tau_unc = float(tau_unc)
        self.max_neigh_sum = float(max_neigh_sum)
        self.use_cnn_prior = bool(use_cnn_prior)
        self.cond_dim = int(cond_dim)

        init_lambda = float(min(max(init_lambda, 1e-3), 1 - 1e-3))
        self.raw_lambda = nn.Parameter(
            torch.tensor(math.log(init_lambda / (1 - init_lambda)), dtype=torch.float32)
        )
        self.raw_eta = nn.Parameter(inv_softplus(float(init_eta)))
        self.raw_rho = nn.Parameter(inv_softplus(float(init_rho)))

        if init_kappa is None:
            init_kappa = init_8n_kernel(diag_ratio=0.3)
        self.raw_kappa = nn.Parameter(init_kappa.clone())

        # 只初始化一次，保留 FiLM 版
        if self.use_cnn_prior:
            self.delta_prior = TinyDeltaPrior(
                in_ch=2,
                hidden_ch=prior_hidden_ch,
                out_ch=1,
                cond_dim=self.cond_dim,
            )
        else:
            self.delta_prior = None

        # 条件调制器：只做小偏移
        if self.cond_dim > 0:
            self.lam_mod = ConditionModulator(self.cond_dim, max_shift=cond_max_shift)
            self.eta_mod = ConditionModulator(self.cond_dim, max_shift=cond_max_shift)
            self.rho_mod = ConditionModulator(self.cond_dim, max_shift=cond_max_shift)
        else:
            self.lam_mod = None
            self.eta_mod = None
            self.rho_mod = None

    def forward(
        self,
        X,
        U,
        gamma,
        Y,
        op,
        a,
        b,
        c,
        d,
        gamma_ema,
        gamma_min,
        gamma_max,
        cond_vec=None,
    ):
        dtype = X.real.dtype
        device = X.device

        # ===== 只在 raw 参数上加 bounded 小偏移 =====
        if cond_vec is not None and self.lam_mod is not None:
            B = cond_vec.shape[0]

            lam_shift = self.lam_mod(cond_vec)   # (B,)
            eta_shift = self.eta_mod(cond_vec)   # (B,)
            rho_shift = self.rho_mod(cond_vec)   # (B,)

            lam = torch.sigmoid(self.raw_lambda + lam_shift).view(B, 1, 1).to(device=device, dtype=dtype)
            eta = (F.softplus(self.raw_eta + eta_shift) + self.eps).view(B, 1, 1).to(device=device, dtype=dtype)
            rho = (F.softplus(self.raw_rho + rho_shift) + self.eps).view(B, 1, 1).to(device=device, dtype=dtype)
        else:
            lam = torch.sigmoid(self.raw_lambda).to(device=device, dtype=dtype)
            eta = (F.softplus(self.raw_eta) + self.eps).to(device=device, dtype=dtype)
            rho = (F.softplus(self.raw_rho) + self.eps).to(device=device, dtype=dtype)

        kappa = build_learned_kappa_from_raw(
            self.raw_kappa,
            max_neigh_sum=self.max_neigh_sum,
            eps=self.eps
        ).to(device=device, dtype=dtype)

        alpha = F.softplus(U) + self.eps

        # ===== Delta 先验：公式不改，只允许 FiLM 调制 CNN prior =====
        if self.delta_prior is not None:
            alpha_in = alpha / (alpha.mean(dim=(-2, -1), keepdim=True) + self.eps)
            xpow = X.abs().pow(2).unsqueeze(1).to(dtype=dtype)
            xpow = xpow / (xpow.mean(dim=(-2, -1), keepdim=True) + self.eps)
            prior_in = torch.cat([alpha_in, xpow], dim=1)
            Delta = F.softplus(self.delta_prior(prior_in, cond_vec=cond_vec)) + self.eps
        else:
            Delta = conv_same_real(alpha, kappa) + self.eps

        # ===== X-step：完全保留原版 =====
        Grad = op.adjoint(op.forward(X) - Y)
        L_lip = op.lipschitz()

        Lt = eta * L_lip
        Den = Delta.squeeze(1) + gamma * Lt
        Den = Den.clamp_min(self.eps)

        Phi_x = 1.0 / Den
        X_tilde = gamma * (Lt * X - Grad) / Den
        X_new = lam * X_tilde + (1.0 - lam) * X

        # ===== alpha-step：完全保留原版 =====
        P_hat = X_new.abs().pow(2) + self.eta_var * Phi_x

        kappa_rot = torch.flip(kappa, dims=(-2, -1))
        grad_alpha = conv_same_real((1.0 / Delta) - P_hat.unsqueeze(1), kappa_rot)
        grad_alpha = grad_alpha + (a - 1.0) / alpha.clamp_min(self.eps) - b
        grad_U = grad_alpha * torch.sigmoid(U)

        if rho.dim() == 3:
            U_new = U + rho.unsqueeze(1) * grad_U
        else:
            U_new = U + rho * grad_U

        # ===== gamma-step：完全保留原版 =====
        Res = Y - op.forward(X_new)
        data_error = Res.abs().pow(2).sum(dim=(-2, -1), keepdim=True)
        unc_term = self.tau_unc * L_lip * Phi_x.sum(dim=(-2, -1), keepdim=True)
        gamma_map = (op.nobs() + c - 1.0) / (data_error + unc_term + d)
        gamma_map = gamma_map.real
        gamma_new = gamma_ema * gamma + (1.0 - gamma_ema) * gamma_map
        gamma_new = torch.clamp(gamma_new, min=gamma_min, max=gamma_max)

        aux = {
            "lambda": lam.detach(),
            "eta": eta.detach(),
            "rho": rho.detach(),
            "kappa": kappa.detach(),
            "Delta_mean": Delta.mean().detach(),
            "Phi_x_mean": Phi_x.mean().detach(),
        }
        return X_new, U_new, gamma_new, aux

class StageWiseFIRGEMNet(nn.Module):
    """stage-wise unfolding + 可选轻量 CNN Delta 先验 + 安全条件化。"""
    def __init__(
        self,
        T=8,
        init_lambda=0.90,
        init_eta=1.0,
        init_rho=1e-1,
        init_alpha0=1.0,
        init_gamma_scale=1.0,
        a=1.1,
        b=1e-6,
        c=1.0,
        d=1e-6,
        eps=1e-8,
        gamma_ema=0.2,
        gamma_min=1e-8,
        gamma_max=1e8,
        share_params=False,
        use_cnn_prior=False,
        prior_hidden_ch=8,
        init_prior_scale=0.05,
        cond_dim=16,
        cond_max_shift=0.5,
    ):
        super().__init__()
        self.T = T
        self.eps = eps
        self.gamma_ema = gamma_ema
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.share_params = share_params
        self.cond_dim = int(cond_dim)

        self.register_buffer("a", torch.tensor(float(a), dtype=torch.float32))
        self.register_buffer("b", torch.tensor(float(b), dtype=torch.float32))
        self.register_buffer("c", torch.tensor(float(c), dtype=torch.float32))
        self.register_buffer("d", torch.tensor(float(d), dtype=torch.float32))

        self.raw_alpha0 = nn.Parameter(inv_softplus(float(init_alpha0)))
        self.raw_gamma_scale = nn.Parameter(inv_softplus(float(init_gamma_scale)))

        # 全局条件化：alpha0 / gamma_scale
        if self.cond_dim > 0:
            self.cond_encoder = ConditionEncoder(self.cond_dim)
            self.alpha0_mod = ConditionModulator(self.cond_dim, max_shift=cond_max_shift)
            self.gamma_scale_mod = ConditionModulator(self.cond_dim, max_shift=cond_max_shift)
        else:
            self.cond_encoder = None
            self.alpha0_mod = None
            self.gamma_scale_mod = None

        stage_kwargs = dict(
            init_lambda=init_lambda,
            init_eta=init_eta,
            init_rho=init_rho,
            init_kappa=init_8n_kernel(diag_ratio=0.3),
            eps=eps,
            use_cnn_prior=use_cnn_prior,
            prior_hidden_ch=prior_hidden_ch,
            init_prior_scale=init_prior_scale,
            cond_dim=self.cond_dim,
            cond_max_shift=cond_max_shift,
        )

        if share_params:
            self.shared_stage = FIRGEMStage(**stage_kwargs)
            self.stages = None
        else:
            self.stages = nn.ModuleList([FIRGEMStage(**stage_kwargs) for _ in range(T)])
            self.shared_stage = None

    def _get_stage(self, t):
        return self.shared_stage if self.share_params else self.stages[t]

    def forward(self, Y, op, X0=None, return_all=False, mask_ratio=None, snr_est=None):
        X = op.adjoint(Y) if X0 is None else X0

        B, H, W = X.shape
        dtype = X.real.dtype
        device = X.device

        # ===== 条件编码 =====
        cond_vec = None
        if self.cond_encoder is not None and mask_ratio is not None and snr_est is not None:
            cond_vec = self.cond_encoder(mask_ratio.to(device), snr_est.to(device))

        # ===== alpha0 / gamma_scale：只加 bounded 小偏移 =====
        if cond_vec is not None and self.alpha0_mod is not None:
            alpha0 = (F.softplus(self.raw_alpha0 + self.alpha0_mod(cond_vec)) + self.eps).view(B, 1, 1).to(device=device, dtype=dtype)
            gamma_scale = (F.softplus(self.raw_gamma_scale + self.gamma_scale_mod(cond_vec)) + self.eps).view(B, 1, 1).to(device=device, dtype=dtype)
        else:
            alpha0 = (F.softplus(self.raw_alpha0) + self.eps).to(device=device, dtype=dtype)
            gamma_scale = (F.softplus(self.raw_gamma_scale) + self.eps).to(device=device, dtype=dtype)

        # ===== 初始化：完全保留原版 =====
        if alpha0.dim() == 3:
            alpha0_map = alpha0.unsqueeze(1).expand(B, 1, H, W)
        else:
            alpha0_map = torch.ones((B, 1, H, W), device=device, dtype=dtype) * alpha0

        U = torch.clamp(inv_softplus(alpha0_map), -20.0, 20.0)

        base_gamma = estimate_base_gamma(Y, op.nobs(), self.eps).to(device=device, dtype=dtype)
        if gamma_scale.dim() == 3:
            gamma = gamma_scale * base_gamma
        else:
            gamma = gamma_scale.view(1, 1, 1) * base_gamma

        gamma = torch.clamp(gamma.real, min=self.gamma_min, max=self.gamma_max)
        gamma = torch.clamp(gamma, max=10.0)

        stage_outputs = []
        stage_aux = []
        for t in range(self.T):
            stage = self._get_stage(t)
            X, U, gamma, aux = stage(
                X, U, gamma, Y, op,
                a=self.a,
                b=self.b,
                c=self.c,
                d=self.d,
                gamma_ema=self.gamma_ema,
                gamma_min=self.gamma_min,
                gamma_max=self.gamma_max,
                cond_vec=cond_vec,
            )
            stage_outputs.append(X)
            stage_aux.append(aux)

        if return_all:
            return X, stage_outputs, stage_aux
        return X

    def get_params_summary(self, mask_ratio_sample=None, snr_sample=None):
        device = self.raw_alpha0.device
        lines = []

        if mask_ratio_sample is None or snr_sample is None or self.cond_encoder is None:
            a0 = (F.softplus(self.raw_alpha0) + 1e-8).item()
            gs = (F.softplus(self.raw_gamma_scale) + 1e-8).item()
            lines.append(f"global(base): alpha0={a0:.4e}, gamma_scale={gs:.4e}")

            for t in range(self.T):
                s = self._get_stage(t)
                lam = torch.sigmoid(s.raw_lambda).item()
                eta = (F.softplus(s.raw_eta) + 1e-8).item()
                rho = (F.softplus(s.raw_rho) + 1e-8).item()
                kc = s.raw_kappa[0, 0, 1, 1].item()
                lines.append(
                    f"stage {t+1}: lambda={lam:.4f}, eta={eta:.4f}, "
                    f"rho={rho:.6f}, kappa_center={kc:.4e}"
                )
            return "\n".join(lines)

        mr = torch.tensor([mask_ratio_sample], device=device, dtype=torch.float32)
        snr = torch.tensor([snr_sample], device=device, dtype=torch.float32)

        with torch.no_grad():
            cv = self.cond_encoder(mr, snr)

            a0 = (F.softplus(self.raw_alpha0 + self.alpha0_mod(cv)) + 1e-8).item()
            gs = (F.softplus(self.raw_gamma_scale + self.gamma_scale_mod(cv)) + 1e-8).item()
            lines.append(f"global(mr={mask_ratio_sample:.2f}, snr={snr_sample:.1f}): alpha0={a0:.4e}, gamma_scale={gs:.4e}")

            for t in range(self.T):
                s = self._get_stage(t)
                lam = torch.sigmoid(s.raw_lambda + s.lam_mod(cv)).item() if s.lam_mod is not None else torch.sigmoid(s.raw_lambda).item()
                eta = (F.softplus(s.raw_eta + s.eta_mod(cv)) + 1e-8).item() if s.eta_mod is not None else (F.softplus(s.raw_eta) + 1e-8).item()
                rho = (F.softplus(s.raw_rho + s.rho_mod(cv)) + 1e-8).item() if s.rho_mod is not None else (F.softplus(s.raw_rho) + 1e-8).item()
                kc = s.raw_kappa[0, 0, 1, 1].item()
                lines.append(
                    f"stage {t+1}: lambda={lam:.4f}, eta={eta:.4f}, "
                    f"rho={rho:.6f}, kappa_center={kc:.4e}"
                )

        return "\n".join(lines)