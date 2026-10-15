function generate_mc2dfir_backend_dataset_sim_physical()
% =========================================================================
% 面向 PyTorch / 深度展开训练的“纯仿真物理数据”生成器
%
% 设计目标：
% 1) 保留你第一个文件的数据集保存格式（train/val/test + sample_xxxxx.mat）
% 2) 不再读取实测 DAT/MAT，也不再从实测数据构造 pseudo GT
% 3) 每个样本都按第二个文件的“物理散射点 -> HRRP -> RD”链路独立生成
% 4) 关键修改：X_gt_used_base / X_gt_used_i 直接使用 Img_Ideal
% 5) 同时保留 X_gt_raw_base / X_gt_raw_i = Img_Label，便于后续对照分析
%
% 说明：
% - 如果你希望 X_gt_raw 也等于 Img_Ideal，只需把 sample 中那一行改成：
%       X_gt_raw_i = Img_Ideal;
% - 当前观测仍按“频域观测 + 稀疏采样 + 复高斯噪声”的格式保存，
%   尽量兼容你现有的 PyTorch 读取代码。
% =========================================================================

    clear; clc; close all;

    %% =========================
    % 0) 用户配置
    % =========================
    cfg = struct();

    % 输出路径
    cfg.save_root = 'D:\document\yanyi\电子书\论文\贝叶斯算法的寻优\2DFIRclaude\仿真数据训练';

    % 数据量
    cfg.num_train = 800;
    cfg.num_val   = 100;
    cfg.num_test  = 100;

    % 随机种子
    cfg.seed = 2026;

    % ---------- SNR ----------
    cfg.snr_min = 0;
    cfg.snr_max = 15;
    cfg.noise_power_mode = 'mean';   % 'mean' | 'peak'

    % ---------- 掩膜：每样本独立 ----------
    cfg.use_random_mask = true;
    cfg.mask_keep_ratio_min = 0.25;
    cfg.mask_keep_ratio_max = 0.50;

    % ---------- 归一化 ----------
    cfg.scale_mode = 'obs_rms';      % 'gt_percentile' | 'gt_max' | 'obs_rms'
    cfg.scale_percentile = 99.5;
    cfg.scale_eps = 1e-8;

    % ---------- 是否画预览 ----------
    cfg.show_preview = true;

    % ---------- 是否生成 teacher ----------
    cfg.generate_teacher = false;

    % ---------- 可选：加入轻微运动扰动 ----------
    cfg.enable_motion_perturb = false;
    cfg.motion_prob = 0.35;
    cfg.v_range = [-0.10, 0.10];
    cfg.a_range = [-0.01, 0.01];

    % ---------- 物理散射场参数（来自第二个文件思路） ----------
    cfg.scene.numBlocks   = 30;
    cfg.scene.ptsPerBlock = 3;
    cfg.scene.body_len    = 14;
    cfg.scene.wing_span   = 18;
    cfg.scene.tail_span   = 7;
    cfg.scene.wing_y      = 2.5;
    cfg.scene.tail_y      = -5.0;

    cfg.scene.snap_to_grid = true;
    cfg.scene.snap_ratio_x = 0.85;
    cfg.scene.snap_ratio_y = 0.85;
    cfg.scene.jitter_center = 0.08;
    cfg.scene.jitter_point  = 0.05;

    % 每个样本可随机浮动一点，增加多样性
    cfg.scene_randomize = true;
    cfg.scene.numBlocks_range = [24, 36];
    cfg.scene.body_len_range  = [12, 16];
    cfg.scene.wing_span_range = [16, 22];
    cfg.scene.tail_span_range = [5, 9];
    cfg.scene.wing_y_range    = [1.5, 3.5];
    cfg.scene.tail_y_range    = [-6.0, -4.0];
    cfg.scene.snap_ratio_range = [0.75, 0.95];
    cfg.scene.jitter_center_range = [0.04, 0.10];
    cfg.scene.jitter_point_range  = [0.03, 0.08];

    % 可选幅相扰动（直接作用于物理散射中心反射系数）
    cfg.sigma_amp_jitter_std   = 0.08;
    cfg.sigma_phase_jitter_std = 0.10;

    % 如果后面要生成 teacher，可在这里补 opts
    cfg.teacher_opts = struct();

    %% =========================
    % 1) 初始化
    % =========================
    rng(cfg.seed);

    if cfg.generate_teacher && ~exist('run_MC_2D_FIR_GEM', 'file')
        error('未找到 run_MC_2D_FIR_GEM.m，请先加入 MATLAB 路径。');
    end

    radar = init_radar_sim();
    total_num = cfg.num_train + cfg.num_val + cfg.num_test;

    prepare_save_dirs(cfg.save_root);

    fprintf('================ Physical Simulation Training Dataset Generator ================\n');
    fprintf('Step 1: 构造参考仿真样本（Img_Label / Img_Ideal / full-aperture HRRP）...\n');

    %% =========================
    % 2) 构造参考 base 样本
    % =========================
    [Y_clean_base, X_gt_raw_base, X_gt_used_base, clean_aux] = generate_physical_training_sample(radar, cfg);

    fprintf('  -> Y_clean_base   size = [%d, %d]\n', size(Y_clean_base,1), size(Y_clean_base,2));
    fprintf('  -> X_gt_raw_base  size = [%d, %d]\n', size(X_gt_raw_base,1), size(X_gt_raw_base,2));
    fprintf('  -> X_gt_used_base size = [%d, %d]\n', size(X_gt_used_base,1), size(X_gt_used_base,2));

    %% =========================
    % 3) 预览
    % =========================
    if cfg.show_preview
        show_preview_figure_sim(Y_clean_base, X_gt_raw_base, X_gt_used_base);
    end

    %% =========================
    % 4) 保存 meta
    % =========================
    dataset_meta = struct();
    dataset_meta.cfg = cfg;
    dataset_meta.radar = radar;
    dataset_meta.base_info = struct( ...
        'data_mode', 'pure_sim_physical', ...
        'note', ['纯仿真训练集：物理散射点 -> full-aperture HRRP -> Img_Label；', ...
                 '其中 X_gt_used 使用 Img_Ideal，X_gt_raw 使用 Img_Label']);

    save(fullfile(cfg.save_root, 'dataset_meta.mat'), 'dataset_meta', '-v7');

    %% =========================
    % 5) 批量生成样本
    % =========================
    fprintf('Step 2: 开始生成 train / val / test 样本...\n');

    keep_ratio_hist = zeros(total_num, 1);
    snr_hist = zeros(total_num, 1);

    for i = 1:total_num

        % ---- 5.1 物理链路生成单样本 ----
        [Y_clean_i, X_gt_raw_i, X_gt_used_i, aug_info] = generate_physical_training_sample(radar, cfg);
        source_type = 'sim_physical';

        % ---- 5.2 可选：加入轻微运动扰动 ----
        motion_params = struct('enabled', false, 'v', 0.0, 'a', 0.0);
        if cfg.enable_motion_perturb && (rand() < cfg.motion_prob)
            motion_params.enabled = true;
            motion_params.v = rand_uniform(cfg.v_range(1), cfg.v_range(2));
            motion_params.a = rand_uniform(cfg.a_range(1), cfg.a_range(2));
            Y_clean_i = apply_motion_phase_error_fc(Y_clean_i, radar, motion_params.v, motion_params.a);
        end

        % ---- 5.3 加噪 ----
        snr_db = rand_uniform(cfg.snr_min, cfg.snr_max);
        Y_noisy = add_complex_noise(Y_clean_i, snr_db, cfg.noise_power_mode);

        % ---- 5.4 每样本独立 Mask ----
        [Mask_single, keep_ratio, pulse_mask] = generate_random_mask(radar, cfg);
        keep_ratio_hist(i) = keep_ratio;
        snr_hist(i) = snr_db;

        if i == 1
            fprintf('Mask size = [%d, %d], kept pulses = %d / %d\n', ...
                size(Mask_single,1), size(Mask_single,2), sum(pulse_mask), radar.Na);
        end

        % ---- 5.5 定义当前样本算子：与原第一份文件保持一致 ----
        PhiH = @(Y) fftshift(fft(ifft(Y, [], 1), [], 2), 2);
        Phi  = @(X) fft(ifft(ifftshift(X, 2), [], 2), [], 1) .* Mask_single;

        % ---- 5.6 观测 + 初值 ----
        Y_input = Y_noisy .* Mask_single;
        X0 = PhiH(Y_input);

        % ---- 5.7 归一化 ----
        scale = compute_scale(Y_input, X_gt_raw_i, cfg, Mask_single);

        Y_input_norm      = Y_input     / scale;
        Y_clean_norm      = Y_clean_i   / scale;
        X0_norm           = X0          / scale;
        X_gt_norm         = X_gt_used_i / scale;
        X_gt_raw_norm     = X_gt_raw_i  / scale;
        X_gt_abs_norm     = abs(X_gt_norm);
        X_gt_raw_abs_norm = abs(X_gt_raw_norm);

        % ---- 5.8 teacher（可选） ----
        if cfg.generate_teacher
            teacher_opts = cfg.teacher_opts;
            teacher_opts.Nobs = nnz(Mask_single);

            obs = Y_input_norm(Mask_single ~= 0);
            teacher_opts.gamma0 = 3.0 / max(mean(abs(obs).^2), 1e-6);
            teacher_opts.X0 = X0_norm;

            alpha_scalar = 1.0 / max(mean(abs(X0_norm(:)).^2), 1e-4);
            teacher_opts.alpha0 = alpha_scalar * ones(size(X0_norm));

            Phi_teacherH = @(Y) fftshift(fft(ifft(Y, [], 1), [], 2), 2);
            Phi_teacher  = @(X) fft(ifft(ifftshift(X, 2), [], 2), [], 1) .* Mask_single;

            [X_teacher_norm, hyper_teacher, info_teacher] = run_MC_2D_FIR_GEM( ...
                Y_input_norm, Phi_teacher, Phi_teacherH, teacher_opts);
        else
            X_teacher_norm = [];
            hyper_teacher = [];
            info_teacher = [];
        end

        % ---- 5.9 划分数据集 ----
        [split_name, idx_local] = get_split_name(i, cfg.num_train, cfg.num_val, cfg.num_test);
        save_dir = fullfile(cfg.save_root, split_name);
        filename = fullfile(save_dir, sprintf('sample_%05d.mat', idx_local));

        % ---- 5.10 保存 ----
        save(filename, ...
            'Y_input_norm', ...
            'Y_clean_norm', ...
            'X0_norm', ...
            'X_gt_norm', ...
            'X_gt_raw_norm', ...
            'X_gt_abs_norm', ...
            'X_gt_raw_abs_norm', ...
            'X_teacher_norm', ...
            'Mask_single', ...
            'pulse_mask', ...
            'snr_db', ...
            'scale', ...
            'keep_ratio', ...
            'source_type', ...
            'motion_params', ...
            'aug_info', ...
            'hyper_teacher', ...
            'info_teacher', ...
            '-v7');

        if mod(i, 50) == 0 || i == total_num
            fprintf('  已生成 %4d / %4d | src=%s | snr=%.2f dB | keep=%.3f\n', ...
                i, total_num, source_type, snr_db, keep_ratio);
        end
    end

    %% =========================
    % 6) 保存参考与统计
    % =========================
    generation_summary = struct();
    generation_summary.source_type       = 'sim_physical';
    generation_summary.snr_min_actual    = min(snr_hist);
    generation_summary.snr_max_actual    = max(snr_hist);
    generation_summary.keep_min_actual   = min(keep_ratio_hist);
    generation_summary.keep_max_actual   = max(keep_ratio_hist);
    generation_summary.keep_mean_actual  = mean(keep_ratio_hist);

    save(fullfile(cfg.save_root, 'groundtruth_reference.mat'), ...
        'Y_clean_base', 'X_gt_raw_base', 'X_gt_used_base', ...
        'clean_aux', 'radar', 'generation_summary', '-v7');

    fprintf('-------------------------------------------------------------------\n');
    fprintf('数据源类型    : sim_physical\n');
    fprintf('SNR 实际范围  : [%.2f, %.2f] dB\n', ...
        generation_summary.snr_min_actual, generation_summary.snr_max_actual);
    fprintf('采样率实际范围: [%.3f, %.3f], mean=%.3f\n', ...
        generation_summary.keep_min_actual, ...
        generation_summary.keep_max_actual, ...
        generation_summary.keep_mean_actual);
    fprintf('输出目录: %s\n', cfg.save_root);
    fprintf('===================================================================\n');
end


% =========================================================================
% 单样本：物理散射点 -> HRRP -> Img_Label / Img_Ideal
% 输出：
%   Y_clean_freq : 频域 clean observation（与原第一文件一致）
%   X_gt_raw     : Img_Label（full-aperture RD image）
%   X_gt_used    : Img_Ideal（理想散射图）
% =========================================================================
function [Y_clean_freq, X_gt_raw, X_gt_used, aux] = generate_physical_training_sample(radar, cfg)

    scene = sample_scene_cfg(cfg, radar);
    scatterers = generate_aircraft_like_physical_scatterers(scene);
    scatterers = jitter_scatterer_sigma(scatterers, cfg);

    % full-aperture HRRP（HRRP 域）
    HRRP_full = synthesize_hrrp_from_physical_scatterers(radar, scatterers);
    HRRP_full = HRRP_full / (max(abs(HRRP_full(:))) + eps);

    % full-aperture label（与第二个文件一致：由 HRRP 方位 FFT 得到）
    Img_Label = fftshift(fft(HRRP_full, [], 2), 2);
    Img_Label = Img_Label / (max(abs(Img_Label(:))) + eps);

    % 理想散射图（用户指定要作为 X_gt_used）
    Img_Ideal = rasterize_physical_scatterers(scatterers, radar);
    Img_Ideal = Img_Ideal / (max(abs(Img_Ideal(:))) + eps);

    % 与原第一文件对齐：观测保存在“频域 clean observation”
    Y_clean_freq = ifft(HRRP_full, [], 1);

    X_gt_raw  = Img_Label;
    X_gt_used = Img_Ideal;

    aux = struct();
    aux.scene = scene;
    aux.scatterers = scatterers;
    aux.HRRP_full = HRRP_full;
    aux.Img_Label = Img_Label;
    aux.Img_Ideal = Img_Ideal;
    aux.type = 'sim_physical';
end


% =========================================================================
% 雷达参数（按第二个文件的物理仿真风格）
% =========================================================================
function radar = init_radar_sim()
    radar = struct();
    radar.c   = 3e8;
    radar.fc  = 10e9;          % Hz
    radar.Bw  = 600e6;         % Hz
    radar.B   = radar.Bw;
    radar.PRF = 100;           % Hz
    radar.Nr  = 256;
    radar.Na  = 256;

    radar.lambda  = radar.c / radar.fc;
    radar.delta_r = radar.c / (2 * radar.Bw);
    radar.omega_r = 0.06;      % rad/s

    radar.t_fast = ((0:radar.Nr-1) - radar.Nr/2) / radar.Bw;
    radar.tm     = ((0:radar.Na-1) - radar.Na/2) / radar.PRF;

    % 兼容第一份文件中的相位补偿函数
    radar.delta_f  = radar.B / radar.Nr;
    radar.freq_vec = radar.fc + ((0:radar.Nr-1)' - radar.Nr/2) * radar.delta_f;
end


% =========================================================================
% 采样一个 scene 配置，增加样本多样性
% =========================================================================
function scene = sample_scene_cfg(cfg, radar)
    scene = cfg.scene;

    if cfg.scene_randomize
        scene.numBlocks   = randi(cfg.scene.numBlocks_range);
        scene.body_len    = rand_uniform(cfg.scene.body_len_range(1),  cfg.scene.body_len_range(2));
        scene.wing_span   = rand_uniform(cfg.scene.wing_span_range(1), cfg.scene.wing_span_range(2));
        scene.tail_span   = rand_uniform(cfg.scene.tail_span_range(1), cfg.scene.tail_span_range(2));
        scene.wing_y      = rand_uniform(cfg.scene.wing_y_range(1),    cfg.scene.wing_y_range(2));
        scene.tail_y      = rand_uniform(cfg.scene.tail_y_range(1),    cfg.scene.tail_y_range(2));
        scene.snap_ratio_x = rand_uniform(cfg.scene.snap_ratio_range(1), cfg.scene.snap_ratio_range(2));
        scene.snap_ratio_y = rand_uniform(cfg.scene.snap_ratio_range(1), cfg.scene.snap_ratio_range(2));
        scene.jitter_center = rand_uniform(cfg.scene.jitter_center_range(1), cfg.scene.jitter_center_range(2));
        scene.jitter_point  = rand_uniform(cfg.scene.jitter_point_range(1),  cfg.scene.jitter_point_range(2));
    end

    scene.dy_bin = radar.c / (2 * radar.Bw);
    scene.dx_bin = radar.lambda * radar.PRF / (2 * radar.omega_r * radar.Na);

    scene.block_dx = 0.8 * scene.dx_bin;
    scene.block_dy = 0.8 * scene.dy_bin;
end


% =========================================================================
% 创建目录
% =========================================================================
function prepare_save_dirs(save_root)
    if ~exist(save_root, 'dir')
        mkdir(save_root);
    end

    subdirs = {'train', 'val', 'test'};
    for k = 1:numel(subdirs)
        p = fullfile(save_root, subdirs{k});
        if ~exist(p, 'dir')
            mkdir(p);
        end
    end
end


% =========================================================================
% Aircraft-like physical scatterers（来自第二个文件的核心思路）
% =========================================================================
function scatterers = generate_aircraft_like_physical_scatterers(scene)

    local_patterns = {
        [0 0;  1 0;  0 1], ...
        [0 0; -1 0;  0 1], ...
        [0 0;  1 0;  0 -1], ...
        [0 0; -1 0;  0 -1], ...
        [0 0;  1 1; -1 0]  ...
    };

    % 将 numBlocks 按机身 / 左右翼 / 左右尾翼分配
    nb_body = max(8, round(scene.numBlocks * 0.40));
    nb_wing_each = max(4, round(scene.numBlocks * 0.20));
    nb_tail_each = max(2, floor((scene.numBlocks - nb_body - 2*nb_wing_each) / 2));

    used_blocks = nb_body + 2*nb_wing_each + 2*nb_tail_each;
    while used_blocks > scene.numBlocks && nb_body > 8
        nb_body = nb_body - 1;
        used_blocks = nb_body + 2*nb_wing_each + 2*nb_tail_each;
    end
    while used_blocks < scene.numBlocks
        nb_body = nb_body + 1;
        used_blocks = nb_body + 2*nb_wing_each + 2*nb_tail_each;
    end

    y_body = linspace(-scene.body_len/2, scene.body_len/2, nb_body).';
    x_body = 0.25 * scene.dx_bin * sin(linspace(-pi/2, pi/2, numel(y_body))).';

    x_wing = linspace(2, scene.wing_span/2, nb_wing_each).';
    y_wing = scene.wing_y + 0.25 * x_wing;

    x_left_wing  = -x_wing;
    y_left_wing  = y_wing;
    x_right_wing =  x_wing;
    y_right_wing =  y_wing;

    x_tail = linspace(1.5, scene.tail_span/2, nb_tail_each).';
    y_tail = scene.tail_y + 0.35 * x_tail;

    x_left_tail  = -x_tail;
    y_left_tail  = y_tail;
    x_right_tail =  x_tail;
    y_right_tail =  y_tail;

    xc = [x_body; x_left_wing; x_right_wing; x_left_tail; x_right_tail];
    yc = [y_body; y_left_wing; y_right_wing; y_left_tail; y_right_tail];

    scatterers = struct('x', {}, 'y', {}, 'sigma', {});
    count = 0;

    for b = 1:numel(xc)
        pat = local_patterns{randi(numel(local_patterns))};

        x0 = xc(b) + scene.jitter_center * scene.block_dx * randn();
        y0 = yc(b) + scene.jitter_center * scene.block_dy * randn();

        base_phase = 2 * pi * rand();

        for k = 1:scene.ptsPerBlock
            count = count + 1;

            xk = x0 + scene.block_dx * pat(k,1) + scene.jitter_point * scene.block_dx * randn();
            yk = y0 + scene.block_dy * pat(k,2) + scene.jitter_point * scene.block_dy * randn();

            if isfield(scene, 'snap_to_grid') && scene.snap_to_grid
                x_snap = round(xk / scene.dx_bin) * scene.dx_bin;
                y_snap = round(yk / scene.dy_bin) * scene.dy_bin;
                xk = (1 - scene.snap_ratio_x) * xk + scene.snap_ratio_x * x_snap;
                yk = (1 - scene.snap_ratio_y) * yk + scene.snap_ratio_y * y_snap;
            end

            amp = 0.8;
            phs = base_phase;
            sigma = amp * exp(1j * phs);

            scatterers(count).x = xk;
            scatterers(count).y = yk;
            scatterers(count).sigma = sigma;
        end
    end
end


% =========================================================================
% 对物理散射中心反射系数做轻微幅相扰动
% =========================================================================
function scatterers = jitter_scatterer_sigma(scatterers, cfg)
    for q = 1:numel(scatterers)
        amp_scale = 1 + cfg.sigma_amp_jitter_std * randn();
        amp_scale = max(amp_scale, 0.10);
        phs_scale = exp(1j * cfg.sigma_phase_jitter_std * randn());
        scatterers(q).sigma = scatterers(q).sigma * amp_scale * phs_scale;
    end
end


% =========================================================================
% 按论文 Section II 风格合成 HRRP
% =========================================================================
function HRRP = synthesize_hrrp_from_physical_scatterers(radar, scatterers)

    Nr = radar.Nr;
    Na = radar.Na;

    t_fast = radar.t_fast(:);
    tm = radar.tm(:).';

    HRRP = complex(zeros(Nr, Na));

    for q = 1:numel(scatterers)
        xq = scatterers(q).x;
        yq = scatterers(q).y;
        sigma_q = scatterers(q).sigma;

        sigma_prime = sigma_q .* exp(-1j * 4 * pi * radar.fc * yq / radar.c);

        tau_q = 2 * yq / radar.c;
        range_prof = sinc(radar.Bw * (t_fast - tau_q));
        az_phase = exp(-1j * 4 * pi * radar.fc * xq * radar.omega_r * tm / radar.c);

        HRRP = HRRP + sigma_prime * (range_prof * az_phase);
    end
end


% =========================================================================
% 将物理散射中心粗映射到 RD 栅格（就是第二个文件里的 Img_Ideal）
% =========================================================================
function Img = rasterize_physical_scatterers(scatterers, radar)
    Img = complex(zeros(radar.Nr, radar.Na));

    for q = 1:numel(scatterers)
        yq = scatterers(q).y;
        xq = scatterers(q).x;
        sigma_q = scatterers(q).sigma;

        rCentered = round(yq / radar.delta_r);
        rr = rCentered + floor(radar.Nr / 2) + 1;

        fD = -2 * xq * radar.omega_r / radar.lambda;
        dCentered = round(fD * radar.Na / radar.PRF);
        dd = dCentered + floor(radar.Na / 2) + 1;

        if rr >= 1 && rr <= radar.Nr && dd >= 1 && dd <= radar.Na
            Img(rr, dd) = Img(rr, dd) + sigma_q;
        end
    end
end


% =========================================================================
% 每样本独立随机 Mask（方位向整列采样）
% =========================================================================
function [Mask_single, keep_ratio, pulse_mask] = generate_random_mask(radar, cfg)
    if cfg.use_random_mask
        keep_ratio_target = rand_uniform(cfg.mask_keep_ratio_min, cfg.mask_keep_ratio_max);
        num_keep = round(radar.Na * keep_ratio_target);
        num_keep = max(1, min(radar.Na, num_keep));
        keep_cols = randperm(radar.Na, num_keep);

        pulse_mask = zeros(1, radar.Na);
        pulse_mask(keep_cols) = 1;
        Mask_single = repmat(pulse_mask, radar.Nr, 1);
        keep_ratio = num_keep / radar.Na;
    else
        pulse_mask = ones(1, radar.Na);
        Mask_single = ones(radar.Nr, radar.Na);
        keep_ratio = 1.0;
    end
end


% =========================================================================
% 复高斯加噪
% =========================================================================
function Y_noisy = add_complex_noise(Y_clean_like, snr_db, power_mode)
    switch lower(power_mode)
        case 'mean'
            sig_power = mean(abs(Y_clean_like(:)).^2);
        case 'peak'
            sig_power = max(abs(Y_clean_like(:)).^2);
        otherwise
            error('未知噪声功率模式: %s', power_mode);
    end

    noise_power = sig_power / (10^(snr_db / 10));
    noise = sqrt(noise_power / 2) * ...
        (randn(size(Y_clean_like)) + 1j * randn(size(Y_clean_like)));

    Y_noisy = Y_clean_like + noise;
end


% =========================================================================
% 归一化尺度
% =========================================================================
function scale = compute_scale(Y_input, X_gt_raw, cfg, Mask_single)
    switch lower(cfg.scale_mode)
        case 'gt_percentile'
            scale = prctile(abs(X_gt_raw(:)), cfg.scale_percentile);
        case 'gt_max'
            scale = max(abs(X_gt_raw(:)));
        case 'obs_rms'
            scale = norm(Y_input(:), 'fro') / sqrt(sum(Mask_single(:)));
        otherwise
            error('未知归一化模式: %s', cfg.scale_mode);
    end

    scale = max(scale, cfg.scale_eps);
end


% =========================================================================
% 轻微运动扰动：频域数据上的慢时间二次相位
% =========================================================================
function Y_out = apply_motion_phase_error_fc(Y_in, radar, v, a)
    t = radar.tm(:).';
    r_t = v * t + 0.5 * a * (t.^2);
    phase = exp(-1i * 4 * pi * radar.fc / radar.c * r_t);
    Y_out = Y_in .* repmat(phase, size(Y_in, 1), 1);
end


% =========================================================================
% 划分 train / val / test
% =========================================================================
function [split_name, idx_local] = get_split_name(i, ntrain, nval, ntest)
    if i <= ntrain
        split_name = 'train';
        idx_local = i;
    elseif i <= ntrain + nval
        split_name = 'val';
        idx_local = i - ntrain;
    elseif i <= ntrain + nval + ntest
        split_name = 'test';
        idx_local = i - ntrain - nval;
    else
        error('索引越界');
    end
end


% =========================================================================
% 预览图
% =========================================================================
function show_preview_figure_sim(Y_clean_freq, X_gt_raw, X_gt_used)
    HRRP_full = fftshift(ifft(Y_clean_freq, [], 1), 1);

    figure('Name', 'Physical-Sim Training Dataset Preview', 'Color', 'w', ...
           'Position', [100, 100, 1500, 420]);

    ref_gt = max([abs(X_gt_raw(:)); abs(X_gt_used(:))]) + eps;

    subplot(1,4,1);
    imagesc(abs(HRRP_full));
    axis image; axis xy;
    colormap(gca, hot); colorbar;
    title('Full-Aperture HRRP |H|');
    xlabel('Azimuth pulse'); ylabel('Range bin');

    subplot(1,4,2);
    show_image_paper_style(abs(X_gt_raw), ref_gt, -35, 'hot');
    title('X\_gt\_raw = Img\_Label');

    subplot(1,4,3);
    show_image_paper_style(abs(X_gt_used), ref_gt, -35, 'hot');
    title('X\_gt\_used = Img\_Ideal');

    subplot(1,4,4);
    show_image_paper_style(abs(X_gt_raw - X_gt_used), [], -35, 'hot');
    title('|Img\_Label - Img\_Ideal|');
end


% =========================================================================
% 论文风格显示
% =========================================================================
function show_image_paper_style(Img, ref_max, db_floor, cmap_name)
    if nargin < 2 || isempty(ref_max)
        ref_max = max(abs(Img(:))) + eps;
    end
    if nargin < 3 || isempty(db_floor)
        db_floor = -35;
    end
    if nargin < 4 || isempty(cmap_name)
        cmap_name = 'hot';
    end

    mag = abs(Img);
    img_db = 20 * log10(mag / ref_max + eps);
    img_db = max(img_db, db_floor);

    imagesc(img_db, [db_floor, 0]);
    axis image; axis xy;
    colorbar;
    xlabel('Azimuth bin');
    ylabel('Range bin');
    set(gca, 'Color', 'k');

    switch lower(cmap_name)
        case 'gray'
            colormap(gca, gray);
        case 'parula'
            colormap(gca, parula);
        otherwise
            colormap(gca, hot);
    end
end


% =========================================================================
% 工具函数
% =========================================================================
function x = rand_uniform(a, b)
    x = a + (b - a) * rand();
end
