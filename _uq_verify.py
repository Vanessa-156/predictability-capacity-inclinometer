r"""UQ（不确定性量化）可行性验证 — MCDropout epistemic + 异方差 aleatoric head。

对照 V5(full) 架构：仅把 head 换成 均值头 + 对数方差头，用 Gaussian NLL 训练；
预测时 MCDropout 多次前向得到 epistemic，总方差 = aleatoric + epistemic。

判据（本脚本直接输出）：
  1. UQ 模型 RMSE 是否接近原 V5（对比 Ablation_New H6 的 V5 均值 0.5114）
  2. 预测区间实际覆盖率（应按近似 90% 名义水平）
  3. epistemic / aleatoric 相对大小（是否合理）

用法：NO_WATER_DEV=cpu|auto|gpu  NO_WATER_HZS=6  NO_WATER_SEEDS=42
"""
import sys, os, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).parent
OUT = ROOT / "_uq_out"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

import Ablation_Experiment as AB
from Ablation_Experiment import set_seed
import _verify_no_water as VW

# =====================================================================
# UQ 模型：V5(full) 完整前向，head 双输出
# =====================================================================
def _init_logvar(small=0.0):
    """logvar 头初始化为较小方差（logvar 偏置），避免训练初期大方差偷懒。"""
    def init(m):
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.bias)
            if getattr(m, 'weight', None) is not None:
                nn.init.xavier_normal_(m.weight)
    return init


class UQAblationModel(nn.Module):
    def __init__(self, cfg, N, in_dim, gcn_hid, tcn_ch, k, dil, attn_hid, hor,
                 adj, out_dim, geo_dim, vert_adj=None, mc_dropout=True,
                 logvar_bias=-1.0):
        super().__init__()
        self.mc_dropout = mc_dropout
        self.logvar_bias = logvar_bias
        c = cfg
        self.tcn = AB.TCNEncoder(in_dim, tcn_ch, k, dil, c.TCN_DROPOUT)
        tcn_dim = tcn_ch[-1]
        self.attn1 = nn.Linear(tcn_dim, attn_hid)
        self.attn_v = nn.Linear(attn_hid, 1, bias=False)
        self.water_node_proj = nn.Linear(8, tcn_dim)
        self.gcn = AB.GatedGCN(tcn_dim + tcn_dim, gcn_hid, adj, dropout=c.DROPOUT)
        self.gcn_norm = nn.LayerNorm(gcn_hid)
        self.gcn_drop = nn.Dropout(c.DROPOUT)
        cur_dim = gcn_hid
        self.register_buffer('static_adj', adj)
        if vert_adj is not None:
            self.register_buffer('vert_adj', vert_adj)
        else:
            self.vert_adj = None
        self.physics_gate = AB.PhysicsGate(cur_dim, geo_dim)
        self.cross_attn = AB.CrossNodeAttention(cur_dim, geo_dim)
        self.vertical_gcn = AB.VerticalGCN(tcn_dim + tcn_dim)
        self.mean_head = nn.Sequential(
            nn.Linear(cur_dim, cur_dim // 2), nn.LeakyReLU(0.1),
            nn.Dropout(c.DROPOUT), nn.Linear(cur_dim // 2, hor * out_dim))
        self.logvar_head = nn.Sequential(
            nn.Linear(cur_dim, cur_dim // 2), nn.LeakyReLU(0.1),
            nn.Dropout(c.DROPOUT), nn.Linear(cur_dim // 2, hor * out_dim))
        # 最后一个 Linear bias 预置为 logvar_bias（如 -1 → var~0.37），温和抑制初始过大
        nn.init.constant_(self.logvar_head[-1].bias, self.logvar_bias)
        self.horizon = hor
        self.out_dim = out_dim

    def _encode(self, x, water_feat, geo_static):
        B, seq, Nodes, in_feat = x.shape
        tcn_in = x.permute(0, 2, 1, 3).reshape(B * Nodes, seq, in_feat)
        tcn_out = self.tcn(tcn_in)
        s = torch.tanh(self.attn1(tcn_out)); s = self.attn_v(s).squeeze(-1)
        w = F.softmax(s, dim=1)
        ctx = torch.bmm(w.unsqueeze(1), tcn_out).squeeze(1).reshape(B, Nodes, -1)
        wn = self.water_node_proj(water_feat)
        ctx = torch.cat([ctx, wn], dim=-1)
        ctx = self.vertical_gcn(ctx, self.vert_adj)
        ctx = self.gcn(ctx, dynamic_adj=None)
        ctx = self.gcn_norm(ctx); ctx = self.gcn_drop(F.relu(ctx))
        ctx = self.physics_gate(ctx, geo_static)
        ctx = self.cross_attn(ctx, geo_static)
        return ctx.mean(dim=1)

    def forward(self, x, water_feat, geo_static):
        g = self._encode(x, water_feat, geo_static)
        mean = self.mean_head(g).reshape(-1, self.horizon, self.out_dim)
        logvar = self.logvar_head(g).reshape(-1, self.horizon, self.out_dim)
        return mean, logvar


def gaussian_nll(mean, logvar, target):
    var = torch.exp(logvar).clamp(min=1e-6)
    return 0.5 * (logvar + (mean - target) ** 2 / var + np.log(2 * np.pi))


def train_uq(cfg, model, tld, vld, save_path, node_mask=None):
    opt = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters()
                    if 'gcn' in n or 'cross_attn' in n or 'physics' in n or 'vert' in n],
         'lr': cfg.LR_GCN},
        {'params': [p for n, p in model.named_parameters()
                    if not ('gcn' in n or 'cross_attn' in n or 'physics' in n or 'vert' in n)],
         'lr': cfg.LR}], weight_decay=cfg.WD)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=15)
    best = float('inf'); wait = 0
    for ep in range(1, cfg.EPOCHS + 1):
        model.train(); tl = 0
        for Xb, yb, wf, gs in tld:
            Xb=Xb.to(cfg.DEVICE); yb=yb.to(cfg.DEVICE); wf=wf.to(cfg.DEVICE); gs=gs.to(cfg.DEVICE)
            opt.zero_grad()
            mean, logvar = model(Xb, wf, gs)
            loss = gaussian_nll(mean, logvar, yb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            opt.step(); tl += loss.item() * Xb.size(0)
        tl /= len(tld.dataset)
        model.eval(); vl = 0
        with torch.no_grad():
            for Xb, yb, wf, gs in vld:
                Xb=Xb.to(cfg.DEVICE); yb=yb.to(cfg.DEVICE); wf=wf.to(cfg.DEVICE); gs=gs.to(cfg.DEVICE)
                mean, logvar = model(Xb, wf, gs)
                vl += gaussian_nll(mean, logvar, yb).mean().item() * Xb.size(0)
        vl /= len(vld.dataset)
        sch.step(vl)
        if vl < best:
            best = vl; wait = 0
            torch.save(model.state_dict(), save_path)
        else:
            wait += 1
            if wait >= cfg.PATIENCE:
                break
    model.load_state_dict(torch.load(save_path, map_location=cfg.DEVICE))
    return model


# 参照 Ablation_Experiment.evaluate 重建累计位移
def to_cum(norm_pred, scY, pca_y, init_disp, hz):
    """把 y_norm 空间预测逆变换为累计位移（不传播不确定性，仅比较 mean）。"""
    ns = norm_pred.shape[0]; out_dim = norm_pred.shape[-1]
    pred_pca = scY.inverse_transform(norm_pred.reshape(-1, out_dim)).reshape(ns, hz, out_dim)
    N_orig = pca_y.n_features_in_
    pred_diff = pca_y.inverse_transform(pred_pca.reshape(-1, out_dim)).reshape(ns, hz, N_orig)
    cum = np.zeros_like(pred_diff)
    for i in range(ns):
        cp = init_disp[i].copy()
        for t in range(hz):
            cp = cp + pred_diff[i, t, :]
            cum[i, t, :] = cp
    return cum


def run(cfg, seed, hz):
    set_seed(42)
    full = AB.load_raw_data_with_meta(cfg)
    ts = np.array(full['测量时间'].unique())
    wd, wa = AB.load_water_data(ts, cfg)
    water_feat = AB.extract_water_features(wd, wa)
    _, _, meta = AB.build_node_features(full, cfg)
    geo_records = AB.load_geology_params(cfg)
    geo_feat_raw = AB.assign_geo_to_nodes(geo_records, meta['depths'], cfg) if geo_records else None
    n = len(ts)
    tr_=np.arange(int(n*cfg.TRAIN_RATIO)); va=np.arange(int(n*cfg.TRAIN_RATIO),int(n*(cfg.TRAIN_RATIO+cfg.VAL_RATIO))); te=np.arange(int(n*(cfg.TRAIN_RATIO+cfg.VAL_RATIO)),n)
    geo_fb = geo_feat_raw if geo_feat_raw is not None else np.zeros((len(meta['depths']), cfg.GEO_FEAT_DIM), dtype=np.float32)
    data = VW.prepare_data_uw(cfg, full, wd, wa, meta, tr_, va, te, water_feat, geo_fb, hz, use_water=True)
    geo_fb_use = geo_feat_raw if geo_feat_raw is not None and np.any(geo_feat_raw) else None
    adj = AB.build_adjacency(meta, geo_fb_use, cfg); adj_t = torch.FloatTensor(adj).to(cfg.DEVICE)
    vert = AB.build_vertical_adjacency(meta, cfg); vert_t = torch.FloatTensor(vert).to(cfg.DEVICE) if vert is not None else None
    geom_t = torch.FloatTensor(data['geo_static']).to(cfg.DEVICE)
    N=data['N']; F_in=data['F_in']; out_dim=data['out_dim']
    def ds(X,y,wf): return TensorDataset(torch.FloatTensor(X),torch.FloatTensor(y),torch.FloatTensor(wf),geom_t.unsqueeze(0).expand(X.shape[0],N,-1))
    tld=DataLoader(ds(data['Xtr'],data['ytr'],data['wf_train']),batch_size=cfg.BATCH_SIZE,shuffle=True)
    vld=DataLoader(ds(data['Xva'],data['yva'],data['wf_val']),batch_size=cfg.BATCH_SIZE)
    testld=DataLoader(ds(data['Xte'],data['yte'],data['wf_test']),batch_size=cfg.BATCH_SIZE)

    AB.set_seed(seed)
    model = UQAblationModel(cfg, N, F_in, cfg.GCN_HIDDEN, cfg.TCN_CHANNELS, cfg.TCN_KERNEL_SIZE,
                            cfg.TCN_DILATIONS, cfg.ATTENTION_HIDDEN, hz, adj_t, out_dim,
                            data['geo_dim_actual'], vert_adj=vert_t).to(cfg.DEVICE)
    save = OUT / f"u_{seed}_H{hz}.pth"
    train_uq(cfg, model, tld, vld, save)

    # ---- MCDropout 预测：mean 累计 + 覆盖率先验 ----
    model.eval()
    MC = 40
    # 关掉 mean/logvar 头后 dropout 仍在前向里生效
    model.mc_dropout = True
    model.train()  # 进入训练模式以激活 dropout
    means, legvars = [], []
    with torch.no_grad():
        for _ in range(MC):
            m_all, lv_all = [], []
            for Xb, yb, wf, gs in testld:
                Xb=Xb.to(cfg.DEVICE); wf=wf.to(cfg.DEVICE); gs=gs.to(cfg.DEVICE)
                m, lv = model(Xb, wf, gs)
                m_all.append(m.cpu().numpy()); lv_all.append(lv.cpu().numpy())
            means.append(np.concatenate(m_all, 0)); legvars.append(np.exp(np.concatenate(lv_all, 0)))
    model.eval(); model.mc_dropout = False
    mean_stack = np.stack(means, 0)                 # (MC, ns, hz, out_dim)
    mean_pred = mean_stack.mean(0)                  # (ns, hz, out_dim)
    epi_var = mean_stack.var(0)                     # epistemic
    ale_var = np.mean(legvars, 0)                   # aleatoric
    total_var = ale_var + epi_var
    std_tot = np.sqrt(total_var)

    # 真值 y（y_norm 空间）
    trues = []
    with torch.no_grad():
        for Xb, yb, wf, gs in testld:
            trues.append(yb.numpy())
    true_y = np.concatenate(trues, 0)

    # 覆盖率：在 y_norm 空间看 mean±z*std 覆盖真实 y 的比例（近似高斯）
    z = 1.645  # 90%
    cover90 = float((np.abs(mean_pred - true_y) <= z * std_tot).mean())
    z50 = 0.6745
    cover50 = float((np.abs(mean_pred - true_y) <= z50 * std_tot).mean())

    # RMSE（mean 预测，y_norm 空间）
    rmse_ynorm = float(np.sqrt(np.mean((mean_pred - true_y) ** 2)))
    # 累计位移 RMSE（与论文对齐）
    pred_cum = to_cum(mean_pred, data['scY'], data['pca_y'], data['init_test'], hz)
    true_cum = to_cum(true_y, data['scY'], data['pca_y'], data['init_test'], hz)
    rmse_cum = float(np.sqrt(np.mean((pred_cum - true_cum) ** 2)))

    out = dict(seed=seed, hz=hz, rmse_ynorm=rmse_ynorm, rmse_cum=rmse_cum,
               cover90=cover90, cover50=cover50,
               epi_sd=float(epi_var.mean()**0.5), ale_sd=float(ale_var.mean()**0.5),
               epi_mean_perc=float(epi_var.mean()/(epi_var.mean()+ale_var.mean()+1e-12)),
               params=AB.count_params(model))
    with open(OUT/f"metrics_{seed}_H{hz}.json","w") as f: json.dump(out,f,indent=2)
    return out


# =====================================================================
# 种子集成 UQ 分析（主入口，post-hoc，零新增训练）
# 直接读 <ABLATION_ROOT>/H*/V5_Full_Model 的 30 个 seed 预测 (pred_cum, true_cum)
# epistemic = 种子间 std；研究与单模型比覆盖/精度；并考察压力区段。
# 2026-09-22：末步口径（论文主口径）+ 输出空间地板分解（读同目录已回填的
# metrics.json），使刷新后的数字可直接与主表 / 零变化基线比较。
# =====================================================================
def _floor_last(ablation_root, hz, variant):
    """从同目录 run_*/metrics.json 读公共地板（Ablation 侧已回填 RMSE_floor_last）。"""
    vals = []
    for mf in sorted((Path(ablation_root) / f"H{hz}" / variant).glob("run_*/metrics.json")):
        with open(mf, 'r', encoding='utf-8') as f:
            gm = json.load(f)
        if 'RMSE_floor_last' in gm:
            vals.append(gm['RMSE_floor_last'])
    return float(np.mean(vals)) if vals else None


def calibration_report(m, s, trues, floor_last=None):
    """区间标定诊断（零训练，post-hoc）：原始 / 并入地板 / 半集 conformal 三种尺度。

    误差 |e|（e = trues − m）可分解为「种子间可解释部分」与「种子间共同部分」（公共
    地板 + 系统性偏差）。只用种子 std 当区间宽度 ⇒ 系统性低估。本函数对比三种修法：

      raw    : 宽度 = z·s                      （现行做法，未标定）
      +floor : 宽度 = z·sqrt(s² + MSE_floor)   （地板是数据侧已知量，不构成标签泄漏）
      conf   : 半集标定尺度 a = Q_target(|e|/s)，再在另一半评覆盖（split-conformal；
               按窗口奇偶切分以保留窗内相关结构，两方向对称取均值，marginal 覆盖）

    Returns
    -------
    dict: 三套 90%/50% 覆盖率、conformal 标定尺度 a、以及 90% 区间的平均宽度（物理单位）。
    """
    e = np.abs(trues - m)
    se = np.maximum(s, 1e-8)
    ratio = e / se
    z90, z50 = 1.6449, 0.6745
    res = dict(cover90_raw=float((e <= z90 * se).mean()),
               cover50_raw=float((e <= z50 * se).mean()),
               width90_raw=float(2 * z90 * se.mean()))
    if floor_last:
        sf = np.sqrt(se ** 2 + float(floor_last) ** 2)
        res['cover90_floor'] = float((e <= z90 * sf).mean())
        res['cover50_floor'] = float((e <= z50 * sf).mean())
        res['width90_floor'] = float(2 * z90 * sf.mean())
    idx = np.arange(m.shape[0])
    even, odd = idx[idx % 2 == 0], idx[idx % 2 == 1]
    for target, tag in ((0.90, 'conf90'), (0.50, 'conf50')):
        covs, scales = [], []
        for cal_i, val_i in ((even, odd), (odd, even)):
            a = float(np.quantile(ratio[cal_i].ravel(), target))
            scales.append(a)
            covs.append(float((ratio[val_i] <= a).mean()))
        a_bar = float(np.mean(scales))
        res[f'{tag}_coverage'] = float(np.mean(covs))
        res[f'{tag}_scale'] = a_bar
        res[f'{tag}_width'] = float(2 * a_bar * se.mean())
    return res


def ensemble_uq(ablation_root, hz, variants=("V5_Full_Model",)):
    out_all = []
    for v in variants:
        runs = sorted((Path(ablation_root) / f"H{hz}" / v).glob("run_*/predictions.npz"))
        preds, trues = [], None
        for r in runs:
            d = np.load(r)
            preds.append(d["pred_cum"])
            trues = d["true_cum"]
        preds = np.stack(preds, 0)        # (seeds, ns, hz, 155)
        m = preds.mean(0)                 # 集成均值
        s = preds.std(0)                  # epistemic = 种子间 std
        ns_, hz_, N_ = m.shape
        # 总体 RMSE（全步口径）：单模型 vs 集成
        rmse_ens = float(np.sqrt(np.mean((m - trues) ** 2)))
        rmse_single = [float(np.sqrt(np.mean((preds[i] - trues) ** 2))) for i in range(preds.shape[0])]
        # 末步口径（论文主口径）：每窗只取 t=hz-1，与视界无关
        rmse_ens_last = float(np.sqrt(np.mean((m[:, -1, :] - trues[:, -1, :]) ** 2)))
        rmse_single_last = [float(np.sqrt(np.mean((preds[i, :, -1, :] - trues[:, -1, :]) ** 2)))
                            for i in range(preds.shape[0])]
        floor_last = _floor_last(ablation_root, hz, v)
        reach_ens_last = (float(np.sqrt(max(rmse_ens_last ** 2 - floor_last ** 2, 0.0)))
                          if floor_last else None)
        reach_mean_last = (float(np.sqrt(max(float(np.mean(rmse_single_last)) ** 2
                                             - floor_last ** 2, 0.0)))
                           if floor_last else None)
        # 集成预测区间：标定诊断（原始 / 并入地板 / 半集 conformal，零训练）
        cal = calibration_report(m, s, trues, floor_last=floor_last)
        cover90, cover50 = cal['cover90_raw'], cal['cover50_raw']
        out = dict(variant=v, hz=hz, n_seeds=preds.shape[0],
                   rmse_ensemble=rmse_ens,
                   rmse_single_mean=float(np.mean(rmse_single)),
                   rmse_single_best=float(np.min(rmse_single)),
                   rmse_single_worst=float(np.max(rmse_single)),
                   rmse_ensemble_last=rmse_ens_last,
                   rmse_single_mean_last=float(np.mean(rmse_single_last)),
                   rmse_single_best_last=float(np.min(rmse_single_last)),
                   rmse_single_worst_last=float(np.max(rmse_single_last)),
                   rmse_floor_last=floor_last,
                   rmse_reach_last_ensemble=reach_ens_last,
                   rmse_reach_last_single_mean=reach_mean_last,
                   cover90_epistemic=cover90, cover50_epistemic=cover50,
                   epi_sd_mean=float(s.mean()), calibration=cal)
        out_all.append(out)
        print(f"[{v} H{hz}] seeds={preds.shape[0]}")
        print(f"  全步口径  RMSE集成={rmse_ens:.5f} vs 单模型mean={np.mean(rmse_single):.5f}"
              f"(最优{np.min(rmse_single):.5f}/最差{np.max(rmse_single):.5f})")
        print(f"  末步口径  RMSE集成={rmse_ens_last:.5f} vs 单模型mean="
              f"{np.mean(rmse_single_last):.5f}(最优{np.min(rmse_single_last):.5f}"
              f"/最差{np.max(rmse_single_last):.5f})")
        if floor_last is not None:
            print(f"  误差分解  地板={floor_last:.5f}  ⇒ 可达：集成 {reach_ens_last:.5f} "
                  f"vs 单模型mean {reach_mean_last:.5f}"
                  f"（降 {(1 - reach_ens_last / reach_mean_last) * 100:+.2f}%）")
        print(f"  区间覆盖  90%={cover90:.3f} 50%={cover50:.3f}  epi_sd={float(s.mean()):.4f}")
        if 'cover90_floor' in cal:
            print(f"  并入地板  90%={cal['cover90_floor']:.3f} 50%={cal['cover50_floor']:.3f}"
                  f"  （宽度 {cal['width90_floor']:.3f} ← 原 {cal['width90_raw']:.3f}）")
        print(f"  半集标定  90%={cal['conf90_coverage']:.3f} (尺度a={cal['conf90_scale']:.2f},"
              f"宽 {cal['conf90_width']:.3f})  50%={cal['conf50_coverage']:.3f}"
              f" (尺度a={cal['conf50_scale']:.2f}, 宽 {cal['conf50_width']:.3f})")
    return out_all


if __name__ == '__main__':
    # 种子集成分析（默认，零训练）。根目录直取 Ablation 的 Config.OUT_ROOT，
    # 避免再指向已过期的 Ablation_New（12 维含水旧架构）。
    ablation_root = os.environ.get('UQ_ABLATION_ROOT', str(AB.Config().OUT_ROOT))
    hzs = [int(x) for x in os.environ.get('NO_WATER_HZS', '6,10').split(',')]
    print(f"Ablation 根目录：{ablation_root}")
    results = []
    for hz in hzs:
        results.extend(ensemble_uq(ablation_root, hz))
    with open(OUT / 'ensemble_uq_summary.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"已落盘：{OUT / 'ensemble_uq_summary.json'}")
    print("DONE")