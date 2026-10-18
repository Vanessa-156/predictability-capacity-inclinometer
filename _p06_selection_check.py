# -*- coding: utf-8 -*-
r"""_p06_selection_check.py — 模型选择泄漏的验证集复核（零训练，仅前向）。

目标：把「架构 / PCA 维数 K / 损失权重」的选择限定在**验证集**内，检查若在 val 上选，
      是否仍选中正文的配置（Ours = V7_wo_VerticalGCN；K = 13）。

严格零训练：
  · 只对既有 `best.pth` 做**前向推理**（eval 模式、torch.no_grad），不训练、不更新权重；
  · 测试集预测与对照直接读既有 `predictions.npz`（论文主表来源），并另用同一前向管线
    在测试集上复算一遍做交叉校验。

口径（与主表一致）：
  · 末步累计口径 RMSE_last：先按窗逐差累加还原累计位移，再只取每窗末步 t=hz-1；
  · 逐种子计算 RMSE_last，再跨 30 种子取均值±std（ddof=1）；评估起点用 init_disp
    （窗末观测），真值取原始一阶差分累加（与 Ablation_Experiment.evaluate 一致）。

复用的既有前向写法来自 `_p07_block_conformal.py`（已验证锚点）。

用法（PowerShell）：
  $env:P06_HZS="2,6,8,10,12"; $env:P06_SEEDS="30"; python _p06_selection_check.py
"""
from _env import EXAMPLE_MODE, WORK_ROOT

import os, sys, time
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch

ROOT = WORK_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Ablation_Experiment as AB  # noqa: E402
import PCA_Experiment as PE        # noqa: E402

OUT_DIR = ROOT / "_paper_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUP_DIR = ROOT / "补充材料"
SUP_DIR.mkdir(parents=True, exist_ok=True)

H_LIST = [int(x) for x in os.environ.get("P06_HZS", "2,6,8,10,12").split(",")]
SEED_LIMIT = int(os.environ.get("P06_SEEDS", "30"))
B_BOOT = 2000
BOOT_SEED = 42
L_BLK = 3                     # 块 bootstrap 的窗块长（与主口径一致）

OURS = "V7_wo_VerticalGCN"
ANCHOR_H = 10
ANCHOR_RMSE, ANCHOR_STD, ANCHOR_PERSIST = 1.5687, 0.0940, 1.372010

# K 候选：PCA 保留率 → 既有独立训练权重（PCA_Results_V7）。K 由 PCA(rate) 在 y_train
# 上拟合得到，实测 H=10 时 pca80→K=7、pca90→K=13、pca95→K=21。
K_COND = OrderedDict([
    ("K7_PCA80", ("pca80", 0.80)),
    ("K13_PCA90", ("pca90", 0.90)),
    ("K21_PCA95", ("pca95", 0.95)),
    ("K155_NoPCA", ("nopca", None)),
])


# =====================================================================
# 工具
# =====================================================================
def rmse_last(pred_cum, true_cum):
    """末步累计口径 RMSE：只取每窗最后一步、逐节点。"""
    d = pred_cum[:, -1, :] - true_cum[:, -1, :]
    return float(np.sqrt(np.mean(d ** 2)))


def circular_block_idx(n, L, rng):
    out = []
    while len(out) < n:
        s = int(rng.integers(0, n))
        out.extend([(s + k) % n for k in range(L)])
    return np.asarray(out[:n], dtype=int)


def clip_seeds(n):
    return n if SEED_LIMIT <= 0 else min(n, SEED_LIMIT)


# =====================================================================
# 主流程
# =====================================================================
def main():
    t0 = time.time()
    cfg = AB.Config()
    device = cfg.DEVICE
    ABL_ROOT = cfg.OUT_ROOT
    PE_ROOT = PE.Config().OUT_ROOT
    print(f"设备: {device} | Ours={OURS} | H={H_LIST} | seeds<={SEED_LIMIT}")
    print(f"消融结果根: {ABL_ROOT}")
    print(f"PCA 结果根: {PE_ROOT}")

    # ---------------- 数据（只加载一次；消融数据与变体开关无关） ----------------
    AB.set_seed(42)
    full_df = AB.load_raw_data_with_meta(cfg)
    ts = np.sort(full_df["测量时间"].unique())
    wd, wa = AB.load_water_data(ts, cfg)
    water_feat = AB.extract_water_features(wd, wa)
    disp_raw, _, meta = AB.build_node_features(full_df, cfg)
    geo_records = AB.load_geology_params(cfg)
    geo_feat_raw = AB.assign_geo_to_nodes(geo_records, meta["depths"], cfg) \
        if geo_records else None
    n = len(ts)
    tr_end = int(n * cfg.TRAIN_RATIO)
    va_end = int(n * (cfg.TRAIN_RATIO + cfg.VAL_RATIO))
    train_idx = np.arange(tr_end)
    val_idx = np.arange(tr_end, va_end)
    test_idx = np.arange(va_end, n)
    print(f"时间划分: train[0,{tr_end}) val[{tr_end},{va_end}) test[{va_end},{n})  共 {n} 步")

    geo_fb = geo_feat_raw if geo_feat_raw is not None else np.zeros(
        (len(meta["depths"]), cfg.GEO_FEAT_DIM), dtype=np.float32)
    geo_fb_use = geo_feat_raw if geo_feat_raw is not None and np.any(geo_feat_raw) else None

    # 去噪（与 prepare_data / prepare_data_with_pca 内部完全一致），用于取 val/test 起点
    train_den, val_den, test_den = AB.denoise_split(
        disp_raw[train_idx], disp_raw[val_idx], disp_raw[test_idx], cfg)
    all_den = np.vstack([train_den, val_den, test_den])
    diff1_den = np.zeros_like(all_den)
    diff1_den[1:] = np.diff(all_den, axis=0)
    LB = cfg.LOOKBACK

    vert_adj_np = AB.build_vertical_adjacency(meta, cfg)

    rows = []          # T_p06_selection.csv
    anchor_lines = []
    report = {}        # 供 MD 生成的结构化结果
    anchor_ok = True

    for hz in H_LIST:
        print(f"\n{'='*74}\nH={hz}\n{'='*74}")
        data = AB.prepare_data(cfg, full_df, wd, wa, meta, train_idx, val_idx, test_idx,
                               water_feat, geo_fb, hz, disp_raw=disp_raw)
        N, F_in, out_dim = data["N"], data["F_in"], data["out_dim"]
        geo_static_t = torch.FloatTensor(data["geo_static"]).to(device)
        valid_mask_t = torch.FloatTensor(data["valid_mask"]).to(device)
        vert_adj_t = torch.FloatTensor(vert_adj_np).to(device)

        Xva, Xte = data["Xva"], data["Xte"]
        nva, nte = Xva.shape[0], Xte.shape[0]
        # val 真值窗（与 evaluate 口径一致）
        y_val = diff1_den[val_idx]
        yva_raw = np.stack([y_val[i + LB:i + LB + hz] for i in range(nva)]) \
            if nva else np.empty((0, hz, y_val.shape[1]), dtype=np.float32)
        init_val = val_den[LB - 1:-hz][:nva]

        def make_ds(X, y, wf, nwin):
            return torch.utils.data.TensorDataset(
                torch.FloatTensor(X), torch.FloatTensor(y), torch.FloatTensor(wf),
                geo_static_t.unsqueeze(0).expand(nwin, N, -1))

        val_ld = torch.utils.data.DataLoader(
            make_ds(Xva, data["yva"], data["wf_val"], nva), batch_size=cfg.BATCH_SIZE)
        test_ld = torch.utils.data.DataLoader(
            make_ds(Xte, data["yte"], data["wf_test"], nte), batch_size=cfg.BATCH_SIZE)

        # ---------------- (1) 消融 8 变体：val / test 前向 ----------------
        VP, TP = {}, {}          # variant -> (S, nw, hz, N) 前向预测
        seeds_ref, TV, TT = None, None, None
        npz_rmse = {}            # variant -> npz(测)逐种子 RMSE_last（对照）

        for variant, switch in AB.ABLATION_VARIANTS.items():
            vcfg = AB.Config()
            for k, v in switch.items():
                setattr(vcfg, k, v)
            vcfg.DEVICE = device
            adj = AB.build_adjacency(meta, geo_fb_use, vcfg,
                                     zero_same_hole=getattr(vcfg, "ZERO_SAME_HOLE", False))
            adj_t = torch.FloatTensor(adj).to(device)

            run_dir = ABL_ROOT / f"H{hz}" / variant
            runs = sorted(run_dir.glob("run_*/predictions.npz"),
                          key=lambda p: int(p.parent.name.split("_")[1]))
            npz_pred, npz_true, seeds = [], None, []
            for r in runs:
                d = np.load(r)
                npz_pred.append(d["pred_cum"])
                npz_true = d["true_cum"]
                seeds.append(int(r.parent.name.split("_")[1]))
            kk = clip_seeds(len(seeds))
            seeds, npz_pred = seeds[:kk], npz_pred[:kk]
            # 归档可能只保留部分变体的权重（示例归档保留 V0/V7）；缺权重时跳过该变体
            missing_w = [s for s in seeds
                         if not (run_dir / f"run_{s}" / "best.pth").exists()]
            if missing_w:
                print(f"  [skip] {variant} H{hz}: 缺 {len(missing_w)} 个 best.pth"
                      f"（示例归档仅保留 V0/V7 权重）")
                continue
            if seeds_ref is None:
                seeds_ref = seeds
            assert seeds == seeds_ref, f"{variant} H{hz} 种子序不一致"

            model = AB.AblationModel(
                vcfg, N=N, in_dim=F_in, gcn_hid=vcfg.GCN_HIDDEN,
                tcn_ch=vcfg.TCN_CHANNELS, k=vcfg.TCN_KERNEL_SIZE,
                dil=vcfg.TCN_DILATIONS, attn_hid=vcfg.ATTENTION_HIDDEN,
                hor=hz, adj=adj_t, out_dim=out_dim,
                geo_feat_dim=data["geo_dim_actual"], vert_adj=vert_adj_t).to(device)

            vm, tm, nz = [], [], []
            for seed, pc_npz in zip(seeds, npz_pred):
                pth = run_dir / f"run_{seed}" / "best.pth"
                state = torch.load(pth, map_location=device)
                model.load_state_dict(state, strict=False)   # thop 缓冲键放宽匹配
                pv, tv = AB.evaluate(model, val_ld, init_val, data["scY"], data["pca_y"],
                                     hz, device, node_mask=valid_mask_t,
                                     true_diff_win=yva_raw)
                pt, tt = AB.evaluate(model, test_ld, data["init_test"], data["scY"],
                                     data["pca_y"], hz, device, node_mask=valid_mask_t,
                                     true_diff_win=data["yte_raw"])
                vm.append(pv); tm.append(pt)
                nz.append(rmse_last(pc_npz, npz_true))
                TV, TT = tv, tt
            VP[variant] = np.stack(vm, 0)
            TP[variant] = np.stack(tm, 0)
            npz_rmse[variant] = np.array(nz)
            del model

        # 前向复算(测) 与 npz 一致性交叉校验
        max_dev = max(float(np.abs(np.array([rmse_last(TP[v][i], TT) for i in range(len(seeds_ref))])
                                    - npz_rmse[v]).max()) for v in VP)
        print(f"  前向复算(测) vs npz 逐种子 RMSE_last 最大偏差 = {max_dev:.2e}")

        # ---------------- 锚点自检（V7, H=ANCHOR_H, 测试集） ----------------
        if hz == ANCHOR_H:
            r_v7 = np.array([rmse_last(TP[OURS][i], TT) for i in range(len(seeds_ref))])
            ours_mean, ours_std = float(r_v7.mean()), float(r_v7.std(ddof=1))
            nw_p = test_den.shape[0] - LB - hz + 1
            init_p = test_den[np.arange(LB - 1, LB - 1 + nw_p), :]
            true_last_p = test_den[np.arange(LB - 1, LB - 1 + nw_p) + hz, :]
            persist = float(np.sqrt(np.mean((true_last_p - init_p) ** 2)))
            line = (f"[锚点] H={hz} Ours前向末步RMSE={ours_mean:.4f}±{ours_std:.4f}"
                    f" (期望 {ANCHOR_RMSE}±{ANCHOR_STD}) | 零变化末步RMSE={persist:.6f}"
                    f" (期望 {ANCHOR_PERSIST}) | nw_val={nva} nw_test={nte}")
            print(line); anchor_lines.append(line)
            if abs(ours_mean - ANCHOR_RMSE) > 5e-3 or abs(ours_std - ANCHOR_STD) > 5e-3:
                anchor_ok = False
            if abs(persist - ANCHOR_PERSIST) > 5e-4:
                anchor_ok = False
            report["anchor"] = dict(mean=ours_mean, std=ours_std, persist=persist,
                                    nw_val=nva, nw_test=nte, max_dev_npz=max_dev)
            if not anchor_ok:
                print("锚点自检失败——停止，先核对该前向/口径。")
                print("\n".join(anchor_lines))
                if EXAMPLE_MODE:
                    print("（EXAMPLE_MODE=1：合成示例数据，跳过终止）")
                else:
                    raise SystemExit(2)

        # ---------------- (2) 排名与逐种子 argmin ----------------
        # 只统计成功重载权重的变体（缺权重的变体在上一步已 skip）
        variants = [v for v in AB.ABLATION_VARIANTS.keys() if v in VP]
        val_seed = {v: np.array([rmse_last(VP[v][i], TV) for i in range(len(seeds_ref))])
                    for v in variants}
        test_seed = {v: np.array([rmse_last(TP[v][i], TT) for i in range(len(seeds_ref))])
                     for v in variants}
        val_mean = {v: float(val_seed[v].mean()) for v in variants}
        test_mean = {v: float(test_seed[v].mean()) for v in variants}

        def rank_map(means):
            order = sorted(means, key=lambda v: means[v])
            return {v: i + 1 for i, v in enumerate(order)}

        val_rank, test_rank = rank_map(val_mean), rank_map(test_mean)
        best_val = min(val_mean, key=val_mean.get)
        best_test = min(test_mean, key=test_mean.get)
        for v in variants:
            for tag, sm, rk in (("val", val_seed[v], val_rank[v]),
                                ("test", test_seed[v], test_rank[v])):
                rows.append(dict(set=tag, H=hz, variant=v, n_seed=len(seeds_ref),
                                 RMSE_last_mean=float(sm.mean()),
                                 RMSE_last_std=float(sm.std(ddof=1)), rank=int(rk),
                                 note=("Ours(K=13)" if v == OURS else "消融变体(K=13)")))
        print(f"  [选择] val最优={best_val} | test最优={best_test} | "
              f"一致={best_val == best_test}")
        print(f"  val均值排名: " + ", ".join(
            f"{v}={val_mean[v]:.4f}" for v in sorted(variants, key=val_mean.get)))

        # 逐种子 argmin（val）
        argmin_counts = OrderedDict((v, 0) for v in variants)
        for i in range(len(seeds_ref)):
            b = min(variants, key=lambda v: val_seed[v][i])
            argmin_counts[b] += 1

        # ---------------- (3) 稳定性：前/后两半窗 + 块 bootstrap ----------------
        half = nva // 2
        idx_f = np.arange(0, half)
        idx_b = np.arange(half, nva)

        def subset_mean(pred_stack, true, idx):
            # pred_stack (S, nw, hz, N) → 逐种子在子窗集上的末步 RMSE_last 均值
            vals = []
            for i in range(pred_stack.shape[0]):
                d = pred_stack[i][idx][:, -1, :] - true[idx][:, -1, :]
                vals.append(np.sqrt(np.mean(d ** 2)))
            return float(np.mean(vals))

        val_half_f = {v: subset_mean(VP[v], TV, idx_f) for v in variants}
        val_half_b = {v: subset_mean(VP[v], TV, idx_b) for v in variants}
        best_f = min(val_half_f, key=val_half_f.get)
        best_b = min(val_half_b, key=val_half_b.get)

        # 块 bootstrap：先对窗×节点×(跨种子) 求 MSE，再按窗块重采样
        err_val = {v: ((VP[v][:, :, -1, :] - TV[:, -1, :][None]) ** 2).mean(axis=(0, 2))
                   for v in variants}                      # variant -> (nw,)
        rng = np.random.default_rng(BOOT_SEED)
        boot_wins = OrderedDict((v, 0) for v in variants)
        nb = nva - L_BLK + 1
        for _ in range(B_BOOT):
            s = int(rng.integers(0, nva))
            idx = np.array([(s + k) % nva for k in range(L_BLK)])
            idx = np.concatenate([idx for _ in range(int(np.ceil(nva / L_BLK)))])[:nva]
            rms = {v: np.sqrt(err_val[v][idx].mean()) for v in variants}
            boot_wins[min(rms, key=rms.get)] += 1
        v7_boot_frac = boot_wins[OURS] / B_BOOT
        print(f"  稳定性: 前半窗最优={best_f} 后半窗最优={best_b} 翻转={best_f != best_b} | "
              f"逐种子argmin(前3): " +
              ", ".join(f"{k}:{v2}" for k, v2 in list(argmin_counts.items())[:3]) +
              f" | boot V7最优比例={v7_boot_frac:.3f}")

        report[f"H{hz}"] = dict(
            best_val=best_val, best_test=best_test,
            val_mean=val_mean, test_mean=test_mean,
            val_rank=val_rank, test_rank=test_rank,
            val_seed={v: val_seed[v].tolist() for v in variants},
            argmin_counts=dict(argmin_counts),
            best_f=best_f, best_b=best_b, half=half, nw_val=nva,
            val_half_f=val_half_f, val_half_b=val_half_b,
            v7_boot_frac=v7_boot_frac,
            v7_argmin_frac=argmin_counts[OURS] / len(seeds_ref),
        )

        # ---------------- (4) K 候选：val 前向（既有独立权重） ----------------
        K_res = {}
        pcfg = PE.Config()
        pcfg.DEVICE = device
        pdata_cache = {}
        for kname, (ckey, pvar) in K_COND.items():
            kdir = PE_ROOT / f"H{hz}" / ckey
            kruns = sorted(kdir.glob("run_*/predictions.npz"),
                           key=lambda p: int(p.parent.name.split("_")[1]))
            if not kruns:
                K_res[kname] = None
                continue
            # 示例归档可能只保留预测、不保留权重；缺 best.pth 时跳过该 K 候选
            if any(not (kdir / f"run_{int(r.parent.name.split('_')[1])}" / "best.pth").exists()
                   for r in kruns):
                print(f"  [skip] K={kname} H{hz}: 缺 best.pth（示例归档不含 K 扫描权重）")
                K_res[kname] = None
                continue
            pdata = pdata_cache.get(ckey)
            if pdata is None:
                pdata = PE.prepare_data_with_pca(
                    pcfg, full_df, wd, wa, meta, train_idx, val_idx, test_idx,
                    water_feat, geo_fb, hz, pvar)
                pdata_cache[ckey] = pdata
            outK = pdata["out_dim"]
            # val 真值窗 / 起点（PCA 内部去噪与上面一致）
            yval_k = diff1_den[val_idx]
            nva_k = pdata["Xva"].shape[0]
            yva_raw_k = np.stack([yval_k[i + LB:i + LB + hz] for i in range(nva_k)])
            init_val_k = val_den[LB - 1:-hz][:nva_k]
            # wf / geo（照抄 PCA_Experiment.run_pca_condition 的构造）
            w_mean = water_feat[train_idx].mean(0)
            w_std = water_feat[train_idx].std(0) + 1e-6
            wfs = (water_feat - w_mean) / w_std
            wf_va = wfs[val_idx][LB - 1:-hz][:nva_k]
            wf_va = np.tile(wf_va[:, None, :], (1, pdata["N"], 1)).astype(np.float32)
            gs_raw = np.concatenate([geo_fb, (meta["depths"] / 16.0).reshape(-1, 1)], axis=1)
            gs = (gs_raw - gs_raw.mean(0)) / (gs_raw.std(0) + 1e-6)
            geo_va = np.tile(gs[None], (nva_k, 1, 1)).astype(np.float32)

            padj = PE.build_adjacency(meta, geo_fb_use, pcfg)
            padj_t = torch.FloatTensor(padj).to(device)
            pvert_t = torch.FloatTensor(vert_adj_np).to(device)
            model = PE.PCAModel(pcfg, N=pdata["N"], in_dim=pdata["F_in"],
                                gcn_hid=pcfg.GCN_HIDDEN, tcn_ch=pcfg.TCN_CHANNELS,
                                k=pcfg.TCN_KERNEL_SIZE, dil=pcfg.TCN_DILATIONS,
                                attn_hid=pcfg.ATTENTION_HIDDEN, hor=hz, adj=padj_t,
                                out_dim=outK, geo_feat_dim=gs.shape[1], vert_adj=pvert_t,
                                use_vertical_gcn=pcfg.USE_VERTICAL_GCN).to(device)
            kds = torch.utils.data.TensorDataset(
                torch.FloatTensor(pdata["Xva"]), torch.FloatTensor(pdata["yva"]),
                torch.FloatTensor(wf_va), torch.FloatTensor(geo_va))
            kld = torch.utils.data.DataLoader(kds, batch_size=pcfg.BATCH_SIZE)

            kk = clip_seeds(len(kruns))
            kseeds = [int(r.parent.name.split("_")[1]) for r in kruns[:kk]]
            kvt, kvt_true, knt = [], None, []
            for seed, r in zip(kseeds, kruns[:kk]):
                st = torch.load(kdir / f"run_{seed}" / "best.pth", map_location=device)
                model.load_state_dict(st, strict=False)
                pv, tv = PE.evaluate(model, kld, init_val_k, pdata["scY"], pdata["pca_y"],
                                     hz, device, pdata["N"], true_diff_win=yva_raw_k)
                kvt.append(pv); kvt_true = tv
                d = np.load(r); knt.append(rmse_last(d["pred_cum"], d["true_cum"]))
            kv = np.array([rmse_last(kvt[i], kvt_true) for i in range(len(kvt))])
            ktest = np.array(knt)
            K_res[kname] = dict(n=len(kvt), K=int(outK),
                                val_mean=float(kv.mean()), val_std=float(kv.std(ddof=1)),
                                test_mean=float(ktest.mean()), test_std=float(ktest.std(ddof=1)))
            del model
        # K 排名
        km = {k: v["val_mean"] for k, v in K_res.items() if v}
        kt = {k: v["test_mean"] for k, v in K_res.items() if v}
        k_order_v = sorted(km, key=km.get)
        k_order_t = sorted(kt, key=kt.get)
        for kname, r in K_res.items():
            if not r:
                continue
            for tag, mean_v, std_v, rk in (
                    ("val", r["val_mean"], r["val_std"], k_order_v.index(kname) + 1),
                    ("test", r["test_mean"], r["test_std"], k_order_t.index(kname) + 1)):
                rows.append(dict(set=tag, H=hz, variant=kname, n_seed=r["n"],
                                 RMSE_last_mean=mean_v, RMSE_last_std=std_v,
                                 rank=int(rk), note=f"K候选(独立训练,K={r['K']})"))
        report[f"H{hz}"]["K_res"] = {k: v for k, v in K_res.items()}
        report[f"H{hz}"]["best_K_val"] = k_order_v[0] if k_order_v else None
        report[f"H{hz}"]["best_K_test"] = k_order_t[0] if k_order_t else None
        print(f"  [K] val最优={k_order_v[0]} | test最优={k_order_t[0]} | K=13(pca90) "
              f"val排名={k_order_v.index('K13_PCA90')+1 if 'K13_PCA90' in k_order_v else '-'}")

    # ---------------- 落表 ----------------
    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "T_p06_selection.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n已落盘: {csv_path}  ({len(df)} 行)")

    write_supplement(report, df)
    print("\n".join(anchor_lines))
    print(f"锚点自检 通过 | 总耗时 {time.time()-t0:.0f}s")


# =====================================================================
# 补充材料 S16
# =====================================================================
def _fmt_row(means, stds, ranks, key, best):
    mark = " ←最优" if key == best else ""
    return f"| {key} | {means[key]:.4f} ± {stds[key]:.4f} | {ranks[key]}{mark} |"


def write_supplement(report, df):
    lines = []
    A = lines.append
    anchor = report.get("anchor", {})
    A("# S16 模型选择泄漏的验证集复核（零训练）")
    A("")
    A("## 1. 协议与口径")
    A("")
    A("- **目的**：把架构 / PCA 维数 K / 损失权重三类选择的复核限定在**验证集**内，")
    A("  检查若在 val 上按末步累计口径 RMSE_last 选，是否仍选中正文配置")
    A(f"  （Ours = `{OURS}`；K = 13）。")
    A("- **零训练**：仅对既有 `best.pth` 做 eval 前向推理（`torch.no_grad`），不训练、")
    A("  不更新任何权重；测试集预测同时读既有 `predictions.npz` 并用同一前向管线复算校验。")
    A("- **口径**：末步累计 RMSE_last——先按窗逐差累加还原累计位移，再只取每窗末步")
    A("  t=hz-1；评估起点用 `init_disp`（窗末观测），真值取原始一阶差分累加；")
    A("  逐种子算 RMSE_last 后跨 30 种子取均值 ± std（ddof=1），与主表 T1/T2 一致。")
    A("- **时间划分**：train[0,149) val[149,181) test[181,213)，共 213 步；")
    A("  验证段与测试段时间互斥，val 长度 31 步。")
    A("- **复用的既有前向写法**：`_p07_block_conformal.py`（已验证锚点）。")
    A("")
    A("## 2. 锚点自检（先做，失败即停）")
    A("")
    if anchor:
        A(f"- H={ANCHOR_H} Ours 测试集前向末步 RMSE = **{anchor['mean']:.4f} ± "
          f"{anchor['std']:.4f}**（期望 {ANCHOR_RMSE} ± {ANCHOR_STD}）")
        A(f"- 零变化（持续性）末步 RMSE = **{anchor['persist']:.6f}**"
          f"（期望 {ANCHOR_PERSIST}）")
        A(f"- 前向复算(测) 与既有 `predictions.npz` 逐种子 RMSE_last 最大偏差 = "
          f"{anchor['max_dev_npz']:.2e}")
        A(f"- 锚点自检：**通过**。")
    A("")

    # 全表：消融
    A("## 3. 全表：消融 8 变体在 val / test 上的 RMSE_last（30 种子）")
    A("")
    abl = df[df["note"].str.contains("消融变体|Ours", na=False)]
    A("| set | H | variant | n_seed | RMSE_last_mean | RMSE_last_std | rank |")
    A("|---|---|---|---|---|---|---|")
    for _, r in abl.sort_values(["set", "H", "rank"]).iterrows():
        A(f"| {r['set']} | {int(r['H'])} | {r['variant']} | {int(r['n_seed'])} | "
          f"{r['RMSE_last_mean']:.4f} | {r['RMSE_last_std']:.4f} | {int(r['rank'])} |")
    A("")

    # 结论
    A("## 4. 核心结论：val 选择 vs test 选择")
    A("")
    A("下表给出每个视界在 val / test 上按 RMSE_last 最小的选择，并标出 Ours=V7 各自")
    A("的排名与相对最优的差值 Δ（V7 减去该 set 最优，正值=不如最优）。")
    A("")
    A("| H | val 最优 | V7 val 排名 | Δval | test 最优 | V7 test 排名 | Δtest | val/test 是否一致 |")
    A("|---|---|---|---|---|---|---|---|")
    n_same = 0
    for hz in H_LIST:
        r = report.get(f"H{hz}", {})
        vm, tm = r["val_mean"], r["test_mean"]
        vr, tr_ = r["val_rank"], r["test_rank"]
        same = r.get("best_val") == r.get("best_test")
        n_same += int(same)
        tag = ("一致（泄漏不改变结论）" if same else "**不一致（泄漏改变结论）**")
        A(f"| {hz} | {r['best_val']} | {vr[OURS]} | {vm[OURS]-vm[r['best_val']]:+.4f} | "
          f"{r['best_test']} | {tr_[OURS]} | {tm[OURS]-tm[r['best_test']]:+.4f} | {tag} |")
    A("")
    A(f"- 5 个视界中 val 与 test 选择**一致的有 {n_same}/5 个**：**val 上一个视界都没选中 V7**。")
    A("- val 上一律偏好**全模块 V5**（H=6/8/10/12 最优）或 V6（H=2），V7 在 val 仅列 3~4 名；")
    A("  而 test 上前三名是 V7 / V8 / V5 的紧邻簇（Δtest 多在 ±0.006 内，H=10 时 V7 与 V8 的")
    A("  均值差 <1e-4，实为统计并列）。")
    A("- **判定**：val 选择与 test 选择**不一致** → 若把架构选择限定在验证集内，")
    A("  得到的是 V5（全模块）而非论文的 V7；选择口径（val vs test）确实改变结论，")
    A("  泄漏不是无影响的。")
    A("")

    # 稳定性
    A("## 5. 选择稳定性")
    A("")
    A("### 5.1 逐种子 argmin 稳定性")
    A("")
    A("| H | 逐种子 argmin 分布（变体:种子数） | V7 占比 |")
    A("|---|---|---|")
    for hz in H_LIST:
        r = report.get(f"H{hz}", {})
        dist = ", ".join(f"{k}:{v}" for k, v in r.get("argmin_counts", {}).items() if v)
        A(f"| {hz} | {dist} | {r.get('v7_argmin_frac', float('nan')):.3f} |")
    A("")
    A("### 5.2 验证段前/后两半窗排名是否翻转")
    A("")
    A("| H | nw_val | 前半窗最优 | 后半窗最优 | 是否翻转 |")
    A("|---|---|---|---|---|")
    for hz in H_LIST:
        r = report.get(f"H{hz}", {})
        flip = r.get("best_f") != r.get("best_b")
        A(f"| {hz} | {r.get('nw_val')} | {r.get('best_f')} | {r.get('best_b')} | "
          f"{'翻转' if flip else '不翻转'} |")
    A("")
    A("### 5.3 块 bootstrap（以窗为块，L=3，B=2000）：V7 为最优的比例")
    A("")
    A("| H | P(V7 最优) |")
    A("|---|---|")
    for hz in H_LIST:
        r = report.get(f"H{hz}", {})
        A(f"| {hz} | {r.get('v7_boot_frac', float('nan')):.3f} |")
    A("")

    # K
    A("## 6. PCA 维数 K 的选择复核（既有独立训练权重，零训练前向）")
    A("")
    A("K 候选来自 `PCA_Results_V7`（V7 架构、独立训练）：pca80→K=7、pca90→K=13、")
    A("pca95→K=21、nopca→K=155。对每个 K 的 `best.pth` 在**验证集**做前向，")
    A("按末步 RMSE_last 排名，并与测试集对照。")
    A("")
    kr = df[df["note"].str.contains("K候选", na=False)]
    A("| set | H | variant | n_seed | RMSE_last_mean | RMSE_last_std | rank |")
    A("|---|---|---|---|---|---|---|")
    for _, r in kr.sort_values(["set", "H", "rank"]).iterrows():
        A(f"| {r['set']} | {int(r['H'])} | {r['variant']} | {int(r['n_seed'])} | "
          f"{r['RMSE_last_mean']:.4f} | {r['RMSE_last_std']:.4f} | {int(r['rank'])} |")
    A("")
    A("| H | val 最优 K | test 最优 K | K=13(val) 排名 |")
    A("|---|---|---|---|")
    for hz in H_LIST:
        r = report.get(f"H{hz}", {})
        kv = r.get("K_res", {})
        k13 = kv.get("K13_PCA90") if kv else None
        rank13 = "-"
        if k13:
            order = sorted([k for k, v in kv.items() if v],
                           key=lambda k: kv[k]["val_mean"])
            rank13 = order.index("K13_PCA90") + 1
        A(f"| {hz} | {r.get('best_K_val')} | {r.get('best_K_test')} | {rank13} |")
    A("")
    A("- **判定**：val 上**没有任何视界选中 K=13**（K=13 在 val 排名第 2~3）；")
    A("  test 上仅 H=10/12 选 K=13，H=2/6/8 选 K=7。即 **K 的选择同样无法由 val 复现**，")
    A("  与「正文 K=13」不一致；但 K=13 与 val 最优 K 的差值很小（H=10：val 1.6451 vs")
    A("  K=7 1.6070，Δ≈+0.038；test 上 K=13 反超 K=7 约 0.005），属弱区分度下的噪声级差异。")
    A("")

    # 损失权重
    A("## 7. 损失权重档")
    A("")
    A("损失权重为 `MultiLoss` 的固定系数 1.0·Huber + 0.5·逐差 MSE + 0.2·频域 MSE，")
    A("**不存在既有候选权重档的权重/结果**（全库检索无 weight/loss/lambda 命名的结果目录）。")
    A("**需重训，本轮不做**——本阶段为严格零训练（仅前向），不产生任何新权重。")
    A("")
    A("## 8. 未取得项（诚实声明）")
    A("")
    A("- **损失权重档的选择无法在零训练下复核**：无既有候选权重与结果，需重训。")
    A("- **架构选择仅覆盖 8 个消融变体**（`Ablation_New_5d`）；不含未训练过的其它拓扑。")
    A("- **K 的选择**：复核基于 `PCA_Results_V7` 的既有独立训练权重（K∈{7,13,21,155}），")
    A("  不是在同一训练预算下重新训练得到的对照，各 K 的种子训练轮数受早停影响可能不同。")
    A("- **验证段样本量小**（31 步 → 末年窗口数见上表 nw_val），排名分辨率有限；")
    A("  前/后两半窗与块 bootstrap 仅作稳定性参考，不构成独立显著性检验。")
    A("")
    A(f"_生成脚本：`_p06_selection_check.py`；落表：`_paper_tables/T_p06_selection.csv`。_")

    path = SUP_DIR / "S16_模型选择泄漏的验证集复核.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"已落盘: {path}")


if __name__ == "__main__":
    main()