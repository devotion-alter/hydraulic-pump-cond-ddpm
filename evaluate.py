"""生成信号质量评价（按工况分别）：时域、FFT、PSD、RMS/Std/Peak/Kurtosis 分布、通道相关矩阵、
不同 guidance_scale 对比，以及“生成结果是否体现对应工况”的条件有效性检查。

用法：D:\\Anaconda\\envs\\DL3.91\\python.exe evaluate.py   （需先运行 sample.py）
参照集合为 Test 段真实窗口（训练未见），并给出 真实Train vs 真实Test 的基线差异作为参考下限。
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from vibddpm.data import build_splits
from vibddpm.metrics import (NearestCentroid, corr_distance, corr_matrices, db, feature_vector, fft_amp, psd,
                             psd_db_distance, time_features)
from vibddpm.utils import load_config, run_dir, save_json

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
REAL_C, GEN_C = '#1f77b4', '#d62728'
W_COLORS = {1.0: '#2ca02c', 2.0: '#d62728', 3.0: '#9467bd'}


def grid9(title, sharex=True):
    fig, axes = plt.subplots(3, 3, figsize=(15, 9), sharex=sharex)
    fig.suptitle(title, fontsize=13)
    return fig, axes.ravel()


def plot_time(real, gen, ch, fs, title, path, k=1):
    fig, axes = grid9(title)
    t = np.arange(real.shape[-1]) / fs * 1000
    for c, ax in enumerate(axes):
        for i in range(k):
            ax.plot(t, real[i, c], color=REAL_C, lw=0.7, alpha=0.9, label='真实' if i == 0 else None)
            ax.plot(t, gen[i, c], color=GEN_C, lw=0.7, alpha=0.75, label='生成' if i == 0 else None)
        ax.set_title(ch[c], fontsize=10); ax.grid(alpha=0.3)
        if c >= 6: ax.set_xlabel('时间 (ms)')
        if c % 3 == 0: ax.set_ylabel('电压 (V)')
    axes[0].legend(loc='upper right', fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_spectrum(f, real, gen, ch, title, ylabel, path, band=True):
    """real/gen: [N, C, F] 线性幅值/功率；画均值 + 10~90% 分位带（对数纵轴）。"""
    fig, axes = grid9(title)
    for c, ax in enumerate(axes):
        for arr, col, lab in ((real, REAL_C, '真实'), (gen, GEN_C, '生成')):
            ax.semilogy(f, arr[:, c].mean(0), color=col, lw=1.0, label=f'{lab} 均值')
            if band:
                lo, hi = np.percentile(arr[:, c], [10, 90], axis=0)
                ax.fill_between(f, lo, hi, color=col, alpha=0.18, lw=0)
        ax.set_title(ch[c], fontsize=10); ax.grid(alpha=0.3, which='both')
        if c >= 6: ax.set_xlabel('频率 (Hz)')
        if c % 3 == 0: ax.set_ylabel(ylabel)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_features(fr, fg, ch, title, path):
    fig, axes = plt.subplots(4, 1, figsize=(15, 13))
    fig.suptitle(title, fontsize=13)
    C = len(ch)
    for ax, name in zip(axes, fr):
        pos = np.arange(C)
        b1 = ax.boxplot([fr[name][:, c] for c in range(C)], positions=pos - 0.18, widths=0.3, patch_artist=True,
                        showfliers=False)
        b2 = ax.boxplot([fg[name][:, c] for c in range(C)], positions=pos + 0.18, widths=0.3, patch_artist=True,
                        showfliers=False)
        for b, col in ((b1, REAL_C), (b2, GEN_C)):
            for p in b['boxes']: p.set(facecolor=col, alpha=0.45)
            for m in b['medians']: m.set(color='k')
        ax.set_xticks(pos); ax.set_xticklabels(ch); ax.set_ylabel(name + ('' if name == 'Kurtosis' else ' (V)'))
        ax.grid(alpha=0.3, axis='y')
        ax.legend([b1['boxes'][0], b2['boxes'][0]], ['真实', '生成'], fontsize=8, loc='upper right')
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_corr(cr, cg, ch, title, path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    fig.suptitle(title, fontsize=13)
    short = [c.replace('电压_', 'V') for c in ch]
    for ax, m, t, cmap, vr in ((axes[0], cr, '真实', 'RdBu_r', 1), (axes[1], cg, '生成', 'RdBu_r', 1),
                               (axes[2], cg - cr, '生成 − 真实', 'PuOr_r', 0.5)):
        im = ax.imshow(m, cmap=cmap, vmin=-vr, vmax=vr)
        ax.set_xticks(range(len(ch))); ax.set_yticks(range(len(ch)))
        ax.set_xticklabels(short); ax.set_yticklabels(short); ax.set_title(t)
        for i in range(len(ch)):
            for j in range(len(ch)):
                ax.text(j, i, f'{m[i, j]:.2f}', ha='center', va='center', fontsize=6.5)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_guidance_psd(f, real_by_label, gen_psd, labels, scales, ch, desc, path):
    """每工况一行：九通道平均 PSD 的通道平均 + 各 w；另画各通道。"""
    fig, axes = plt.subplots(len(labels), 9, figsize=(26, 3.3 * len(labels)), sharex=True, squeeze=False)
    fig.suptitle('不同 guidance_scale 的平均 PSD（Test 真实 vs 生成）', fontsize=13)
    for r, l in enumerate(labels):
        for c in range(9):
            ax = axes[r, c]
            ax.semilogy(f, real_by_label[l][:, c].mean(0), color='k', lw=1.3, label='真实')
            for w in scales:
                ax.semilogy(f, gen_psd[(l, w)][:, c].mean(0), color=W_COLORS.get(w), lw=0.9, label=f'w={w:g}')
            ax.set_title(f'{l} {ch[c]}', fontsize=9); ax.grid(alpha=0.3)
            if r == len(labels) - 1: ax.set_xlabel('Hz')
        axes[r, 0].set_ylabel(f'{l} {desc[l]}\nPSD (V²/Hz)', fontsize=9)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_cross(labels, scales, cross, path):
    """生成(标签, w) 与各真实工况的 PSD dB 距离热图：对角线应最小。"""
    rows = [f'gen {l} w={w:g}' for w in scales for l in labels]
    M = np.array([[cross[(l, w)][lr] for lr in labels] for w in scales for l in labels])
    fig, ax = plt.subplots(figsize=(5 + len(labels), 0.55 * len(rows) + 1.5))
    im = ax.imshow(M, cmap='viridis_r')
    ax.set_xticks(range(len(labels))); ax.set_xticklabels([f'real {l}' for l in labels])
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', color='w', fontsize=9)
    ax.set_title('平均 PSD 距离 (dB)：越小越接近')
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None)
    ap.add_argument('--samples', default='samples')
    ap.add_argument('--run_name', default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.run_name:
        cfg['project']['run_name'] = args.run_name
    rd = run_dir(cfg)
    sd = rd / args.samples
    fig_dir = rd / 'figures'; fig_dir.mkdir(exist_ok=True)
    d, ec = cfg['data'], cfg['eval']
    fs, ch, labels, desc = d['target_fs'], d['channels'], d['labels'], d['label_desc']
    scales = cfg['sample']['guidance_scales']
    w0 = cfg['sample']['default_guidance_scale']
    nps = ec['welch_nperseg']

    splits, _, _ = build_splits(cfg)
    real_te = {l: splits['test']['x'][splits['test']['y'] == l][:ec['num_real']] for l in labels}
    real_tr = {l: splits['train']['x'][splits['train']['y'] == l] for l in labels}
    gen = {(l, w): np.load(sd / f'gen_label{l}_w{w:g}.npy') for l in labels for w in scales}
    for k, v in gen.items():
        assert v.shape[1:] == (9, cfg['model']['seq_length']), (k, v.shape)

    # 工况判别器：在真实 Train 窗口上拟合
    clf = NearestCentroid().fit(
        np.concatenate([feature_vector(real_tr[l], fs, nps) for l in labels]),
        np.concatenate([np.full(len(real_tr[l]), l) for l in labels]))

    f_psd, _ = psd(real_te[labels[0]][:1], fs, nps)
    P = {('real', l): psd(real_te[l], fs, nps)[1] for l in labels}
    P.update({('train', l): psd(real_tr[l], fs, nps)[1] for l in labels})
    P.update({('gen', l, w): psd(gen[(l, w)], fs, nps)[1] for l in labels for w in scales})
    CR = {l: corr_matrices(real_te[l]) for l in labels}
    FR = {l: time_features(real_te[l]) for l in labels}

    report = {'baseline_real_train_vs_test': {}, 'per_condition': {}, 'condition_check': {}}
    for l in labels:
        report['baseline_real_train_vs_test'][l] = dict(
            psd_db=psd_db_distance(P[('train', l)], P[('real', l)]),
            corr=corr_distance(corr_matrices(real_tr[l]), CR[l]))
    # 真实工况之间的差异（生成两工况之间的差异应与之同量级，否则可能趋同）
    report['real_between_conditions_psd_db'] = psd_db_distance(P[('real', labels[0])], P[('real', labels[1])])
    report['real_test_classifier_acc'] = {
        l: float((clf.predict(feature_vector(real_te[l], fs, nps)) == l).mean()) for l in labels}

    cross = {}
    for w in scales:
        for l in labels:
            g = gen[(l, w)]
            fg, cg = time_features(g), corr_matrices(g)
            cross[(l, w)] = {lr: psd_db_distance(P[('gen', l, w)], P[('real', lr)]) for lr in labels}
            pred = clf.predict(feature_vector(g, fs, nps))
            report['per_condition'].setdefault(str(l), {})[f'w={w:g}'] = dict(
                n=int(len(g)),
                psd_db_to_same_real=cross[(l, w)][l],
                psd_db_to_other_real={str(lr): cross[(l, w)][lr] for lr in labels if lr != l},
                corr_dist=corr_distance(cg, CR[l]),
                rms_rel_err_per_ch=((fg['RMS'].mean(0) - FR[l]['RMS'].mean(0)) / FR[l]['RMS'].mean(0)).round(4).tolist(),
                kurtosis_real_vs_gen=[FR[l]['Kurtosis'].mean().round(3).item(), fg['Kurtosis'].mean().round(3).item()],
                classifier_acc=float((pred == l).mean()),
            )
        report['condition_check'][f'w={w:g}'] = dict(
            gen_between_conditions_psd_db=psd_db_distance(P[('gen', labels[0], w)], P[('gen', labels[1], w)]),
            all_closer_to_own_condition=bool(all(cross[(l, w)][l] == min(cross[(l, w)].values()) for l in labels)),
        )

    # ---------------- 图表（默认 w）
    for l in labels:
        g, r = gen[(l, w0)], real_te[l]
        tag = f'标签 {l} {desc[l]}，w={w0:g}'
        plot_time(r, g, ch, fs, f'时域波形：真实(Test) vs 生成 — {tag}', fig_dir / f'{l}_1_time.png', ec['plot_windows'])
        fa, ar = fft_amp(r, fs); _, ag = fft_amp(g, fs)
        plot_spectrum(fa, ar, ag, ch, f'FFT 幅值谱（窗口平均，10–90% 带）— {tag}，fs=10 kHz', '幅值 (V)',
                      fig_dir / f'{l}_2_fft.png')
        plot_spectrum(f_psd, P[('real', l)], P[('gen', l, w0)], ch,
                      f'Welch PSD（{len(r)} 真实 / {len(g)} 生成窗口，均值 + 10–90% 分布）— {tag}', 'PSD (V²/Hz)',
                      fig_dir / f'{l}_3_psd.png')
        plot_features(FR[l], time_features(g), ch, f'每通道 RMS / Std / Peak / Kurtosis 分布 — {tag}',
                      fig_dir / f'{l}_4_features.png')
        plot_corr(CR[l].mean(0), corr_matrices(g).mean(0), ch, f'九通道相关矩阵（窗口平均）— {tag}',
                  fig_dir / f'{l}_5_corr.png')
    plot_guidance_psd(f_psd, {l: P[('real', l)] for l in labels}, {(l, w): P[('gen', l, w)] for l in labels for w in scales},
                      labels, scales, ch, desc, fig_dir / 'guidance_psd.png')
    plot_cross(labels, scales, cross, fig_dir / 'condition_cross_psd_distance.png')
    # 两工况真实与生成的通道 RMS 对比（检查工况差异是否保留）
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(9); wd = 0.2
    for i, l in enumerate(labels):
        ax.bar(x + (2 * i - 1.5) * wd, FR[l]['RMS'].mean(0), wd, label=f'真实 {l}', color=['#1f77b4', '#ff7f0e'][i])
        ax.bar(x + (2 * i - 0.5) * wd, time_features(gen[(l, w0)])['RMS'].mean(0), wd, label=f'生成 {l} (w={w0:g})',
               color=['#1f77b4', '#ff7f0e'][i], alpha=0.45, hatch='//')
    ax.set_xticks(x); ax.set_xticklabels(ch); ax.set_ylabel('平均 RMS (V)'); ax.legend(); ax.grid(alpha=0.3, axis='y')
    ax.set_title('两工况各通道平均 RMS：真实 vs 生成'); fig.tight_layout(); fig.savefig(fig_dir / 'rms_by_condition.png', dpi=120)
    plt.close(fig)

    save_json(report, rd / 'eval_report.json')
    # ---------------- 文本摘要
    print('\n===== 评价摘要 =====')
    print(f'真实 Train vs Test 基线 PSD 距离(dB): ' +
          ', '.join(f'{l}: {v["psd_db"]:.2f}' for l, v in report['baseline_real_train_vs_test'].items()))
    print(f'真实两工况之间 PSD 距离(dB): {report["real_between_conditions_psd_db"]:.2f}')
    print(f'工况判别器在真实 Test 上准确率: {report["real_test_classifier_acc"]}')
    for w in scales:
        cc = report['condition_check'][f'w={w:g}']
        print(f'\n-- w={w:g}: 生成两工况之间 PSD 距离 {cc["gen_between_conditions_psd_db"]:.2f} dB，'
              f'各工况均更接近自身真实工况: {cc["all_closer_to_own_condition"]}')
        for l in labels:
            r = report['per_condition'][str(l)][f'w={w:g}']
            print(f'   标签 {l}: PSD→同工况 {r["psd_db_to_same_real"]:.2f} dB / →其他 {r["psd_db_to_other_real"]} | '
                  f'相关矩阵距离 {r["corr_dist"]:.3f} | 判别器归为本工况 {r["classifier_acc"] * 100:.1f}% | '
                  f'峭度 真/生 {r["kurtosis_real_vs_gen"]}')
    print(f'\n图表: {fig_dir}\n报告: {rd / "eval_report.json"}')


if __name__ == '__main__':
    main()
