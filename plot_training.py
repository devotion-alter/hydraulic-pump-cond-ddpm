"""训练过程曲线：训练损失（按工况）、验证损失（按工况）、条件敏感性（交换/null 标签 - 正确标签）。
可在训练进行中反复运行，多个 run 叠加对比。

用法：D:\\Anaconda\\envs\\DL3.91\\python.exe plot_training.py [--runs cfg_ddpm_v1 cfg_ddpm_v2]
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from vibddpm.utils import ROOT

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ap = argparse.ArgumentParser()
ap.add_argument('--runs', nargs='*', default=None, help='默认 outputs 下所有含 train_log.csv 的 run')
ap.add_argument('--out', default=str(ROOT / 'outputs' / 'training_curves.png'))
args = ap.parse_args()

runs = args.runs or sorted(p.parent.name for p in (ROOT / 'outputs').glob('*/train_log.csv') if 'smoke' not in p.parent.name)
fig, axes = plt.subplots(2, 2, figsize=(15, 9))
(ax_tr, ax_va), (ax_lab, ax_gap) = axes
styles = ['-', '--', ':', '-.']
cols = {'19': '#1f77b4', '20': '#ff7f0e'}
for ri, run in enumerate(runs):
    df = pd.read_csv(ROOT / 'outputs' / run / 'train_log.csv')
    labels = [c for c in df.columns if c.isdigit()]
    ls = styles[ri % len(styles)]
    tr, va = df[df.kind == 'train'], df[df.kind == 'val']
    ax_tr.plot(tr.step, tr['all'].rolling(5, min_periods=1).mean(), ls, color=f'C{ri}', label=f'{run}')
    for l in labels:
        ax_lab.plot(tr.step, tr[l].rolling(5, min_periods=1).mean(), ls, color=cols.get(l), alpha=0.6, lw=1,
                    label=f'{run} 训练 {l}')
        if len(va):
            ax_lab.plot(va.step, va[l], ls, color=cols.get(l), marker='o', ms=3, lw=1.8, label=f'{run} 验证 {l}')
    if len(va):
        ax_va.plot(va.step, va['all'], ls, color=f'C{ri}', marker='o', ms=3, label=f'{run} 正确标签')
        ax_va.plot(va.step, va['null_all'], ls, color=f'C{ri}', alpha=0.4, label=f'{run} null 标签')
        best = va.loc[va['all'].idxmin()]
        ax_va.scatter([best.step], [best['all']], s=90, facecolor='none', edgecolor=f'C{ri}', lw=2, zorder=5)
        ax_va.annotate(f'best {best["all"]:.4f}\n@{int(best.step)}', (best.step, best['all']),
                       textcoords='offset points', xytext=(5, -28), fontsize=8, color=f'C{ri}')
        ax_gap.plot(va.step, (va['swapped_all'] - va['all']) / va['all'] * 100, ls, color=f'C{ri}', marker='o',
                    ms=3, label=f'{run} 交换标签')
        ax_gap.plot(va.step, (va['null_all'] - va['all']) / va['all'] * 100, ls, color=f'C{ri}', alpha=0.4,
                    marker='x', ms=3, label=f'{run} null 标签')
    last = df.step.max()
    print(f'{run}: 当前 step {last}' + (f'，最佳验证 {va["all"].min():.4f} @ {int(va.loc[va["all"].idxmin()].step)}，'
                                        f'最新验证 {va["all"].iloc[-1]:.4f}' if len(va) else ''))

ax_tr.set_title('训练损失（全部，滑动平均）'); ax_va.set_title('验证损失（EMA 模型，固定 t 与噪声）')
ax_lab.set_title('按工况：训练（细线）vs 验证（带点）'); ax_gap.set_title('条件敏感性：错误条件相对正确条件的损失增幅 (%)')
ax_gap.axhline(0, color='k', lw=0.8)
for ax in axes.ravel():
    ax.set_xlabel('step'); ax.grid(alpha=0.3); ax.legend(fontsize=7)
for ax in (ax_tr, ax_va, ax_lab):
    ax.set_ylabel('MSE (pred_noise)')
ax_gap.set_ylabel('%')
fig.suptitle('条件 1D DDPM 训练过程', fontsize=13)
fig.tight_layout()
fig.savefig(args.out, dpi=120)
print(f'已保存 {args.out}')
