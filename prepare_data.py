"""读取 CSV（分块）-> 抗混叠降采样到 10 kHz -> 缓存 -> 50 s 截取 -> 时间段划分 -> 切窗 -> 拟合归一化参数。
用法：D:\\Anaconda\\envs\\DL3.91\\python.exe prepare_data.py [--force]
"""
import argparse

import numpy as np

from vibddpm.data import norm_path, prepare
from vibddpm.utils import load_config

ap = argparse.ArgumentParser()
ap.add_argument('--config', default=None)
ap.add_argument('--force', action='store_true', help='忽略缓存重新降采样')
args = ap.parse_args()

cfg = load_config(args.config)
splits, nz, stats = prepare(cfg, force_cache=args.force)
print(f'\n归一化参数 {stats["version"]} 已保存到 {norm_path(cfg)}')
print('通道        mean        std')
for ch, m, s in zip(stats['channels'], stats['mean'], stats['std']):
    print(f'{ch:8s} {m:+.5f}  {s:.5f}')
for l in cfg['data']['labels']:
    print(f'标签 {l} ({cfg["data"]["label_desc"][l]}): 段界 {stats["split"][l]["bounds"]} 窗口数 {stats["split"][l]["n_windows"]}')
for k, v in splits.items():
    print(f'{k}: x {v["x"].shape}')
    for l in cfg['data']['labels']:
        xs = v['x'][v['y'] == l]
        print(f'   标签 {l}: 各通道 RMS(V) ' + ' '.join(f'{r:.3f}' for r in np.sqrt((xs ** 2).mean((0, 2)))))
