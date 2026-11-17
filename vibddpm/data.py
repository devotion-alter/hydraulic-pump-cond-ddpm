"""CSV 分块读取、抗混叠降采样、缓存、时间段划分、切窗与归一化。"""
import hashlib
import os
import time
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from .utils import ROOT, load_json, save_json


# ------------------------------------------------------------------ CSV 读取
def read_header(path, encoding, header_lines):
    with open(path, 'rb') as f:
        meta = [f.readline().decode(encoding).strip() for _ in range(header_lines)]
        cols = f.readline().decode(encoding).strip().split(',')
    info = dict(m.split(':', 1) for m in meta if ':' in m)
    return info, cols


def iter_csv_chunks(path, cfg_data, chunk_rows=None):
    """按列名读取指定 9 路通道（与物理列位置无关），逐块返回 float64 数组 [n, 9]。"""
    channels = cfg_data['channels']
    info, cols = read_header(path, cfg_data['encoding'], cfg_data['header_lines'])
    missing = [c for c in channels if c not in cols]
    assert not missing, f'{path} 缺少列 {missing}，实际列: {cols}'
    assert int(info.get('SampleRate', cfg_data['raw_fs'])) == cfg_data['raw_fs'], f'采样率不符: {info}'
    reader = pd.read_csv(
        path, skiprows=cfg_data['header_lines'], header=0, usecols=channels,
        encoding=cfg_data['encoding'], chunksize=chunk_rows or cfg_data['read_chunk_rows'],
        dtype={c: np.float64 for c in channels}, engine='c',
    )
    for df in reader:
        yield df[channels].to_numpy(np.float64)   # 显式按配置顺序排列，保证通道对应关系


# ------------------------------------------------------------------ 分块降采样
class StreamingResampler:
    """带重叠上下文的分块 resample_poly。

    resample_poly(up=1, down=q) 是零相位 FIR：输出 y[j] 对应输入 x[q*j]。
    每次处理时让缓冲区左右各保留 ctx 个输入点作为滤波上下文，只输出不受块边界影响的部分，
    因此结果与对整段信号一次性 resample_poly 一致（仅文件首尾存在全局边缘效应）。
    """

    def __init__(self, up, down, ctx):
        g = gcd(up, down)
        self.up, self.down = up // g, down // g
        assert self.up == 1, '当前实现针对整数倍降采样'
        assert ctx % self.down == 0
        self.ctx = ctx
        self.buf = None
        self.started = False   # 已经输出过（缓冲区首部只是上下文）

    def push(self, x):
        self.buf = x if self.buf is None else np.concatenate([self.buf, x], axis=0)
        q, ctx = self.down, self.ctx
        n = len(self.buf)
        j1 = (n - ctx) // q                       # 右侧保留 ctx 个点上下文
        j0 = ctx // q if self.started else 0
        if j1 <= j0:
            return None
        y = resample_poly(self.buf, 1, q, axis=0)
        out = y[j0:j1]
        cut = q * j1 - ctx                        # 新缓冲区起点（保留左上下文 ctx 个点）
        self.buf = self.buf[cut:]
        self.started = True
        return out

    def flush(self):
        if self.buf is None or len(self.buf) == 0:
            return None
        y = resample_poly(self.buf, 1, self.down, axis=0)
        j0 = self.ctx // self.down if self.started else 0
        self.buf = None
        return y[j0:]


def downsample_file(path, cfg_data, max_rows=None, chunk_rows=None):
    q = cfg_data['raw_fs'] // cfg_data['target_fs']
    assert q * cfg_data['target_fs'] == cfg_data['raw_fs']
    chunk_rows = chunk_rows or cfg_data['read_chunk_rows']
    assert chunk_rows % q == 0
    rs = StreamingResampler(1, q, cfg_data['resample_context'])
    outs, n_raw = [], 0
    for chunk in iter_csv_chunks(path, cfg_data, chunk_rows):
        if max_rows is not None and n_raw + len(chunk) > max_rows:
            chunk = chunk[: max_rows - n_raw]
        n_raw += len(chunk)
        o = rs.push(chunk)
        if o is not None:
            outs.append(o)
        if max_rows is not None and n_raw >= max_rows:
            break
    o = rs.flush()
    if o is not None:
        outs.append(o)
    return np.concatenate(outs, 0), n_raw


def cache_path(cfg, label):
    d = cfg['data']
    return ROOT / d['cache_dir'] / f"{label}_{d['target_fs']}Hz.npy"


def build_cache(cfg, force=False):
    """对每个工况 CSV 降采样并缓存为 npy（float32, [T, 9]），原始 CSV 只读。"""
    d = cfg['data']
    (ROOT / d['cache_dir']).mkdir(parents=True, exist_ok=True)
    for label in d['labels']:
        src = ROOT / d['raw_dir'] / f'{label}.csv'
        out = cache_path(cfg, label)
        meta_p = out.with_suffix('.json')
        st = os.stat(src)
        sig = dict(src=str(src), size=st.st_size, mtime=st.st_mtime, channels=d['channels'],
                   raw_fs=d['raw_fs'], target_fs=d['target_fs'], ctx=d['resample_context'])
        if not force and out.exists() and meta_p.exists():
            m = load_json(meta_p)
            if all(m.get(k) == v for k, v in sig.items()):
                print(f'[cache] 标签 {label}: 已存在 {out.name}，跳过')
                continue
        t0 = time.time()
        print(f'[cache] 标签 {label}: 读取 {src.name} 并降采样 {d["raw_fs"]}->{d["target_fs"]} Hz ...')
        y, n_raw = downsample_file(src, d)
        np.save(out, y.astype(np.float32))
        sig.update(n_raw=n_raw, n_out=len(y), duration_s=n_raw / d['raw_fs'],
                   method='scipy.signal.resample_poly(up=1, down=10, window=kaiser 5.0) chunked w/ context',
                   elapsed_s=round(time.time() - t0, 1))
        save_json(sig, meta_p)
        print(f'[cache]   原始 {n_raw} 行 ({n_raw / d["raw_fs"]:.2f} s) -> {len(y)} 点，用时 {time.time() - t0:.1f}s')


def load_segment(cfg, label):
    """读取缓存并截取统一时长段 [edge_trim, edge_trim + segment_duration)。"""
    d = cfg['data']
    x = np.load(cache_path(cfg, label))
    fs = d['target_fs']
    s = int(round(d['edge_trim_s'] * fs))
    n = int(round(d['segment_duration_s'] * fs))
    assert s + n <= len(x), f'标签 {label} 时长不足: 需要 {(s + n) / fs:.2f}s, 实际 {len(x) / fs:.2f}s'
    return x[s:s + n]


# ------------------------------------------------------------------ 划分 / 切窗
def split_bounds(n, ratio, gap):
    """连续时间段划分：[train] gap [val] gap [test]，返回各段 (start, end)。"""
    avail = n - 2 * gap
    n_tr = int(avail * ratio['train'])
    n_va = int(avail * ratio['val'])
    tr = (0, n_tr)
    va = (n_tr + gap, n_tr + gap + n_va)
    te = (va[1] + gap, n)
    return {'train': tr, 'val': va, 'test': te}


def make_windows(x, win, stride):
    """x: [T, C] -> [N, C, win]，窗口完全落在本段内。"""
    T = len(x)
    if T < win:
        return np.zeros((0, x.shape[1], win), np.float32), np.zeros(0, np.int64)
    starts = np.arange(0, T - win + 1, stride)
    idx = starts[:, None] + np.arange(win)[None]
    return np.ascontiguousarray(x[idx].transpose(0, 2, 1)), starts


def build_splits(cfg):
    """返回 {split: {'x': [N,9,1024] 原始电压, 'y': 标签(文件编号), 'start': 段内起点}} 及划分信息。"""
    d = cfg['data']
    win, stride, gap = d['window_size'], d['stride'], d['split_gap']
    assert gap >= win, 'split_gap 必须 >= window_size，避免跨集合窗口重叠'
    out = {k: {'x': [], 'y': [], 'start': []} for k in ('train', 'val', 'test')}
    info, segments = {}, {}
    for label in d['labels']:
        seg = load_segment(cfg, label)
        b = split_bounds(len(seg), d['split_ratio'], gap)
        info[label] = {'bounds': {k: [int(v[0]), int(v[1])] for k, v in b.items()}, 'n_windows': {}}
        for k, (s, e) in b.items():
            w, starts = make_windows(seg[s:e], win, stride)
            out[k]['x'].append(w)
            out[k]['y'].append(np.full(len(w), label, np.int64))
            out[k]['start'].append(starts + s)
            segments.setdefault(k, []).append(seg[s:e])
            info[label]['n_windows'][k] = int(len(w))
    for k in out:
        out[k] = {kk: np.concatenate(vv, 0) for kk, vv in out[k].items()}
        out[k]['segments'] = segments[k]          # 各工况连续段 [T, C]（按 labels 顺序），供随机裁剪
    return out, info, segments


# ------------------------------------------------------------------ 归一化
def fit_norm(cfg, train_segments, split_info):
    """两工况共用：在所有工况的训练段连续数据上拟合逐通道 mean/std（不是逐窗口）。"""
    x = np.concatenate(train_segments, 0).astype(np.float64)
    mean, std = x.mean(0), x.std(0)
    payload = dict(channels=cfg['data']['channels'], mean=mean.tolist(), std=std.tolist(),
                   fs=cfg['data']['target_fs'], labels=cfg['data']['labels'], split=split_info)
    h = hashlib.sha1(np.concatenate([mean, std]).tobytes()).hexdigest()[:8]
    payload['version'] = f'norm_{h}'
    payload['note'] = 'z = (x - mean) / std，逐通道；参数仅由所有工况训练段拟合，两工况共用，保留工况间幅值差异'
    return payload


def norm_path(cfg):
    return ROOT / cfg['data']['cache_dir'] / cfg['data']['norm_file']


class Normalizer:
    def __init__(self, stats):
        self.mean = np.asarray(stats['mean'], np.float32)[:, None]   # [C,1]
        self.std = np.asarray(stats['std'], np.float32)[:, None]
        self.version = stats['version']
        self.channels = stats['channels']

    def __call__(self, x):          # [..., C, L]
        return (x - self.mean) / self.std

    def inverse(self, z):
        return z * self.std + self.mean


def prepare(cfg, force_cache=False):
    build_cache(cfg, force=force_cache)
    splits, info, segments = build_splits(cfg)
    stats = fit_norm(cfg, segments['train'], info)
    p = norm_path(cfg)
    if p.exists():
        old = load_json(p)
        if old['version'] != stats['version']:
            print(f'[norm] 归一化参数更新: {old["version"]} -> {stats["version"]}')
    save_json(stats, p)
    return splits, Normalizer(stats), stats


class RandomCropDataset(Dataset):
    """训练增强：在每个工况的连续训练段内随机起点裁剪 window_size 长窗口（数据仍只来自训练段，无泄漏）。
    每次取样按工况均衡；__len__ 仅定义一个 epoch 的名义长度。"""

    def __init__(self, segments_norm, labels, cfg, epoch_len):
        from .utils import labels_to_index
        self.segs = [torch.from_numpy(np.ascontiguousarray(s.T, dtype=np.float32)) for s in segments_norm]  # [C,T]
        self.idx = torch.from_numpy(labels_to_index(cfg, labels))
        self.win = cfg['data']['window_size']
        self.epoch_len = epoch_len

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, i):
        k = i % len(self.segs)
        seg = self.segs[k]
        s = np.random.randint(0, seg.shape[1] - self.win + 1)
        return seg[:, s:s + self.win].clone(), self.idx[k]


class WindowDataset(Dataset):
    def __init__(self, x_norm, labels, cfg):
        from .utils import labels_to_index
        self.x = torch.from_numpy(np.ascontiguousarray(x_norm, dtype=np.float32))
        self.c = torch.from_numpy(labels_to_index(cfg, labels))
        self.labels = torch.from_numpy(np.asarray(labels))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.c[i]
