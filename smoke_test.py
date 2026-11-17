"""小规模 Smoke Test：列选择、分块降采样一致性、切窗/划分无重叠、模型张量形状、CFG 采样、Checkpoint。
用法：D:\\Anaconda\\envs\\DL3.91\\python.exe smoke_test.py
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly

from vibddpm.data import downsample_file, make_windows, read_header, split_bounds, Normalizer
from vibddpm.model import build_model
from vibddpm.utils import ROOT, load_config, labels_to_index, seed_everything

cfg = load_config()
d = cfg['data']
seed_everything(0)
print('标签 / 内部索引:', d['label_to_index'], d['label_desc'])

# 1. 列选择：按列名读取的结果应与按“物理列位置”手工解析的 电压_0..电压_8 一致，且排除 Time/电压_9/电压_10
for label in d['labels']:
    p = ROOT / d['raw_dir'] / f'{label}.csv'
    info, cols = read_header(p, d['encoding'], d['header_lines'])
    print(f'[1] {p.name} 元信息 {info}，列 {cols}')
    df = pd.read_csv(p, skiprows=d['header_lines'], header=0, usecols=d['channels'], encoding=d['encoding'], nrows=5)
    with open(p, 'rb') as f:
        lines = f.read(4000).decode(d['encoding']).splitlines()[4:9]
    manual = np.array([[float(v) for v in ln.split(',')[1:10]] for ln in lines])
    pos = [cols.index(c) for c in d['channels']]
    assert pos == list(range(1, 10)), pos
    assert np.allclose(df[d['channels']].to_numpy(), manual), '列选择错误'
    assert list(df.columns) == d['channels']
print('[1] 列选择 OK：只取 电压_0..电压_8（物理列 1..9），排除 Time、电压_9、电压_10')

# 2. 分块降采样 vs 整段 resample_poly
p = ROOT / d['raw_dir'] / f'{d["labels"][0]}.csv'
N = 300_000
full = pd.read_csv(p, skiprows=d['header_lines'], header=0, usecols=d['channels'], encoding=d['encoding'],
                   nrows=N)[d['channels']].to_numpy(np.float64)
ref = resample_poly(full, 1, 10, axis=0)
y, n_raw = downsample_file(p, d, max_rows=N, chunk_rows=37_000 // 10 * 10)   # 刻意用不整齐的小块
assert n_raw == N and y.shape == ref.shape, (y.shape, ref.shape)
err = np.abs(y - ref).max()
print(f'[2] 分块降采样 {N} -> {len(y)} 点，与整段处理最大误差 {err:.2e}（信号幅值 ~{np.abs(ref).max():.2f}）')
assert err < 1e-9
# 抗混叠检查：与简单抽取的差异（应存在且显著）
naive = full[::10]
print(f'[2] 与简单每10点抽1点的 RMS 差 {np.sqrt(((naive - ref) ** 2).mean()):.4f}（非零说明有抗混叠滤波）')

# 3. 划分与切窗无重叠
seg = y
b = split_bounds(len(seg), d['split_ratio'], d['split_gap'])
w = {k: make_windows(seg[s:e], d['window_size'], d['stride']) for k, (s, e) in b.items()}
spans = {k: (s + ww[1].min(), s + ww[1].max() + d['window_size']) for k, ((s, e), ww) in zip(b, zip(b.values(), w.values())) if len(ww[1])}
print('[3] 段界', b, '窗口覆盖', spans, {k: v[0].shape for k, v in w.items()})
ks = list(spans)
for a, bb in zip(ks[:-1], ks[1:]):
    assert spans[a][1] <= spans[bb][0], '跨集合窗口重叠'
    print(f'[3] {a}->{bb} 隔离 {spans[bb][0] - spans[a][1]} 点')

# 4. 归一化往返
stats = dict(mean=seg.mean(0).tolist(), std=seg.std(0).tolist(), version='smoke', channels=d['channels'])
nz = Normalizer(stats)
xw = w['train'][0][:4].astype(np.float32)
assert np.allclose(nz.inverse(nz(xw)), xw, atol=1e-5)
print('[4] 归一化/反归一化往返 OK')

# 5. 模型形状、损失、CFG 采样
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg['model']['dim'] = 16                    # 缩小模型做形状测试
cfg['diffusion']['timesteps'] = 50
cfg['diffusion']['sampling_timesteps'] = 50
diff = build_model(cfg).to(device)
x = torch.from_numpy(nz(xw)).to(device)
c = torch.from_numpy(labels_to_index(cfg, [d['labels'][0], d['labels'][1]] * 2)).to(device)
assert x.shape == (4, 9, 1024)
loss, per = diff(x, c)
loss.backward()
print(f'[5] 输入 {tuple(x.shape)}, 条件 {c.tolist()}, loss {loss.item():.4f}, per-sample {tuple(per.shape)}')
assert diff.model.class_emb.weight.grad is not None and diff.model.class_emb.weight.grad.abs().sum() > 0
# 条件 dropout：训练模式下 p=1 时必须全部使用 null embedding
diff.model.train()
t = torch.zeros(4, dtype=torch.long, device=device)
e_drop = diff.model.cond_embedding(t, c, 1.0)
e_null = diff.model.cond_embedding(t, torch.full_like(c, diff.model.null_index), 0.)
assert torch.allclose(e_drop, e_null)
# 条件确实改变输出
diff.eval()
with torch.no_grad():
    o0 = diff.model(x, t + 10, torch.zeros_like(c), cond_drop_prob=0.)
    o1 = diff.model(x, t + 10, torch.ones_like(c), cond_drop_prob=0.)
    ow = diff.model.forward_with_cfg(x, t + 10, c, 2.0)
print(f'[5] 不同条件输出差 {(o0 - o1).abs().mean().item():.2e}, CFG 输出 {tuple(ow.shape)}')
for w_ in (1.0, 2.0, 3.0):
    s = diff.sample(c, guidance_scale=w_, progress=False)
    s10 = diff.sample(c, guidance_scale=w_, sampling_timesteps=10, progress=False)
    assert s.shape == (4, 9, 1024) and s10.shape == (4, 9, 1024) and torch.isfinite(s).all()
print('[5] DDPM / DDIM 采样形状 [N,9,1024] OK（w=1,2,3）')

# 6. Checkpoint 保存 / 恢复一致
with tempfile.TemporaryDirectory() as td:
    torch.save(diff.state_dict(), Path(td) / 'a.pt')
    diff2 = build_model(cfg).to(device)
    diff2.load_state_dict(torch.load(Path(td) / 'a.pt'))
    with torch.no_grad():
        assert torch.allclose(diff2.model(x, t + 5, c, 0.), diff.model(x, t + 5, c, 0.))
print('[6] Checkpoint 往返 OK')
print('SMOKE TEST PASSED')
