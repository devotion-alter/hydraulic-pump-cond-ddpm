# 液压泵双工况条件 1D DDPM 振动信号生成

基于 [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)（v2.1.1）的 `Unet1D` / `GaussianDiffusion1D`，实现类别条件 + Classifier-Free Guidance 的九通道振动信号生成。

## 环境

```powershell
D:\Anaconda\envs\DL3.91\python.exe   # torch 2.0.0+cu118，详见 CLAUDE.md
```

## 工况标签

每个 CSV 对应一种运行工况（不是故障类别），**文件名编号即条件标签**：

| 文件 | 标签 | 工况 |
|---|---|---|
| `00DATA/19.csv` | 19 | 全流量 |
| `00DATA/20.csv` | 20 | 零流量 |

配置 `data.labels: auto` 会自动扫描 `00DATA` 下文件名为纯数字的 CSV。模型内部的 Embedding 索引按标签升序映射（19→0，20→1），对外一律使用文件编号。工况描述在 `data.label_desc` 中修改。

## 流程

```powershell
D:\Anaconda\envs\DL3.91\python.exe smoke_test.py      # 列选择/分块降采样/划分/形状/CFG 采样/Checkpoint 自检
D:\Anaconda\envs\DL3.91\python.exe prepare_data.py    # CSV -> 10 kHz 缓存 -> 50 s 截取 -> 划分 -> 归一化参数
D:\Anaconda\envs\DL3.91\python.exe train.py           # 训练（--resume auto 从 last.pt 恢复）
D:\Anaconda\envs\DL3.91\python.exe sample.py          # 按工况 × guidance_scale 生成，反归一化后保存
D:\Anaconda\envs\DL3.91\python.exe evaluate.py        # 时域/FFT/PSD/统计量/相关矩阵/CFG 对比/条件有效性
```

输出目录：`outputs/<run_name>/`，包括 `checkpoints/`、`train_log.csv`、`samples/`、`figures/` 和 `eval_report.json`。

## 关键设计

- **读取**：按**列名**选取 `电压_0`～`电压_8`（物理第 2～10 列），排除 `Time`、`电压_9`、`电压_10`。表头为 GBK 编码。分块读取，每块 100 万行。
- **降采样**：`resample_poly(up=1, down=10)`（Kaiser FIR 抗混叠）。分块处理时，每块左右各保留 2000 个原始点作为上下文，结果与整段一次性处理逐点一致（smoke test 中最大误差为 0）。
- **截取与划分**：先去掉降采样结果开头 0.05 s 的滤波边缘段，再截取之后连续的 **50 s**（500,000 点）。在其中按时间顺序划分 Train 70% / Val 15% / Test 15%，段间留 2048 点隔离，**然后再切窗**（1024/512），保证跨集合的窗口不重叠。
- **归一化**：逐通道 z-score，参数只用**两工况训练段的连续数据**拟合，两工况共用，不做逐窗口归一化，因此保留了工况间的幅值差异。参数和版本号保存在 `cache/norm_stats.json`。
- **条件 Unet1D**（`vibddpm/model.py`）：复用仓库 Unet1D 的全部层，增加 `Embedding(K+1, dim)`，其中第 K 个为单独的 null embedding。类别嵌入经 MLP 后与时间嵌入相加，通过每个 ResnetBlock 的 scale/shift 注入。训练时以 `cond_drop_prob=0.1` 将条件替换为 null。
- **CFG 采样**：`eps = eps_u + w·(eps_c − eps_u)`，objective 为 `pred_noise`。仓库原本把 x0 硬裁剪到 [-1,1]，这里改为 z-score 空间下可配置的 `clip_x0`。
- **条件有效性检查**：
  - 训练中：验证损失分别按正确 / 交换 / null 标签计算。
  - 评价时：生成样本与各真实工况的 PSD 距离矩阵；在真实 Train 上拟合的工况判别器对生成样本的归类率；生成两工况间差异与真实两工况间差异的对比。

## 局限

每个工况只有一次采集（一个 CSV）。Train/Val/Test 是同一次采集中不同的时间段，评价结果只反映**同次采集内**的生成能力，**不能证明跨批次或跨设备的泛化**。
