# CLAUDE.md

## Python 环境

- 本项目统一使用 conda 环境 **DL3.91**：`D:\Anaconda\envs\DL3.91\python.exe`
- **不要使用 `DL` 环境**，也不要使用 base 或系统的 `python`。
- 缺少的库一律安装到 DL3.91：`D:\Anaconda\envs\DL3.91\python.exe -m pip install --no-user <pkg>`
- 运行 Python 时直接写完整路径 `D:\Anaconda\envs\DL3.91\python.exe`，不要用 `$py` 之类的变量代替。
- denoising-diffusion-pytorch 固定为 **2.1.1**。2.3.x 依赖 `torch.nn.attention`，而这个模块需要 torch>=2.3，与环境里的 torch 2.0 不兼容。
- DL3.91 自带 torch 2.0.0+cu118（CUDA 可用）。安装其他库时不要升级或替换 torch（必要时用 `--no-deps`）。
- 安装前确认环境目录可写。若不可写，pip 会悄悄装到 `%APPDATA%\Python\Python39\site-packages`（用户目录）。这个目录被所有 Python 3.9 环境共享，而且优先级更高，会影响 DL 和 base 环境。禁止装到这里；必要时加 `--no-user`，让 pip 直接报错。

## 数据

- `00DATA/19.csv`（全流量，标签 0）、`00DATA/20.csv`（零流量，标签 1）。只读，不得修改原始 CSV。
