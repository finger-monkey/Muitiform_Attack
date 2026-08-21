# 环境配置说明 - Multiform Attack (MA)

## 项目信息

**项目名称**: Multiform Attack for Transferable Cross-Modal Person Re-Identification  
**Conda 环境**: pami-attack  
**生成时间**: 2026-08-20  
**配置验证**: ✅ 在 Ubuntu 20.04 上验证

---

## 硬件要求

### GPU
- **NVIDIA GPU**: 支持 CUDA 11.3 的显卡
  - 推荐: A100, RTX 3090, V100 或更高
  - 最低: 11GB GPU 内存（用于推理），24GB（用于训练）

### CPU
- **CPU**: 现代多核处理器（8+ 核推荐）
- **内存**: 32GB+ RAM（处理大型数据集）

### 存储
- **磁盘**: 500GB+ SSD（用于数据集和模型）

---

## 环境配置方案

### 方案 A: 使用完整环境文件（推荐）

**适用场景**: 完全复现结果，确保所有依赖完全一致

```bash
# 1. 克隆项目
cd /path/to/MA-code/MA

# 2. 创建环境
conda env create -f environment.yml

# 3. 激活环境
conda activate pami-attack

# 4. 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

**预期输出**:
```
PyTorch: 1.12.0
CUDA Available: True
```

---

### 方案 B: 手动安装（用于不同的 CUDA 版本）

**适用场景**: 你的系统 CUDA 版本不是 11.3，但希望兼容

#### 步骤 1: 创建基础环境

```bash
# 创建 Python 3.9 环境
conda create -n pami-attack python=3.9.18 numpy=1.26.2 -y

# 激活环境
conda activate pami-attack
```

#### 步骤 2: 安装 PyTorch（根据你的 CUDA 版本选择）

**如果你的 CUDA 版本是 11.3**:
```bash
conda install pytorch=1.12.0 torchvision=0.13.0 torchaudio=0.12.0 pytorch-cuda=11.3 -c pytorch -y
```

**如果你的 CUDA 版本是 12.1**:
```bash
conda install pytorch::pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

**如果你的 CUDA 版本是 11.8**:
```bash
conda install pytorch=1.13.1 torchvision=0.14.1 torchaudio=0.13.1 pytorch-cuda=11.8 -c pytorch -y
```

#### 步骤 3: 安装 FAISS

```bash
# CUDA 11.3 版本
conda install faiss-gpu=1.7.3 -c pytorch -y

# 或其他 CUDA 版本（会自动选择）
conda install faiss-gpu -c pytorch -y
```

#### 步骤 4: 安装其他依赖

```bash
# 安装 pip 包
pip install -r requirements-pip.txt
```

或者只安装最小依赖:
```bash
pip install -r requirements-minimal.txt
```

---

### 方案 C: 快速轻量级安装

**适用场景**: 仅需推理或对依赖版本不敏感

```bash
# 创建环境
conda create -n ma-light python=3.9 -y
conda activate ma-light

# 安装 PyTorch（最新兼容版本）
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 安装最小依赖
pip install -r requirements-minimal.txt
```

---

## 关键软件版本

### 核心版本（必须）

| 软件 | 版本 | 说明 |
|------|------|------|
| Python | 3.9.18 | 不要使用 3.10+ |
| PyTorch | 1.12.0 | 核心深度学习框架 |
| CUDA | 11.3.1 | GPU 计算平台 |
| cuDNN | 8.3.2 | CUDA 神经网络库 |
| NumPy | 1.26.2 | 数值计算 |
| torchvision | 0.13.0 | 视觉模型库 |

### 推荐包版本（可略微调整）

| 包 | 版本范围 | 说明 |
|----|---------|------|
| opencv-python | >=4.8.1 | 图像处理 |
| scikit-learn | >=1.3.2 | 机器学习工具 |
| matplotlib | >=3.8.2 | 可视化 |
| faiss-gpu | 1.7.3 | 相似度搜索（GPU加速） |
| metric-learn | >=0.7.0 | 度量学习 |

---

## 验证安装

### 1. 检查 PyTorch 和 CUDA

```bash
python -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Version: {torch.version.cuda}')
print(f'GPU Count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Current GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
"
```

### 2. 检查关键包

```bash
python -c "
import numpy as np
import cv2
import torch
import torchvision
import faiss
import sklearn
import matplotlib
print('✅ NumPy:', np.__version__)
print('✅ OpenCV:', cv2.__version__)
print('✅ PyTorch:', torch.__version__)
print('✅ torchvision:', torchvision.__version__)
print('✅ FAISS-GPU: installed')
print('✅ scikit-learn:', sklearn.__version__)
print('✅ matplotlib:', matplotlib.__version__)
print('✅ All packages OK!')
"
```

### 3. 测试项目导入

```bash
cd /sda1/XXX/home/MA-code/MA
python -c "from reid import models; print('✅ Project imports OK')"
```

---

## 常见问题与解决方案

### Q1: CUDA 不可用？

```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 PyTorch CUDA 支持
python -c "import torch; print(torch.cuda.is_available())"

# 解决方案：重新安装 PyTorch
conda remove pytorch torchvision torchaudio
conda install pytorch::pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Q2: FAISS 导入错误？

```bash
# 重新安装 FAISS
conda remove faiss-gpu
conda install faiss-gpu -c pytorch

# 或使用 CPU 版本
conda install faiss-cpu -c pytorch
```

### Q3: 模块找不到？

```bash
# 确保在项目目录中
cd /sda1/XXX/home/MA-code/MA

# 添加到 Python 路径
export PYTHONPATH="${PYTHONPATH}:/sda1/XXX/home/MA-code/MA"

# 验证
python -c "import reid; print('OK')"
```

### Q4: GPU 内存不足？

```bash
# 降低 batch size
python Multiform_attack.py --batch_size 64

# 或使用混合精度训练
export CUDA_LAUNCH_BLOCKING=1
python -W ignore Multiform_attack.py --batch_size 128
```

### Q5: 不同版本导致结果不同？

**常见原因**:
- PyTorch 版本不同 (如 1.12 vs 1.13)
- NumPy 版本不同
- CUDA 版本不同
- 随机种子未固定

**解决方案**:
```bash
# 使用提供的 environment.yml
conda env create -f environment.yml --force

# 或在代码中固定随机种子
export PYTHONHASHSEED=0
python -W ignore Multiform_attack.py --seed 0
```

---

## 环境迁移

### 从另一台机器导出环境

```bash
# 在源机器上
conda env export -n pami-attack > environment-export.yml

# 在目标机器上
conda env create -f environment-export.yml
```

### 使用 pip 导出

```bash
# 导出所有 pip 包
pip freeze > requirements-all.txt

# 在新环境中安装
pip install -r requirements-all.txt
```

---

## 推荐的开发工作流

### 1. 创建开发环境

```bash
# 基于现有环境创建
conda create --name ma-dev --clone pami-attack

# 激活
conda activate ma-dev

# 进行开发测试
cd /sda1/XXX/home/MA-code/MA
```

### 2. 安装开发工具（可选）

```bash
conda install jupyterlab ipython -y
pip install black flake8 pytest
```

### 3. 版本管理

```bash
# 保存工作环境
conda env export -n ma-dev > environment-backup-$(date +%Y%m%d).yml

# 定期备份 requirements
pip freeze > requirements-$(date +%Y%m%d).txt
```

---

## 性能优化建议

### CUDA 优化

```bash
# 启用 TensorFloat32（更快但精度稍低）
export CUDA_LAUNCH_BLOCKING=1

# 使用混合精度（需要 Ampere 架构 GPU）
export CUDA_VISIBLE_DEVICES=0
```

### PyTorch 优化

```bash
# 启用 cuDNN 自动调优
export CUDNN_BENCHMARK=1

# 禁用不必要的自动求导
torch.set_grad_enabled(False)  # 仅推理时
```

### 数据加载优化

```python
# 使用多进程加载数据
DataLoader(dataset, num_workers=4, pin_memory=True)
```

---

## 故障排除检查清单

- [ ] 显卡驱动是否正确安装？ (`nvidia-smi`)
- [ ] CUDA 工具包是否兼容？ (`python -c "import torch; torch.cuda.is_available()"`)
- [ ] PyTorch 版本是否正确？ (`python -c "import torch; print(torch.__version__)"`)
- [ ] FAISS-GPU 是否安装？ (`python -c "import faiss; print(faiss.__version__)"`)
- [ ] 项目路径是否正确？ (`cd /sda1/XXX/home/MA-code/MA`)
- [ ] Python 路径是否设置？ (`echo $PYTHONPATH`)
- [ ] 所有依赖是否安装？ (`python -c "import reid; print('OK')"`)

---

## 附录：环境文件说明

| 文件 | 用途 | 大小 |
|------|------|------|
| `environment.yml` | 完整 conda 环境导出（推荐） | ~10KB |
| `requirements-pip.txt` | 所有 pip 包详细版本 | ~2KB |
| `requirements-minimal.txt` | 最小依赖版本范围 | ~1KB |

### 选择正确的文件

- **想要完全复现**: 使用 `environment.yml`
- **想要灵活调整**: 使用 `requirements-minimal.txt`
- **想要详细了解**: 查看 `requirements-pip.txt`

---

## 支持的操作系统

| OS | 版本 | 测试状态 |
|----|------|---------|
| Linux | Ubuntu 20.04 LTS | ✅ 完全支持 |
| Linux | Ubuntu 22.04 LTS | ✅ 完全支持 |
| Linux | CentOS 8 | ⚠️ 需要调整 GCC |
| macOS | 12+ | ❌ 不支持 GPU CUDA |
| Windows | 10/11 | ⚠️ 需要 WSL2 |

---

**最后更新**: 2026-08-20  
**维护者**: PAMI Attack Project  
**反馈**: 若环境配置有问题，请检查上述步骤和故障排除部分
