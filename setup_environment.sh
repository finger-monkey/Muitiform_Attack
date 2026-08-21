#!/bin/bash




set -e


RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Multiform Attack - 环境安装脚本                             ║${NC}"
echo -e "${BLUE}║   Environment Setup Script                                    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""


if ! command -v conda &> /dev/null; then
    echo -e "${RED}❌ Conda 未找到。请先安装 Anaconda 或 Miniconda。${NC}"
    echo "   Download from: https://www.anaconda.com/download"
    exit 1
fi

echo -e "${GREEN}✅ Conda 已安装${NC}"
echo ""


echo -e "${YELLOW}选择安装方案:${NC}"
echo "  1) 方案 A - 完整环境（推荐） [完全复现结果]"
echo "  2) 方案 B - 手动安装 [自定义 CUDA 版本]"
echo "  3) 方案 C - 轻量级安装 [仅推理]"
echo ""
read -p "请选择 (1-3): " CHOICE

case $CHOICE in
    1)
        echo -e "${BLUE}[1/3] 删除旧环境...${NC}"
        conda remove -n pami-attack --all -y 2>/dev/null || true

        echo -e "${BLUE}[2/3] 创建新环境...${NC}"
        conda env create -f environment.yml -y

        echo -e "${BLUE}[3/3] 验证安装...${NC}"
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate pami-attack

        python -c "
import torch
import torchvision
import numpy as np
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA: {torch.version.cuda}')
print(f'✅ NumPy: {np.__version__}')
print(f'✅ torchvision: {torchvision.__version__}')
print(f'✅ GPU Available: {torch.cuda.is_available()}')
"

        echo -e "${GREEN}✅ 安装完成！${NC}"
        echo ""
        echo "激活环境:"
        echo "  conda activate pami-attack"
        ;;

    2)
        read -p "请输入你的 CUDA 版本 (11.3/11.8/12.1): " CUDA_VERSION

        echo -e "${BLUE}[1/4] 创建基础环境...${NC}"
        conda create -n pami-attack python=3.9.18 numpy=1.26.2 -y

        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate pami-attack

        echo -e "${BLUE}[2/4] 安装 PyTorch ($CUDA_VERSION)...${NC}"
        case $CUDA_VERSION in
            11.3)
                conda install pytorch=1.12.0 torchvision=0.13.0 torchaudio=0.12.0 pytorch-cuda=11.3 -c pytorch -y
                ;;
            11.8)
                conda install pytorch=1.13.1 torchvision=0.14.1 torchaudio=0.13.1 pytorch-cuda=11.8 -c pytorch -y
                ;;
            12.1)
                conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
                ;;
            *)
                echo -e "${RED}❌ 不支持的 CUDA 版本${NC}"
                exit 1
                ;;
        esac

        echo -e "${BLUE}[3/4] 安装 FAISS...${NC}"
        conda install faiss-gpu -c pytorch -y

        echo -e "${BLUE}[4/4] 安装其他依赖...${NC}"
        pip install -r requirements-pip.txt -q

        echo -e "${GREEN}✅ 安装完成！${NC}"
        ;;

    3)
        echo -e "${BLUE}[1/2] 创建轻量环境...${NC}"
        conda create -n ma-light python=3.9 -y

        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate ma-light

        echo -e "${BLUE}[2/2] 安装依赖...${NC}"
        conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
        pip install -r requirements-minimal.txt -q

        echo -e "${GREEN}✅ 安装完成！${NC}"
        echo "环境名: ma-light"
        ;;

    *)
        echo -e "${RED}❌ 无效选择${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}验证安装（可选）:${NC}"
echo "  python -c \"import torch; print(f'CUDA: {torch.cuda.is_available()}')\" "
echo ""
echo -e "${GREEN}下一步:${NC}"
echo "  cd /sda1/XXX/home/MA-code/MA"
echo "  python Multiform_attack.py -s sysu_v2 -m CnMix ..."
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
