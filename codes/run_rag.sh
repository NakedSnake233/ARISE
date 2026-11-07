#!/bin/bash

# 确保在正确的目录下运行
cd /workspace/CARE
echo "当前工作目录: $(pwd)"

# 激活conda虚拟环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cluster
echo "已激活conda环境: $(conda info --envs | grep '*')"

# 设置PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/workspace/CARE
echo "PYTHONPATH: $PYTHONPATH"

# 检查alpaca数据集路径
echo "检查alpaca数据集路径..."
if [ -d "/workspace/CARE/result/llama_factory_dataset/alpaca" ]; then
    echo "发现alpaca数据集"
    ls -la /workspace/CARE/result/llama_factory_dataset/alpaca
else
    echo "警告: alpaca数据集路径不存在！"
    mkdir -p /workspace/CARE/result/llama_factory_dataset/alpaca
fi

# 创建输出目录
mkdir -p /workspace/CARE/result/rag_diagnosis
echo "创建输出目录: /workspace/CARE/result/rag_diagnosis"

# 运行RAG故障诊断系统
echo "开始运行RAG故障诊断系统..."
python -u codes/rag_fault_diagnosis.py 

echo "RAG故障诊断系统已完成运行，结果保存在 /workspace/CARE/result/rag_diagnosis 目录下" 