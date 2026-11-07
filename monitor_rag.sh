#!/bin/bash

# 激活conda环境并直接运行脚本
# 这个脚本会在前台运行，所以您可以直接看到输出

echo "开始运行RAG整段诊断..."

# 激活conda环境并运行脚本
conda activate llama_factory
python /workspace/CARE/codes/rag_llama_factory.py

echo "RAG整段诊断完成" 