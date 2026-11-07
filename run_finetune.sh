#!/bin/bash

# Activate conda environment
source /workspace/anaconda3/etc/profile.d/conda.sh
conda activate llama_factory

# 创建输出目录
mkdir -p /workspace/CARE/output

# 设置日志文件的绝对路径
LOG_FILE="/workspace/CARE/training.log"

echo "开始训练，日志将保存到: $LOG_FILE"

# 检查数据集目录
echo "检查数据集目录..."
ls -la /workspace/CARE/result/llama_factory_dataset/alpaca

# 检查模型目录
echo "检查模型目录..."
ls -la /workspace/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | head -n 10

# 处理命令行参数
if [ "$1" == "train" ]; then
    # Run the fine-tuning script using Llama-factory CLI with correct parameters
    cd /workspace/LLaMA-Factory
    echo "当前工作目录: $(pwd)"

    # 终止之前的训练进程（如果有）
    pkill -f "python src/train.py"
    echo "已终止之前的训练进程"

    nohup python src/train.py \
        --model_name_or_path /workspace/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --dataset alpaca \
        --dataset_dir /workspace/CARE/result/llama_factory_dataset \
        --output_dir /workspace/CARE/output \
        --num_train_epochs 3 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --learning_rate 2e-5 \
        --cutoff_len 2048 \
        --logging_steps 10 \
        --save_steps 100 \
        --save_total_limit 3 \
        --lora_rank 8 \
        --lora_alpha 32 \
        --lora_dropout 0.1 \
        --fp16 \
        --do_train \
        --trust_remote_code \
        --finetuning_type lora \
        --template default > "$LOG_FILE" 2>&1 &

    # 获取后台进程PID
    PID=$!
    echo "训练已在后台启动，进程ID: $PID"
    echo "日志文件保存在: $LOG_FILE"
    echo "可以使用 'tail -f $LOG_FILE' 查看训练进度"

elif [ "$1" == "rag" ]; then
    # 运行RAG整段功能
    echo "运行RAG整段功能..."
    cd /workspace/CARE
    
    # 执行RAG整段脚本
    python codes/rag_llama_factory.py
    
    echo "RAG整段功能执行完成"

elif [ "$1" == "dev" ]; then
    # 对dev.json数据进行诊断
    echo "开始对dev.json数据进行RAG整段诊断..."
    cd /workspace/CARE
    
    # 执行RAG整段诊断脚本
    python codes/rag_llama_factory.py
    
    # 查看结果文件大小
    RESULT_FILE="/workspace/CARE/result/rag_diagnosis/rag_dev_results.json"
    if [ -f "$RESULT_FILE" ]; then
        FILE_SIZE=$(du -h "$RESULT_FILE" | cut -f1)
        echo "诊断结果已保存到 $RESULT_FILE (文件大小: $FILE_SIZE)"
    else
        echo "警告: 结果文件未找到"
    fi
    
    echo "dev.json数据诊断完成"

else
    echo "用法: $0 [train|rag|dev]"
    echo "  train - 执行模型微调"
    echo "  rag   - 执行RAG整段功能测试"
    echo "  dev   - 使用RAG对dev.json数据进行诊断"
fi 