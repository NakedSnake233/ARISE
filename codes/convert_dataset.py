#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse

def convert_to_llama_factory_format(input_file, output_dir):
    """
    将best_prompts.json转换为LLaMA-Factory兼容的格式
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 转换为LLaMA-Factory格式
    train_data = []
    
    for item in data:
        human_text = ""
        assistant_text = ""
        
        for conv in item["conversations"]:
            if conv["from"] == "human":
                human_text = conv["value"]
            elif conv["from"] == "assistant":
                assistant_text = conv["value"]
        
        # 检查数据有效性
        if not human_text or not assistant_text:
            print(f"警告：跳过无效数据: {item}")
            continue
        
        # 使用简单的格式，适用于unsloth等多个库
        row = {
            "instruction": "Analyze the service nodes and identify potential faults",
            "input": human_text.strip(),
            "output": assistant_text.strip(),
            "text": f"<s>[INST] {human_text.strip()} [/INST] {assistant_text.strip()}</s>"
        }
        if "fault_node" in item:
            row["fault_node"] = item["fault_node"]
        train_data.append(row)
    
    # 创建标准结构
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    
    # 保存为单个JSON文件
    full_output_file = os.path.join(output_dir, "train.json")
    with open(full_output_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    # 保存为JSONL文件 (每行一个JSON对象)
    jsonl_output_file = os.path.join(output_dir, "train.jsonl")
    with open(jsonl_output_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 创建LLaMA-Factory期望的结构
    for dataset_name in ["alpaca", "sharegpt", "hh", "oasst"]:
        dataset_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # 保存训练数据
        dataset_train_file = os.path.join(dataset_dir, "train.json")
        with open(dataset_train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        # 创建一个小验证集
        dev_data = train_data[:min(len(train_data), 10)]
        dataset_dev_file = os.path.join(dataset_dir, "dev.json")
        with open(dataset_dev_file, 'w', encoding='utf-8') as f:
            json.dump(dev_data, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成！共处理了 {len(train_data)} 条数据")
    print(f"数据已保存到多个格式，主文件: {full_output_file}")
    
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将best_prompts.json转换为LLaMA-Factory兼容格式")
    parser.add_argument("--input", type=str, default="../result/best_prompts.json", help="输入文件路径")
    parser.add_argument("--output", type=str, default="../result/llama_factory_dataset", help="输出目录")
    args = parser.parse_args()
    
    convert_to_llama_factory_format(args.input, args.output) 
