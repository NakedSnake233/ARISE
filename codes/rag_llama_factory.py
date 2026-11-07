#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import torch
import sys
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from neo4j import GraphDatabase

# 添加当前项目目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codes.preprocess.util import Info

# NEO4J 连接配置
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = __import__('os').environ['NEO4J_PASSWORD']

# 模型路径
BASE_MODEL_PATH = "/workspace/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # 基础模型路径
LORA_MODEL_PATH = "/workspace/CARE/output"  # LoRA微调模型路径

def get_service_calls_and_faults():
    """获取服务调用关系和故障传播信息"""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            # 获取服务调用关系
            calls_result = session.run(
                """
                MATCH (s1:Service)-[r:CALLS]->(s2:Service)
                RETURN s1.name AS caller, s2.name AS callee
                """
            )
            
            # 获取故障传播关系
            fault_result = session.run(
                """
                MATCH (f:Fault)-[r:AFFECTS]->(s:Service)
                RETURN f.name AS fault_service, f.type AS fault_type, s.name AS affected_service
                """
            )
            
            # 构建结果
            calls = [f"{record['caller']} calls {record['callee']}" for record in calls_result]
            faults = []
            for record in fault_result:
                faults.append(f"Fault service {record['fault_service']} ({record['fault_type']}) affects {record['affected_service']}")
            
            return {
                "calls": calls,
                "faults": faults
            }
    finally:
        driver.close()

def format_prompt(input_text, graph_info):
    """
    格式化提示词
    """
    prompt = f"""## Part 1: User Input
{input_text}

## Part 2: Service Call and Fault Propagation Information
### Service Call Relationships:
{chr(10).join(graph_info['calls'][:50])}

### Fault Propagation:
{chr(10).join(graph_info['faults'][:30])}

## Task: Based on the above information, identify the fault node with highest probability. You are looking for a node with a fault that is causing issues in the system. Output should include the node number and fault probability.
"""
    return prompt

def load_model():
    """加载微调后的模型"""
    print("正在加载LoRA微调模型...")
    try:
        # 打印PEFT版本
        import peft
        print(f"PEFT版本: {peft.__version__}")
        
        # 加载配置
        config = PeftConfig.from_pretrained(LORA_MODEL_PATH)
        print(f"PEFT模型配置: 类型={config.peft_type}, 基础模型={config.base_model_name_or_path}")
        
        # 加载分词器和模型
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
        print("Base tokenizer loaded")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("Base model loaded")
        
        # 加载PEFT模型
        model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
        print("微调模型加载成功！")
        
        return tokenizer, model
    except Exception as e:
        print(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def generate_response(tokenizer, model, prompt):
    """生成模型响应"""
    if model is None or tokenizer is None:
        print("模型未加载，无法生成响应")
        return None
    
    try:
        print("编码输入文本...")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        print(f"输入编码完成，输入长度: {inputs['input_ids'].shape[1]}")
        
        print("开始生成响应...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=128,  # 减少token生成数量，加快速度
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
        print("生成完成")
        
        print("解码响应...")
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print("解码完成")
        return response.strip()
    
    except Exception as e:
        print(f"生成响应时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_alpaca_data(filepath):
    """加载Alpaca格式数据"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功加载数据文件: {filepath}，包含{len(data)}条数据")
        return data
    except Exception as e:
        print(f"加载数据文件失败: {e}")
        return []

def parse_diagnosis_result(output_text):
    """
    解析诊断结果
    提取节点故障概率和故障节点
    """
    try:
        if "fault probability" in output_text:
            # 尝试提取所有节点的故障概率
            fault_probs = {}
            fault_node = None
            max_prob = 0.0
            
            # 解析每一行
            lines = output_text.split('\n')
            for line in lines:
                # 提取"Node X fault probability: Y"的模式
                if "Node" in line and "fault probability" in line:
                    try:
                        # 使用字符串分割提取节点和概率
                        parts = line.split("Node ")[1].split(" fault probability:")
                        if len(parts) == 2:
                            node_id = int(parts[0].strip())
                            prob = float(parts[1].strip())
                            fault_probs[node_id] = prob
                            
                            # 更新最大概率节点
                            if prob > max_prob:
                                max_prob = prob
                                fault_node = node_id
                    except Exception as e:
                        print(f"Error parsing line '{line}': {e}")
            
            # 尝试从总结语句提取故障节点
            if not fault_node:
                for line in lines:
                    if "Diagnosis result:" in line and "is the fault node" in line:
                        try:
                            # 提取"Node X is the fault node with fault probability Y"
                            node_str = line.split("Node ")[1].split(" is the fault node")[0].strip()
                            fault_node = int(node_str)
                            # 也尝试提取概率
                            if "probability" in line:
                                prob_str = line.split("probability ")[1].strip().rstrip('.')
                                fault_probs[fault_node] = float(prob_str)
                        except Exception as e:
                            print(f"Error parsing summary '{line}': {e}")
            
            # 如果找到故障节点，返回节点ID和概率
            if fault_node is not None:
                return {
                    "fault_node": fault_node,
                    "fault_probability": fault_probs.get(fault_node, 0.0),
                    "all_probabilities": fault_probs
                }
            elif fault_probs:
                # 如果有概率但没有明确标识故障节点，返回概率最高的节点
                max_node = max(fault_probs.items(), key=lambda x: x[1])
                return {
                    "fault_node": max_node[0],
                    "fault_probability": max_node[1],
                    "all_probabilities": fault_probs
                }
        
        # 如果没有特定格式，返回原始输出
        return {"raw_output": output_text}
    
    except Exception as e:
        print(f"解析诊断结果时出错: {e}")
        return {"error": str(e), "raw_output": output_text}

def main():
    """主函数"""
    print("=== RAG整段诊断 (LLaMA-Factory环境) ===")
    
    # 获取图数据库信息
    print("\n获取图数据库信息...")
    graph_info = get_service_calls_and_faults()
    print(f"获取到 {len(graph_info['calls'])} 条调用关系和 {len(graph_info['faults'])} 条故障传播关系")
    
    # 加载模型
    tokenizer, model = load_model()
    
    # 加载dev.json数据
    dev_file = "/workspace/CARE/result/llama_factory_dataset/alpaca/dev.json"
    dev_data = load_alpaca_data(dev_file)
    
    if not dev_data:
        print("没有加载到有效数据，退出程序")
        return
    
    # 准备输出目录和文件
    output_dir = "/workspace/CARE/result/rag_diagnosis"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "rag_dev_results.json")
    
    results = []
    
    # 处理每个样本
    print(f"\n开始处理数据，共{len(dev_data)}条...")
    
    # 限制处理数量，避免处理时间过长
    max_samples = min(len(dev_data), 10)  # 最多处理10条
    for idx, item in enumerate(dev_data[:max_samples]):
        print(f"\n正在处理第 {idx+1}/{max_samples} 条数据...")
        
        input_text = item.get('input', '')
        instruction = item.get('instruction', '')
        original_output = item.get('output', '')
        
        # 构建提示词
        prompt = format_prompt(input_text, graph_info)
        
        # 生成响应
        if tokenizer is not None and model is not None:
            model_response = generate_response(tokenizer, model, prompt)
            
            if model_response:
                print(f"生成响应成功，长度：{len(model_response)}")
                
                # 解析诊断结果
                parsed_result = parse_diagnosis_result(model_response)
                print(f"解析结果: {json.dumps(parsed_result, ensure_ascii=False)}")
                
                # 添加到结果列表
                result = {
                    "index": idx,
                    "input": input_text,
                    "instruction": instruction,
                    "original_output": original_output,
                    "rag_response": model_response,
                    "parsed_result": parsed_result
                }
                results.append(result)
            else:
                print("生成响应失败")
                results.append({
                    "index": idx,
                    "input": input_text,
                    "error": "生成响应失败",
                    "original_output": original_output
                })
        else:
            print("模型未加载，无法生成响应")
            break
    
    # 保存结果
    print(f"\n处理完成，正在保存结果到 {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存，共处理 {len(results)} 条数据")
    
    print("\n诊断完成")

if __name__ == "__main__":
    main() 