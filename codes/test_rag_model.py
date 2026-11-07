#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codes.preprocess.util import Info
from neo4j import GraphDatabase

# NEO4J 连接配置
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = __import__('os').environ['NEO4J_PASSWORD']

# 模型路径
BASE_MODEL_PATH = "/workspace/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # 基础模型路径

class BasicRAGTest:
    def __init__(self):
        # Connect to Neo4j database
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        # 直接加载基础模型，不使用LoRA
        print("正在加载基础模型...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_PATH,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            print("基础模型加载成功！")
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.tokenizer = None
            self.model = None
            print("将使用纯计算方法进行诊断修正")
        
        # Load service information from util.py
        info = Info(bench='TrainTicket')
        self.service2index = info.service2nid
        self.index2service = {v: k for k, v in self.service2index.items()}
        self.api_list = info.service_names
    
    def close(self):
        """Close Neo4j connection"""
        self.driver.close()

    def get_service_calls_and_faults(self):
        """Get service call relationships and fault propagation information from Neo4j"""
        with self.driver.session() as session:
            # Get service call relationships
            calls_result = session.run(
                """
                MATCH (s1:Service)-[r:CALLS]->(s2:Service)
                RETURN s1.name AS caller, s2.name AS callee
                """
            )
            
            # Get fault propagation relationships - changed to use AFFECTS relationship
            fault_result = session.run(
                """
                MATCH (f:Fault)-[r:AFFECTS]->(s:Service)
                RETURN f.name AS fault_service, f.type AS fault_type, s.name AS affected_service
                """
            )
            
            # Build results
            calls = [f"{record['caller']} calls {record['callee']}" for record in calls_result]
            faults = []
            for record in fault_result:
                faults.append(f"Fault service {record['fault_service']} ({record['fault_type']}) affects {record['affected_service']}")
            
            return {
                "calls": calls,
                "faults": faults
            }
    
    def format_prompt(self, input_text, graph_info):
        """
        Format the prompt for RAG整段 approach
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
    
    def generate_model_response(self, prompt):
        """
        Generate response using the model
        """
        if self.model is None or self.tokenizer is None:
            print("模型未加载，无法生成响应")
            return None
            
        # 设置生成参数
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # 生成响应
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码响应
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            print(f"生成响应时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    def parse_diagnosis_result(self, output_text):
        """简单解析诊断结果"""
        try:
            # 解析每一行
            lines = output_text.split('\n')
            for line in lines:
                # 提取"Node X fault probability: Y"的模式
                if "Node" in line and "fault probability" in line:
                    print(f"发现故障节点信息: {line}")
                    return line
            
            # 如果没有特定格式，返回整个输出
            return output_text
        except Exception as e:
            print(f"解析诊断结果时出错: {e}")
            return output_text

def main():
    """测试RAG整段功能 - 使用基础模型版本"""
    print("启动RAG整段功能测试 (基础模型版)")
    
    # 初始化测试系统
    print("初始化测试系统...")
    rag_test = BasicRAGTest()
    
    try:
        # 获取图数据库信息
        print("获取图数据库信息...")
        graph_info = rag_test.get_service_calls_and_faults()
        print(f"获取到 {len(graph_info['calls'])} 条调用关系和 {len(graph_info['faults'])} 条故障传播关系")
        
        # 创建一个测试输入
        test_input = """
故障现象: 用户无法正常查询列车详细时刻表信息，系统显示"服务暂时不可用，请稍后再试"的错误。
系统日志显示: 服务间调用超时，票务服务尝试调用列车信息服务的detailedTimeTable API时超时。
观察到的异常: 列车信息服务中的getRouteByTripId函数执行时间过长，且CPU占用率过高。
已检查的部分: 数据库连接正常，网络延迟在正常范围内。其他依赖列车信息服务的API调用也出现类似超时。
分析并找出可能的故障节点(服务): 
"""
        
        # 构建测试提示
        prompt = rag_test.format_prompt(test_input, graph_info)
        print("\n测试提示构建完成:")
        print("="*50)
        print(prompt)
        print("="*50)
        
        # 使用基础模型生成响应
        if rag_test.model is not None and rag_test.tokenizer is not None:
            print("\n使用基础模型生成响应...")
            response = rag_test.generate_model_response(prompt)
            
            print("\n模型响应:")
            print("-"*50)
            print(response)
            print("-"*50)
            
            # 解析响应
            print("\n解析响应...")
            parsed_result = rag_test.parse_diagnosis_result(response)
            print(f"解析结果: {parsed_result}")
        else:
            print("模型未加载，无法生成响应")
    
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 关闭资源
        rag_test.close()
        print("测试完成")

if __name__ == "__main__":
    main() 