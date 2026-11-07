import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from neo4j import GraphDatabase
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codes.preprocess.util import Info
from codes.rrrca_slow_path import SlowThinkingCorrector

# 尝试导入PEFT，如果不可用，设置一个标志
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
    print("PEFT库加载成功，将支持LoRA微调模型")
except ImportError:
    PEFT_AVAILABLE = False
    print("警告: 未找到PEFT库，无法加载LoRA微调模型。将仅支持完整模型加载。")
    print("提示: 可以使用 'pip install peft' 安装PEFT库")

# NEO4J 连接配置
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = __import__('os').environ['NEO4J_PASSWORD']

# 模型路径
BASE_MODEL_PATH = "/workspace/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # 基础模型路径
FINETUNED_MODEL_PATH = "/workspace/CARE/output"  # 微调模型路径

class GraphFaultDiagnosisRAG:
    def __init__(self):
        # Connect to Neo4j database
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        # 加载微调后的模型
        print("正在加载模型...")
        try:
            # 检查是否有adapter_model目录，这是LLaMA-Factory使用LoRA时的特征
            adapter_path = os.path.join(FINETUNED_MODEL_PATH, "adapter_model")
            
            # 只有当PEFT可用且adapter存在时，才使用PEFT加载方式
            if PEFT_AVAILABLE and os.path.exists(adapter_path) and os.path.isdir(adapter_path):
                print(f"找到adapter模型路径: {adapter_path}")
                print("尝试使用PEFT方式加载LLaMA-Factory微调模型...")
                # 先加载基础模型
                self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
                base_model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL_PATH,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                # 然后加载PEFT适配器
                self.model = PeftModel.from_pretrained(base_model, adapter_path)
                print("PEFT模型加载成功！")
            else:
                # 如果PEFT不可用或没有adapter_model目录，尝试直接加载完整模型
                if not PEFT_AVAILABLE and os.path.exists(adapter_path):
                    print("发现adapter模型，但PEFT库不可用，将尝试直接加载完整微调模型...")
                else:
                    print("未找到adapter模型或PEFT不可用，尝试直接加载完整微调模型...")
                
                self.tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    FINETUNED_MODEL_PATH,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                print("完整微调模型加载成功！")
            print("模型加载成功！")
        except Exception as e:
            print(f"模型加载失败: {e}")
            # 尝试加载基础模型作为备选
            try:
                print("尝试加载基础模型...")
                self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL_PATH,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                print("基础模型加载成功！")
            except Exception as base_e:
                print(f"基础模型加载失败: {base_e}")
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
    
    def get_service_degree(self, service_name):
        """Get service node degree (in-degree + out-degree)"""
        with self.driver.session() as session:
            # Get out-degree
            out_result = session.run(
                """
                MATCH (s:Service {name: $service_name})-[r]->() 
                RETURN count(r) AS out_degree
                """, 
                service_name=service_name
            )
            out_degree = out_result.single()["out_degree"] if out_result.peek() else 0
            
            # Get in-degree
            in_result = session.run(
                """
                MATCH ()-[r]->(s:Service {name: $service_name}) 
                RETURN count(r) AS in_degree
                """, 
                service_name=service_name
            )
            in_degree = in_result.single()["in_degree"] if in_result.peek() else 0
            
            return out_degree + in_degree
    
    def get_all_service_degrees(self):
        """Get degrees of all service nodes"""
        with self.driver.session() as session:
            # Get all service nodes
            services_result = session.run(
                """
                MATCH (s:Service)
                RETURN s.name AS service_name
                """
            )
            
            # Calculate degree for each service
            degrees = {}
            for record in services_result:
                service_name = record["service_name"]
                # Get out-degree
                out_result = session.run(
                    """
                    MATCH (s:Service {name: $service_name})-[r]->() 
                    RETURN count(r) AS out_degree
                    """, 
                    service_name=service_name
                )
                out_degree = out_result.single()["out_degree"] if out_result.peek() else 0
                
                # Get in-degree
                in_result = session.run(
                    """
                    MATCH ()-[r]->(s:Service {name: $service_name}) 
                    RETURN count(r) AS in_degree
                    """, 
                    service_name=service_name
                )
                in_degree = in_result.single()["in_degree"] if in_result.peek() else 0
                
                # Calculate total degree
                degrees[service_name] = out_degree + in_degree
            
            return degrees
    
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
            
            propagation_result = session.run(
                """
                MATCH (f:Fault)-[r:PROPAGATES_TO]->(s:Service)
                RETURN f.name AS fault_service, f.type AS fault_type, s.name AS affected_service
                """
            )

            service_affect_result = session.run(
                """
                MATCH (s1:Service)-[r:AFFECTS]->(s2:Service)
                RETURN s1.name AS fault_service, s2.name AS affected_service
                """
            )

            legacy_affect_result = session.run(
                """
                MATCH (f:Fault)-[r:AFFECTS]->(s:Service)
                RETURN f.name AS fault_service, s.name AS affected_service
                """
            )
            
            # Build results
            calls = []
            call_edges = []
            for record in calls_result:
                edge = (record["caller"], record["callee"])
                calls.append(f"{edge[0]} calls {edge[1]}")
                call_edges.append(edge)

            faults = []
            propagation_edges = []
            for record in propagation_result:
                edge = (record["fault_service"], record["affected_service"])
                propagation_edges.append(edge)
                faults.append(f"Fault service {record['fault_service']} ({record['fault_type']}) affects {record['affected_service']}")

            affect_edges = []
            for record in service_affect_result:
                affect_edges.append((record["fault_service"], record["affected_service"]))
            for record in legacy_affect_result:
                affect_edges.append((record["fault_service"], record["affected_service"]))
            
            return {
                "calls": calls,
                "faults": faults,
                "call_edges": call_edges,
                "propagation_edges": propagation_edges,
                "affect_edges": affect_edges,
            }
    
    def load_alpaca_data(self, file_path):
        """Load Alpaca format data"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def format_prompt(self, input_text, graph_info):
        """
        Format the prompt for RAG整段 approach using LLaMA-Factory指令格式
        input_text: Input part from Alpaca data
        graph_info: Graph data from Neo4j
        """
        content = f"""## Part 1: User Input
{input_text}

## Part 2: Service Call and Fault Propagation Information
### Service Call Relationships:
{chr(10).join(graph_info['calls'][:50])}

### Fault Propagation:
{chr(10).join(graph_info['faults'][:30])}"""

        instruction = "Based on the above information, identify the fault node with highest probability. You are looking for a node with a fault that is causing issues in the system. Output should include the node number and fault probability."
        
        # 根据LLaMA-Factory的指令格式构建提示
        prompt = f"""### Instruction:
{instruction}

### Input:
{content}

### Response:
"""
        return prompt
    
    def generate_model_response(self, prompt):
        """
        Generate response using the fine-tuned model
        """
        if self.model is None or self.tokenizer is None:
            print("模型未加载，无法生成响应")
            return None
            
        # 设置生成参数
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # 生成响应
            with torch.no_grad():
                # 对于PEFT模型，我们使用模型的generate方法
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码响应 - 仅获取生成的新令牌
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
            # 针对LLaMA-Factory微调的模型可能有特定的输出格式，需要额外处理
            # 如果输出包含常见的指令模板标记，尝试提取实际响应部分
            if "### Response:" in response:
                response = response.split("### Response:")[1].strip()
            elif "### Assistant:" in response:
                response = response.split("### Assistant:")[1].strip()
            
            return response.strip()
            
        except Exception as e:
            print(f"生成响应时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def parse_diagnosis_result(self, output_text):
        """
        Parse the diagnosis result from output text
        Extract node fault probabilities and the fault node from formatted output
        """
        try:
            # 处理可能的多余格式标记
            # LLaMA-Factory模型可能会有额外的格式标记
            if isinstance(output_text, str):
                # 清理可能的格式标记
                if "### Response:" in output_text:
                    output_text = output_text.split("### Response:")[1].strip()
                elif "### Assistant:" in output_text:
                    output_text = output_text.split("### Assistant:")[1].strip()
                    
            # Extract service information from the output
            if isinstance(output_text, str):
                # 如果输出文本包含节点故障概率格式的信息
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
                
                # 如果以上解析失败，尝试更简单的方法
                try:
                    # 尝试解析为JSON
                    data = json.loads(output_text)
                    if isinstance(data, list):
                        return data
                    else:
                        return [data]
                except json.JSONDecodeError:
                    # 如果不是JSON，尝试提取服务名称
                    services = []
                    lines = output_text.split('\n')
                    for line in lines:
                        for service in self.api_list:
                            if service in line:
                                services.append(service)
                    return list(set(services))
            elif isinstance(output_text, list):
                # 如果已经是列表，直接返回
                return output_text
            else:
                print(f"Unexpected output format: {type(output_text)}")
                return []
        except Exception as e:
            print(f"Error parsing diagnosis result: {e}")
            return []
    
    def _parse_llm_scores(self, model_response):
        parsed = self.parse_diagnosis_result(model_response) if model_response else None
        if isinstance(parsed, dict) and parsed.get("all_probabilities"):
            return parsed
        return None

    def _build_corrector(self, graph_info):
        return SlowThinkingCorrector.from_graph_info(
            self.api_list,
            graph_info,
            alpha=0.6,
            beta=0.25,
            gamma=0.15,
            distance_decay=0.6,
            max_hops=3,
            edge_alpha=0.5,
        )

    def calculate_correction(self, small_model_result, graph_info, llm_result=None):
        """Apply RRRCA slow-thinking structural and rule-based correction."""
        try:
            corrector = self._build_corrector(graph_info)

            if isinstance(small_model_result, dict) and "fault_node" in small_model_result:
                fast_scores = small_model_result.get("all_probabilities", {})
                llm_scores = llm_result.get("all_probabilities", {}) if isinstance(llm_result, dict) else None
                corrected = corrector.correct(fast_scores, llm_scores=llm_scores)
                original_node = small_model_result.get("fault_node")
                corrected["original_node"] = original_node
                corrected["original_probability"] = small_model_result.get("fault_probability", 0.0)

                observed_node = original_node
                if fast_scores:
                    observed_node = max(fast_scores.items(), key=lambda item: item[1])[0]
                corrected["propagation_path"] = corrector.propagation_path(
                    observed_node,
                    corrected["fault_node"],
                    fast_scores=fast_scores,
                )
                return corrected

            elif isinstance(small_model_result, list):
                fast_scores = {service: 1.0 for service in small_model_result}
                return corrector.correct(fast_scores)
            else:
                print(f"Unrecognized small model result format: {type(small_model_result)}")
                return small_model_result
            
        except Exception as e:
            print(f"Error in correction calculation: {e}")
            import traceback
            traceback.print_exc()
            return small_model_result

    def run_diagnosis(self, alpaca_file):
        """Run the entire fault diagnosis process with RAG整段 approach"""
        try:
            # Load Alpaca data
            data = self.load_alpaca_data(alpaca_file)
            
            # Get graph database information
            graph_info = self.get_service_calls_and_faults()
            
            results = []
            
            for item in data:
                input_text = item.get("input", "")
                output_text = item.get("output", "")
                instruction = item.get("instruction", "")
                
                # Parse small model diagnostic results
                small_model_result = self.parse_diagnosis_result(output_text)
                
                if not small_model_result:
                    print(f"Could not parse diagnostic result from output: {output_text[:100]}...")
                    # 即使无法解析，也记录这条数据
                    result_entry = {
                        "input": input_text,
                        "instruction": instruction,
                        "original_output": output_text,
                        "small_model_result": None,
                        "corrected_result": None,
                        "status": "parse_failed"
                    }
                    results.append(result_entry)
                    continue
                
                # Format prompt for RAG整段
                prompt = self.format_prompt(input_text, graph_info)
                
                # 使用微调模型生成响应
                if self.model is not None and self.tokenizer is not None:
                    print("使用微调模型生成RAG整段响应...")
                    model_response = self.generate_model_response(prompt)
                    
                    if model_response:
                        try:
                            print(f"模型生成的响应: {model_response}")
                            parsed_response = self._parse_llm_scores(model_response)

                            if parsed_response:
                                final_result = self.calculate_correction(
                                    small_model_result,
                                    graph_info,
                                    llm_result=parsed_response,
                                )
                                status = "model_structural_correction"
                            else:
                                print("无法解析模型输出，回退到计算方法")
                                final_result = self.calculate_correction(small_model_result, graph_info)
                                status = "model_parse_failed"
                        except Exception as e:
                            print(f"解析模型响应时出错: {e}")
                            final_result = self.calculate_correction(small_model_result, graph_info)
                            status = "model_parse_error"
                    else:
                        # 模型响应生成失败，使用计算方法
                        print("模型响应生成失败，使用计算方法")
                        final_result = self.calculate_correction(small_model_result, graph_info)
                        status = "model_generation_failed"
                        model_response = "无法生成模型响应"
                else:
                    # 没有模型，直接使用计算公式
                    print("模型未加载，使用计算方法")
                    final_result = self.calculate_correction(small_model_result, graph_info)
                    status = "no_model"
                    model_response = None
                
                # 详细记录所有信息
                result_entry = {
                    "input": input_text,
                    "instruction": instruction,
                    "original_output": output_text,
                    "small_model_result": small_model_result,
                    "model_response": model_response,
                    "corrected_result": final_result,
                    "prompt": prompt,
                    "status": status
                }
                
                results.append(result_entry)
                print(f"Processed one data item, status: {status}")
            
            return results
        
        except Exception as e:
            print(f"Error during diagnosis process: {e}")
            import traceback
            traceback.print_exc()
            return []

def main():
    # 设置输入输出路径
    alpaca_dir = "/workspace/CARE/result/llama_factory_dataset/alpaca"
    output_dir = "/workspace/CARE/result/rag_diagnosis"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 检查alpaca目录中的文件
        json_files = [f for f in os.listdir(alpaca_dir) if f.endswith('.json')]
        if not json_files:
            print(f"警告: 在{alpaca_dir}中未找到JSON文件")
            return
        
        print(f"找到以下JSON文件: {json_files}")
        
        # 检查Neo4j连接
        try:
            temp_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            with temp_driver.session() as session:
                result = session.run("MATCH (n:Service) RETURN count(n) as count")
                service_count = result.single()["count"]
                print(f"Neo4j连接成功! 当前数据库有{service_count}个Service节点")
            temp_driver.close()
        except Exception as e:
            print(f"Neo4j连接测试失败: {e}")
            return
        
        # 检查模型路径
        print(f"检查模型路径...")
        print(f"基础模型路径: {BASE_MODEL_PATH}")
        print(f"微调模型路径: {FINETUNED_MODEL_PATH}")
        
        adapter_path = os.path.join(FINETUNED_MODEL_PATH, "adapter_model")
        if os.path.exists(adapter_path) and os.path.isdir(adapter_path):
            print(f"发现LLaMA-Factory LoRA适配器: {adapter_path}")
            if PEFT_AVAILABLE:
                print(f"PEFT库可用，将使用PEFT加载模型")
            else:
                print(f"PEFT库不可用，将尝试直接加载完整微调模型")
        else:
            print(f"未发现LoRA适配器，将尝试加载完整微调模型")
        
        # 实例化RAG系统
        print("初始化RAG系统...")
        rag_system = GraphFaultDiagnosisRAG()
        print("RAG系统初始化完成")
        
        # 处理alpaca目录中的所有json文件
        for filename in json_files:
            print(f"\nProcessing file: {filename}")
            file_path = os.path.join(alpaca_dir, filename)
            
            # 读取数据
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"Successfully loaded {filename}, containing {len(data)} data items")
                
                # 检查第一条数据
                if data and len(data) > 0:
                    first_item = data[0]
                    print(f"First data item fields: {list(first_item.keys())}")
                    print(f"Diagnosis result example: {first_item.get('output', '')[-100:]}")
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
            
            # 运行诊断
            results = rag_system.run_diagnosis(file_path)
            
            # 检查结果
            if not results:
                print(f"Warning: No results produced from processing {filename}")
                # Create an empty result to ensure non-empty output file
                results = [{"warning": f"No results produced from processing {filename}"}]
            else:
                print(f"Successfully extracted {len(results)} results from {filename}")
            
            # 保存结果
            output_path = os.path.join(output_dir, f"rag_{filename}")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"Processing completed, results saved to: {output_path}")
    
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close resources
        if 'rag_system' in locals():
            rag_system.close()
            print("RAG system resources closed")
        print("Processing completed")

if __name__ == "__main__":
    main() 
