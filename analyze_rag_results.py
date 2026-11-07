import json
import re
import os
from collections import defaultdict

def extract_actual_fault_node(input_text):
    """从输入文本中提取真实故障节点"""
    if not isinstance(input_text, str):
        return None
    
    pattern = r"Node (\d+):[^A]*Actual fault node"
    match = re.search(pattern, input_text)
    if match:
        return int(match.group(1))
    return None

def extract_actual_fault_from_result(result):
    for key in ("fault_node", "actual_fault_node", "actual_fault"):
        value = result.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.lstrip("-").isdigit():
            return int(value)

    raw = result.get("raw")
    if isinstance(raw, dict):
        value = extract_actual_fault_from_result(raw)
        if value is not None:
            return value

    output_text = result.get("original_output", "") or result.get("output", "")
    if isinstance(output_text, str):
        match = re.search(r"Diagnosis result:\s*Node\s+(\d+)\s+is the fault node", output_text)
        if match:
            return int(match.group(1))

    input_text = result.get("input", "")
    if isinstance(input_text, dict):
        input_text = input_text.get("text", "")
    return extract_actual_fault_node(input_text)

def extract_fault_probabilities(output_text):
    """从输出文本中提取节点故障概率"""
    if not isinstance(output_text, str):
        return {}
        
    fault_probs = {}
    pattern = r"Node (\d+) fault probability: ([0-9.]+)"
    matches = re.findall(pattern, output_text)
    for node_id, prob in matches:
        fault_probs[int(node_id)] = float(prob)
    return fault_probs

def get_top_n_nodes(fault_probs, n=5):
    """获取故障概率最高的前N个节点"""
    sorted_nodes = sorted(fault_probs.items(), key=lambda x: x[1], reverse=True)
    return [node for node, _ in sorted_nodes[:n]]

def analyze_results(results_file):
    """分析RAG诊断结果，计算精确度"""
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # 处理不同格式的结果文件
        if isinstance(results, dict):
            # 如果是字典，转换为列表
            results = [results]
        elif not isinstance(results, list):
            print(f"无法识别的结果格式: {type(results)}")
            return {
                'total_cases': 0,
                'correct_top1': 0,
                'correct_top3': 0,
                'correct_top5': 0,
                'accuracy_top1': 0,
                'accuracy_top3': 0,
                'accuracy_top5': 0
            }
        
        valid_cases = 0
        correct_top1 = 0
        correct_top3 = 0
        correct_top5 = 0
        
        for result in results:
            if not isinstance(result, dict):
                continue
                
            actual_fault = extract_actual_fault_from_result(result)
            if actual_fault is None:
                continue
            
            # 提取模型预测的故障概率
            output_text = result.get('original_output', '')
            if isinstance(output_text, dict):
                output_text = output_text.get('text', '')
                
            fault_probs = extract_fault_probabilities(output_text)
            if not fault_probs:
                continue
            
            # 计数有效样本
            valid_cases += 1
            
            # 获取前N个候选节点
            top1_nodes = get_top_n_nodes(fault_probs, 1)
            top3_nodes = get_top_n_nodes(fault_probs, 3)
            top5_nodes = get_top_n_nodes(fault_probs, 5)
            
            # 计算精确度
            if actual_fault in top1_nodes:
                correct_top1 += 1
            if actual_fault in top3_nodes:
                correct_top3 += 1
            if actual_fault in top5_nodes:
                correct_top5 += 1
        
        # 计算精确度指标
        accuracy_top1 = correct_top1 / valid_cases if valid_cases > 0 else 0
        accuracy_top3 = correct_top3 / valid_cases if valid_cases > 0 else 0
        accuracy_top5 = correct_top5 / valid_cases if valid_cases > 0 else 0
        
        return {
            'total_cases': valid_cases,
            'correct_top1': correct_top1,
            'correct_top3': correct_top3,
            'correct_top5': correct_top5,
            'accuracy_top1': accuracy_top1,
            'accuracy_top3': accuracy_top3,
            'accuracy_top5': accuracy_top5
        }
    except Exception as e:
        print(f"处理文件时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return {
            'total_cases': 0,
            'correct_top1': 0,
            'correct_top3': 0,
            'correct_top5': 0,
            'accuracy_top1': 0,
            'accuracy_top3': 0,
            'accuracy_top5': 0
        }

def main():
    """主函数"""
    results_dir = 'result/rag_diagnosis'
    results_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    
    all_results = {}
    
    for file in results_files:
        file_path = os.path.join(results_dir, file)
        print(f"分析文件: {file}")
        try:
            metrics = analyze_results(file_path)
            if metrics['total_cases'] > 0:
                all_results[file] = metrics
                
                print(f"有效样本数: {metrics['total_cases']}")
                print(f"Top-1精确度: {metrics['correct_top1']}/{metrics['total_cases']} = {metrics['accuracy_top1']:.4f}")
                print(f"Top-3精确度: {metrics['correct_top3']}/{metrics['total_cases']} = {metrics['accuracy_top3']:.4f}")
                print(f"Top-5精确度: {metrics['correct_top5']}/{metrics['total_cases']} = {metrics['accuracy_top5']:.4f}")
            else:
                print(f"文件中没有有效样本")
            print("-" * 50)
        except Exception as e:
            print(f"处理文件 {file} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 计算所有文件的平均精确度
    if all_results:
        total_cases = sum(r['total_cases'] for r in all_results.values())
        total_correct_top1 = sum(r['correct_top1'] for r in all_results.values())
        total_correct_top3 = sum(r['correct_top3'] for r in all_results.values())
        total_correct_top5 = sum(r['correct_top5'] for r in all_results.values())
        
        avg_accuracy_top1 = total_correct_top1 / total_cases if total_cases > 0 else 0
        avg_accuracy_top3 = total_correct_top3 / total_cases if total_cases > 0 else 0
        avg_accuracy_top5 = total_correct_top5 / total_cases if total_cases > 0 else 0
        
        print("\n总体结果:")
        print(f"总样本数: {total_cases}")
        print(f"Top-1精确度: {total_correct_top1}/{total_cases} = {avg_accuracy_top1:.4f}")
        print(f"Top-3精确度: {total_correct_top3}/{total_cases} = {avg_accuracy_top3:.4f}")
        print(f"Top-5精确度: {total_correct_top5}/{total_cases} = {avg_accuracy_top5:.4f}")

if __name__ == "__main__":
    main() 
