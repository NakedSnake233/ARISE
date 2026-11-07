#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
from rag_fault_diagnosis import GraphFaultDiagnosisRAG

def main():
    """测试RAG整段功能 - 使用LLaMA-Factory微调模型版本"""
    print("启动RAG整段功能测试 (使用微调模型)")
    
    # 初始化RAG系统
    print("初始化RAG系统...")
    rag_system = GraphFaultDiagnosisRAG()
    
    try:
        # 获取图数据库信息
        print("获取图数据库信息...")
        graph_info = rag_system.get_service_calls_and_faults()
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
        prompt = rag_system.format_prompt(test_input, graph_info)
        print("\n测试提示构建完成:")
        print("="*50)
        print(prompt)
        print("="*50)
        
        # 使用微调模型生成响应
        if rag_system.model is not None and rag_system.tokenizer is not None:
            print("\n使用模型生成响应...")
            response = rag_system.generate_model_response(prompt)
            
            print("\n模型响应:")
            print("-"*50)
            print(response)
            print("-"*50)
            
            # 解析响应
            print("\n解析响应...")
            parsed_result = rag_system.parse_diagnosis_result(response)
            print(f"解析结果: {json.dumps(parsed_result, ensure_ascii=False, indent=2)}")
            
            # 计算修正后的结果
            if parsed_result:
                print("\n应用修正计算...")
                corrected_result = rag_system.calculate_correction(parsed_result, graph_info)
                print(f"修正后结果: {json.dumps(corrected_result, ensure_ascii=False, indent=2)}")
            else:
                print("无法解析模型响应，不进行修正计算")
        else:
            print("模型未加载，无法生成响应")
    
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 关闭资源
        rag_system.close()
        print("测试完成")

if __name__ == "__main__":
    main() 