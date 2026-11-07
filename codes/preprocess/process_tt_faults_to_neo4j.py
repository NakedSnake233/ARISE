import os
import subprocess
import time

def run_command(command, description):
    """运行命令并显示状态"""
    print(f"开始 {description}...")
    start_time = time.time()
    process = subprocess.run(command, shell=True, check=True)
    end_time = time.time()
    print(f"{description} 完成，耗时 {end_time - start_time:.2f} 秒")
    return process

def main():
    """主函数：执行TT故障数据处理和Neo4j导入的完整流程"""
    
    # 确保结果目录存在
    os.makedirs("results", exist_ok=True)
    
    try:
        # 步骤1: 确保Neo4j服务器正在运行
        print("请确保Neo4j服务器已启动并运行在localhost:7687")
        input("准备好继续请按回车...")
        
        # 步骤2: 首先导入Train Ticket服务调用关系图
        run_command("python codes/preprocess/store_tt_edges_to_neo4j.py", 
                   "导入Train Ticket服务调用关系图")
        
        # 步骤3: 处理故障数据并导入到Neo4j
        run_command("python codes/preprocess/store_tt_fault_data_to_neo4j.py", 
                   "处理故障数据并导入到Neo4j")
        
        # 步骤4: 分析调用图和故障影响
        run_command("python codes/preprocess/analyze_tt_call_graph.py", 
                   "分析调用图和故障影响")
        
        print("\n所有处理完成! 结果已保存在results目录中")
        print("可以在Neo4j Browser中使用以下查询查看结果:")
        print("1. 查看所有服务和它们的调用关系:")
        print("   MATCH (s1:Service)-[r:CALLS]->(s2:Service) RETURN s1, r, s2")
        print("2. 查看所有故障及其影响的服务:")
        print("   MATCH (f:Fault)-[r:PROPAGATES_TO]->(s:Service) RETURN f, r, s")
        print("3. 查看特定服务的故障:")
        print("   MATCH (f:Fault)-[:INJECTED_IN]->(s:Service {name: 'ts-auth-service'}) RETURN f, s")
        
    except Exception as e:
        print(f"执行过程中出错: {e}")

if __name__ == "__main__":
    main() 
