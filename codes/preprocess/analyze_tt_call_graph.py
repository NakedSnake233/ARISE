import os
import json
import glob
from util import Info
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

def visualize_call_graph(info):
    """可视化Train Ticket的调用图"""
    G = nx.DiGraph()
    
    # 添加服务节点
    for service in info.service_names:
        G.add_node(service)
    
    # 添加调用边
    for source, targets in info.edge_info.items():
        for target in targets:
            G.add_edge(source, target)
    
    # 绘制调用图
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx(G, pos, node_size=700, node_color="skyblue", 
                     font_size=8, arrows=True, arrowsize=15, width=1.5)
    plt.title("Train Ticket 服务调用图")
    plt.axis('off')
    
    # 保存图像
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "tt_call_graph.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    return G

def analyze_fault_impact(dataset_path, info):
    """分析故障影响并生成统计数据"""
    path = os.path.join(dataset_path, 'data')
    fault_files = glob.glob(os.path.join(path, 'TT.fault-*.json'))
    fault_files.sort()
    
    # 创建故障统计数据结构
    fault_stats = {
        'services': {},
        'fault_types': {'cpu_load': 0, 'network_delay': 0, 'network_loss': 0},
        'impact_counts': {}
    }
    
    # 初始化所有服务的故障计数器
    for service in info.service_names:
        fault_stats['services'][service] = 0
    
    # 构建服务依赖图
    G = nx.DiGraph()
    for service in info.service_names:
        G.add_node(service)
    
    for source, targets in info.edge_info.items():
        for target in targets:
            G.add_edge(source, target)
    
    # 分析每个故障文件
    for file_path in fault_files:
        with open(file_path, 'r') as f:
            record = json.load(f)
        
        for fault in record['faults']:
            # 获取服务名（去除前缀和版本号）
            service_name = fault['name'].split('_')[1]  # dockercomposemanifests_ts-auth-service_1 -> ts-auth-service
            
            # 增加故障类型计数
            fault_stats['fault_types'][fault['fault']] += 1
            
            # 增加服务故障计数
            if service_name in fault_stats['services']:
                fault_stats['services'][service_name] += 1
            
            # 分析故障影响
            # 上游影响：哪些服务调用了出故障的服务
            upstream_services = []
            for s in info.service_names:
                if service_name in info.edge_info.get(s, []):
                    upstream_services.append(s)
            
            # 下游影响：出故障的服务调用了哪些服务
            downstream_services = info.edge_info.get(service_name, [])
            
            # 记录影响计数
            impact_key = f"{service_name}_{fault['fault']}"
            if impact_key not in fault_stats['impact_counts']:
                fault_stats['impact_counts'][impact_key] = {
                    'upstream': len(upstream_services),
                    'downstream': len(downstream_services),
                    'total': len(upstream_services) + len(downstream_services)
                }
    
    # 生成结果
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存故障统计数据为JSON
    with open(os.path.join(output_dir, "tt_fault_stats.json"), 'w') as f:
        json.dump(fault_stats, f, indent=2)
    
    # 生成服务故障频率图表
    service_faults = pd.DataFrame(list(fault_stats['services'].items()), columns=['Service', 'Fault Count'])
    service_faults = service_faults.sort_values('Fault Count', ascending=False)
    
    plt.figure(figsize=(12, 8))
    plt.bar(service_faults['Service'], service_faults['Fault Count'], color='skyblue')
    plt.xticks(rotation=90)
    plt.title('每个服务的故障次数')
    plt.xlabel('服务')
    plt.ylabel('故障次数')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tt_service_fault_count.png"), dpi=300)
    plt.close()
    
    # 生成故障类型分布图表
    fault_types = pd.DataFrame(list(fault_stats['fault_types'].items()), columns=['Fault Type', 'Count'])
    
    plt.figure(figsize=(8, 8))
    plt.pie(fault_types['Count'], labels=fault_types['Fault Type'], autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
    plt.axis('equal')
    plt.title('故障类型分布')
    plt.savefig(os.path.join(output_dir, "tt_fault_type_distribution.png"), dpi=300)
    plt.close()
    
    return fault_stats

def main():
    # 设置路径
    root_path = os.getenv('ROOT_PATH', 'data')
    dataset_path = os.path.join(root_path, 'TT Dataset')
    
    # 获取TT服务实例信息
    info = Info('TrainTicket')
    
    # 可视化调用图
    G = visualize_call_graph(info)
    
    # 分析故障影响
    fault_stats = analyze_fault_impact(dataset_path, info)
    
    print("分析完成，结果已保存到results目录")

if __name__ == "__main__":
    main() 