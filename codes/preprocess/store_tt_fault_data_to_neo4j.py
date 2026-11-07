from neo4j import GraphDatabase
import os
import json
import glob
from util import Info

class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()
    
    def create_fault_relationships(self, fault_data):
        with self.driver.session() as session:
            # 创建约束，确保故障节点的唯一性
            session.run("CREATE CONSTRAINT fault_id IF NOT EXISTS FOR (f:Fault) REQUIRE f.id IS UNIQUE")
            
            # 遍历并处理每个故障数据
            for idx, fault_record in enumerate(fault_data):
                record_id = f"fault_record_{idx}"
                
                # 创建故障记录节点
                session.run(
                    "CREATE (r:FaultRecord {id: $record_id, start_time: $start, end_time: $end})",
                    record_id=record_id,
                    start=fault_record['start'],
                    end=fault_record['end']
                )
                
                # 处理每个故障
                for i, fault in enumerate(fault_record['faults']):
                    fault_id = f"{record_id}_fault_{i}"
                    service_name = fault['service']
                    
                    # 确保服务名称格式正确
                    if not service_name.startswith('ts-'):
                        service_name = f"ts-{service_name}-service"
                    
                    # 创建故障节点
                    session.run(
                        "CREATE (f:Fault {id: $fault_id, type: $fault_type, start_time: $start, end_time: $end})",
                        fault_id=fault_id,
                        fault_type=fault['fault_type'],
                        start=fault['s'],
                        end=fault['e']
                    )
                    
                    # 将故障与其记录关联
                    session.run(
                        "MATCH (f:Fault {id: $fault_id}), (r:FaultRecord {id: $record_id}) "
                        "CREATE (r)-[:CONTAINS]->(f)",
                        fault_id=fault_id,
                        record_id=record_id
                    )
                    
                    # 将故障事件与注入服务关联；论文中的 AFFECTS 保留给
                    # Service -> Service 的故障传播关系。
                    session.run(
                        "MATCH (f:Fault {id: $fault_id}), (s:Service {name: $service_name}) "
                        "CREATE (f)-[:INJECTED_IN]->(s)",
                        fault_id=fault_id,
                        service_name=service_name
                    )
                    session.run(
                        "MATCH (f:Fault {id: $fault_id}), (s:Service {name: $service_name}) "
                        "CREATE (f)-[:PROPAGATES_TO]->(s)",
                        fault_id=fault_id,
                        service_name=service_name
                    )
    
    def create_fault_impact_relationships(self, info):
        """创建故障影响关系 - 根据调用图分析故障传播"""
        with self.driver.session() as session:
            # 对于每个故障，找出受影响的服务以及调用这个服务的其他服务
            session.run("""
                MATCH (f:Fault)-[:INJECTED_IN]->(s:Service)
                MATCH (caller:Service)-[:CALLS]->(s)
                WHERE NOT EXISTS((f)-[:IMPACTS]->(caller))
                CREATE (f)-[:IMPACTS {type: 'upstream_impact'}]->(caller)
                MERGE (s)-[:AFFECTS]->(caller)
            """)
            
            # 对于每个故障，找出受影响的服务以及它调用的其他服务
            session.run("""
                MATCH (f:Fault)-[:INJECTED_IN]->(s:Service)
                MATCH (s)-[:CALLS]->(callee:Service)
                WHERE NOT EXISTS((f)-[:IMPACTS]->(callee))
                CREATE (f)-[:IMPACTS {type: 'downstream_impact'}]->(callee)
                MERGE (s)-[:AFFECTS]->(callee)
            """)

def process_tt_fault_files(dataset_path):
    """处理TT数据集中的故障文件"""
    path = os.path.join(dataset_path, 'data')
    fault_files = glob.glob(os.path.join(path, 'TT.fault-*.json'))
    fault_files.sort()
    
    fault_data = []
    
    for file_path in fault_files:
        with open(file_path, 'r') as f:
            record = json.load(f)
        
        # 处理记录，保持原始时间戳
        processed_record = {
            'start': record['start'],
            'end': record['end'],
            'faults': []
        }
        
        for fault in record['faults']:
            service_name = fault['name'].split('_')[1]  # dockercomposemanifests_ts-auth-service_1 -> ts-auth-service
            
            processed_record['faults'].append({
                'service': service_name,
                'fault_type': fault['fault'],
                's': fault['start'],
                'e': fault['start'] + fault['duration']
            })
        
        fault_data.append(processed_record)
    
    return fault_data

def main():
    # 设置路径
    root_path = os.getenv('ROOT_PATH', 'data')
    dataset_path = os.path.join(root_path, 'TT Dataset')
    
    # 处理故障文件
    fault_data = process_tt_fault_files(dataset_path)
    
    # 获取TT服务实例信息
    info = Info('TrainTicket')
    
    # Neo4j连接参数
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = __import__('os').environ['NEO4J_PASSWORD']
    
    # 连接Neo4j并存储数据
    conn = Neo4jConnection(uri, user, password)
    try:
        # 首先确保基本服务关系已经建立
        # 可以先运行store_tt_edges_to_neo4j.py
        
        # 存储故障数据和关系
        conn.create_fault_relationships(fault_data)
        
        # 创建故障影响关系
        conn.create_fault_impact_relationships(info)
        
        print("成功将TT故障数据和调用图关系存储到Neo4j中")
    except Exception as e:
        print(f"错误: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main() 
