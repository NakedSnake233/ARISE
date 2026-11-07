from neo4j import GraphDatabase

# Neo4j连接设置
uri = "bolt://localhost:7687"
user = "neo4j"
password = __import__('os').environ['NEO4J_PASSWORD']

# 创建Neo4j驱动
driver = GraphDatabase.driver(uri, auth=(user, password))

def delete_numeric_suffix_services():
    with driver.session() as session:
        # 1. 先查询需要删除的节点
        result = session.run("""
            MATCH (s:Service) 
            WHERE s.name =~ '.*-[0-9]+$'
            RETURN s.name AS name
        """)
        
        service_names = [record["name"] for record in result]
        print(f"Found {len(service_names)} Service nodes with numeric suffixes")
        
        if not service_names:
            print("No nodes to delete")
            return
        
        # 打印前5个示例（如果有）
        if service_names:
            print("Examples:")
            for name in service_names[:5]:
                print(f"  - {name}")
        
        # 2. 删除这些节点及其关系
        delete_result = session.run("""
            MATCH (s:Service) 
            WHERE s.name =~ '.*-[0-9]+$'
            DETACH DELETE s
            RETURN count(*) AS deleted_count
        """)
        
        deleted_count = delete_result.single()["deleted_count"]
        print(f"Successfully deleted {deleted_count} nodes with their relationships")

try:
    delete_numeric_suffix_services()
finally:
    driver.close() 