from neo4j import GraphDatabase
import sys

def query_neo4j(uri, user, password):
    print(f"Connecting to Neo4j at {uri}...")
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        print("Connection established!")
        
        # Test connection
        with driver.session() as session:
            result = session.run("MATCH (n) RETURN COUNT(n) AS count")
            record = result.single()
            print(f"Database contains {record['count']} nodes")
            
            # 查询所有故障节点
            print("\n==== 故障节点 ====")
            result = session.run("""
                MATCH (f:Fault)
                RETURN f.service as service, f.type as type, f.start_time as start_time
                ORDER BY service, type
            """)
            
            fault_count = 0
            for record in result:
                fault_count += 1
                print(f"Service: {record['service']}, Type: {record['type']}, Start Time: {record['start_time']}")
            
            print(f"Total fault nodes: {fault_count}")
            
            # 查询所有服务节点
            print("\n==== 服务节点 ====")
            result = session.run("""
                MATCH (s:Service)
                RETURN s.name as name
                ORDER BY name
            """)
            
            service_count = 0
            for record in result:
                service_count += 1
                print(f"Service: {record['name']}")
            
            print(f"Total service nodes: {service_count}")
            
            # 查询所有关系
            print("\n==== 关系统计 ====")
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as type, COUNT(r) as count
                ORDER BY count DESC
            """)
            
            for record in result:
                print(f"Relationship: {record['type']}, Count: {record['count']}")
            
        driver.close()
        print("Connection closed.")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        print(f"Connection failed to {uri} with user {user}")
        sys.exit(1)

if __name__ == "__main__":
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = __import__('os').environ['NEO4J_PASSWORD']
    
    query_neo4j(uri, user, password) 