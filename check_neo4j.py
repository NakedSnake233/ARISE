import json
from neo4j import GraphDatabase

class Neo4jChecker:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def check_fault_nodes(self):
        with self.driver.session() as session:
            result = session.run("MATCH (f:Fault) RETURN COUNT(f) AS faultCount")
            count = result.single()["faultCount"]
            print(f"Number of Fault nodes: {count}")
            
            if count > 0:
                result = session.run("MATCH (f:Fault) RETURN f.name, f.type LIMIT 5")
                print("\nSample Fault nodes:")
                for record in result:
                    print(f"  {record['f.name']} - {record['f.type']}")
            return count

    def check_service_nodes(self):
        with self.driver.session() as session:
            result = session.run("MATCH (s:Service) RETURN COUNT(s) AS serviceCount")
            count = result.single()["serviceCount"]
            print(f"\nNumber of Service nodes: {count}")
            
            if count > 0:
                result = session.run("MATCH (s:Service) RETURN s.name LIMIT 5")
                print("\nSample Service nodes:")
                for record in result:
                    print(f"  {record['s.name']}")
            return count

    def check_propagation_relationships(self):
        with self.driver.session() as session:
            result = session.run("MATCH ()-[r:PROPAGATES_TO]->() RETURN COUNT(r) AS propagationCount")
            count = result.single()["propagationCount"]
            print(f"\nNumber of PROPAGATES_TO relationships: {count}")
            
            if count > 0:
                result = session.run("""
                    MATCH (f:Fault)-[r:PROPAGATES_TO]->(s:Service) 
                    RETURN f.name AS fault, f.type AS type, s.name AS service 
                    LIMIT 5
                """)
                print("\nSample PROPAGATES_TO relationships:")
                for record in result:
                    print(f"  {record['fault']} ({record['type']}) -> {record['service']}")
            return count

    def check_affects_relationships(self):
        with self.driver.session() as session:
            result = session.run("MATCH ()-[r:AFFECTS]->() RETURN COUNT(r) AS affectsCount")
            count = result.single()["affectsCount"]
            print(f"\nNumber of AFFECTS relationships: {count}")
            
            if count > 0:
                result = session.run("""
                    MATCH (n1)-[r:AFFECTS]->(n2) 
                    RETURN labels(n1)[0] AS sourceType, 
                           CASE WHEN n1.name IS NOT NULL THEN n1.name ELSE 'Unknown' END AS source, 
                           labels(n2)[0] AS targetType, 
                           CASE WHEN n2.name IS NOT NULL THEN n2.name ELSE 'Unknown' END AS target 
                    LIMIT 5
                """)
                print("\nSample AFFECTS relationships:")
                for record in result:
                    print(f"  {record['sourceType']}({record['source']}) -> {record['targetType']}({record['target']})")
            return count

    def check_calls_relationships(self):
        with self.driver.session() as session:
            result = session.run("MATCH ()-[r:CALLS]->() RETURN COUNT(r) AS callsCount")
            count = result.single()["callsCount"]
            print(f"\nNumber of CALLS relationships: {count}")
            
            if count > 0:
                result = session.run("""
                    MATCH (s1:Service)-[r:CALLS]->(s2:Service) 
                    RETURN s1.name AS caller, s2.name AS callee 
                    LIMIT 5
                """)
                print("\nSample CALLS relationships:")
                for record in result:
                    print(f"  {record['caller']} -> {record['callee']}")
            return count

    def run_all_checks(self):
        faults = self.check_fault_nodes()
        services = self.check_service_nodes()
        propagations = self.check_propagation_relationships()
        affects = self.check_affects_relationships()
        calls = self.check_calls_relationships()
        
        print("\n=== Summary ===")
        print(f"Fault nodes: {faults}")
        print(f"Service nodes: {services}")
        print(f"PROPAGATES_TO relationships: {propagations}")
        print(f"AFFECTS relationships: {affects}")
        print(f"CALLS relationships: {calls}")

def main():
    # Neo4j connection details
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = __import__('os').environ['NEO4J_PASSWORD']
    
    # Initialize checker
    checker = Neo4jChecker(uri, user, password)
    
    try:
        # Run all checks
        checker.run_all_checks()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        checker.close()

if __name__ == "__main__":
    main() 