import json
from neo4j import GraphDatabase

class FaultPropagationViewer:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_fault_types(self):
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (f:Fault)
                    RETURN DISTINCT f.type AS faultType, count(*) AS count
                    ORDER BY count DESC
                """)
                
                print("Available Fault Types:")
                records = list(result)
                if records:
                    for record in records:
                        print(f"  {record['faultType']}: {record['count']} faults")
                else:
                    print("  No fault types found")
                
                print("\n")
        except Exception as e:
            print(f"Error in get_fault_types: {e}")

    def get_services_with_faults(self):
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (f:Fault)
                    WITH f.name AS faultName, f.type AS faultType
                    RETURN faultName, faultType, 1 AS count
                    ORDER BY faultName
                """)
                
                print("Faults in the database:")
                records = list(result)
                if records:
                    for record in records:
                        print(f"  {record['faultName']} ({record['faultType']})")
                else:
                    print("  No faults found")
                
                result = session.run("""
                    MATCH (s:Service)<-[:AFFECTS]-(f:Fault)
                    RETURN s.name AS service, count(*) AS faultCount
                    ORDER BY faultCount DESC
                """)
                
                print("\nServices with Faults:")
                records = list(result)
                if records:
                    for record in records:
                        print(f"  {record['service']}: {record['faultCount']} faults")
                else:
                    print("  No services with faults found")
                
                print("\n")
        except Exception as e:
            print(f"Error in get_services_with_faults: {e}")

    def get_affected_services_by_fault_type(self, fault_type):
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (f:Fault {type: $fault_type})-[:PROPAGATES_TO]->(s:Service)
                    RETURN f.name AS fault, collect(DISTINCT s.name) AS affected_services, 
                           count(DISTINCT s.name) AS affected_count
                    ORDER BY affected_count DESC
                """, fault_type=fault_type)
                
                print(f"Fault Propagation for {fault_type} Faults:")
                records = list(result)
                if records:
                    for record in records:
                        print(f"  {record['fault']} affects {record['affected_count']} services:")
                        services = sorted(record['affected_services'])
                        for service in services:
                            print(f"    - {service}")
                        print()
                else:
                    print(f"  No propagation found for {fault_type} faults\n")
                    return
                
                # Summary of all affected services by this fault type
                result = session.run("""
                    MATCH (f:Fault {type: $fault_type})-[:PROPAGATES_TO]->(s:Service)
                    RETURN count(DISTINCT f) AS faults, 
                           count(DISTINCT s.name) AS total_affected_services,
                           collect(DISTINCT s.name) AS affected_services
                """, fault_type=fault_type)
                
                record = result.single()
                if record and record['faults'] > 0:
                    print(f"Summary for {fault_type} Faults:")
                    print(f"  {record['faults']} faults affect {record['total_affected_services']} unique services")
                    print(f"  Affected services: {sorted(record['affected_services'])}")
                else:
                    print(f"  No summary data available for {fault_type} faults")
                print()
        except Exception as e:
            print(f"Error in get_affected_services_by_fault_type: {e}")

    def get_propagation_paths(self, fault_service, fault_type):
        try:
            with self.driver.session() as session:
                # First check if the fault exists
                result = session.run("""
                    MATCH (f:Fault {name: $service, type: $type})
                    RETURN count(f) AS count
                """, service=fault_service, type=fault_type)
                
                record = result.single()
                if not record or record['count'] == 0:
                    print(f"Fault {fault_service} ({fault_type}) not found in the database")
                    return
                
                result = session.run("""
                    MATCH path = (f:Fault {name: $service, type: $type})-[:PROPAGATES_TO]->(s:Service)
                    RETURN s.name AS affected_service
                    ORDER BY affected_service
                """, service=fault_service, type=fault_type)
                
                affected_services = [record["affected_service"] for record in result]
                
                print(f"Propagation from {fault_service} ({fault_type}):")
                if affected_services:
                    for service in affected_services:
                        print(f"  - {service}")
                else:
                    print("  No propagation paths found")
                print()
                
                # Now get the service call paths that may have contributed to propagation
                result = session.run("""
                    MATCH (f:Fault {name: $service, type: $type})-[:AFFECTS]->(s1:Service),
                          path = (s1:Service)-[:CALLS*]->(s2:Service)
                    WHERE s1.name = $service
                    RETURN [node in nodes(path) | node.name] AS service_path
                """, service=fault_service, type=fault_type)
                
                print(f"Service call paths from {fault_service}:")
                paths = [record["service_path"] for record in result]
                
                if paths:
                    unique_paths = []
                    for path in paths:
                        if path not in unique_paths:
                            unique_paths.append(path)
                    
                    for path in unique_paths:
                        print(f"  {' -> '.join(path)}")
                else:
                    print("  No service call paths found")
        except Exception as e:
            print(f"Error in get_propagation_paths: {e}")

def main():
    # Neo4j connection details
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = __import__('os').environ['NEO4J_PASSWORD']
    
    # Initialize viewer
    viewer = FaultPropagationViewer(uri, user, password)
    
    try:
        # Show available fault types
        viewer.get_fault_types()
        
        # Show services with faults
        viewer.get_services_with_faults()
        
        # Show affected services by fault type
        fault_types = ['cpu_load', 'network_delay', 'network_loss']
        for fault_type in fault_types:
            viewer.get_affected_services_by_fault_type(fault_type)
        
        # Show propagation paths for a specific fault
        viewer.get_propagation_paths('ts-travel-service', 'network_loss')
        viewer.get_propagation_paths('ts-auth-service', 'cpu_load')
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        viewer.close()

if __name__ == "__main__":
    main() 