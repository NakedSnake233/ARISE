import json
import os
from neo4j import GraphDatabase

class FaultPropagationAnalyzer:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def load_fault_data(self, fault_file):
        with open(fault_file, 'r') as f:
            return json.load(f)

    def load_trace_data(self, trace_file):
        with open(trace_file, 'r') as f:
            data = json.load(f)
            print(f"\nLoaded trace data from {trace_file}")
            print(f"Number of timestamps: {len(data)}")
            if len(data) > 0:
                print("First timestamp example:")
                first_timestamp = next(iter(data))
                print(f"Timestamp: {first_timestamp}")
                print(f"Services: {list(data[first_timestamp].keys())[:5]}... (showing first 5)")
            return data

    def get_service_name(self, full_name):
        # Remove the numeric ID from service name
        # e.g., "ts-ticketinfo-service-20" -> "ts-ticketinfo-service"
        parts = full_name.split('-')
        if parts[-1].isdigit():
            return '-'.join(parts[:-1])
        return full_name

    def find_affected_services(self, fault_data, trace_data):
        # Dictionary to store affected services for each fault
        # {fault_id: {"service": service_name, "type": fault_type, "direct": set(direct_services), "indirect": set(indirect_services)}}
        affected_services = {}
        
        print(f"\nProcessing fault data:")
        
        for fault_idx, fault in enumerate(fault_data['faults']):
            fault_service = fault['service']
            fault_start = fault['s']
            fault_end = fault['e']
            fault_type = fault['fault_type']
            fault_id = f"{fault_service}_{fault_type}_{fault_start}"
            
            print(f"\nProcessing fault {fault_idx+1}/{len(fault_data['faults'])}: {fault_service} ({fault_type}) from {fault_start} to {fault_end}")
            
            # Initialize sets for this fault
            affected_services[fault_id] = {
                "service": fault_service,
                "type": fault_type,
                "direct": set(),
                "indirect": set()
            }
            
            # Count traces during fault period
            trace_count = 0
            
            # Find services involved during fault period
            for timestamp, services_data in trace_data.items():
                timestamp = int(timestamp)
                if fault_start <= timestamp <= fault_end:
                    trace_count += 1
                    
                    # Get all services involved in this trace
                    services = [self.get_service_name(service) for service in services_data.keys()]
                    
                    if services:
                        print(f"Found {len(services)} services at timestamp {timestamp}")
                        
                        # Classify services as direct or indirect based on fault service
                        if fault_service in services:
                            # Services directly affected (called with fault service in same trace)
                            for service in services:
                                if service != fault_service:
                                    affected_services[fault_id]["direct"].add(service)
                                    print(f"  Direct service affected: {service}")
                        else:
                            # Services indirectly affected (in traces during fault period but fault service not present)
                            for service in services:
                                affected_services[fault_id]["indirect"].add(service)
                                print(f"  Indirect service affected: {service}")
            
            print(f"Total traces during fault period: {trace_count}")
            print(f"Direct affected services: {len(affected_services[fault_id]['direct'])}")
            print(f"Indirect affected services: {len(affected_services[fault_id]['indirect'])}")
        
        return affected_services

    def store_fault_propagation(self, affected_services):
        with self.driver.session() as session:
            for fault_id, data in affected_services.items():
                fault_service = data["service"]
                fault_type = data["type"]
                direct_services = data["direct"]
                indirect_services = data["indirect"]
                
                print(f"\nStoring fault propagation for {fault_id}")
                
                # Create fault node
                session.run("""
                    MERGE (f:Fault {
                        service: $service,
                        type: $type,
                        id: $fault_id
                    })
                    RETURN f
                """, service=fault_service, type=fault_type, fault_id=fault_id)
                
                # Create service node for fault service
                session.run("""
                    MERGE (s:Service {name: $service})
                """, service=fault_service)
                
                # Create direct propagation relationships
                for service in direct_services:
                    print(f"Creating PROPAGATES_TO relationship from {fault_service} to {service}")
                    
                    # Create service node
                    session.run("""
                        MERGE (s:Service {name: $service})
                    """, service=service)
                    
                    # Create direct propagation relationship
                    session.run("""
                        MATCH (f:Fault {id: $fault_id})
                        MATCH (s:Service {name: $service})
                        MERGE (f)-[:PROPAGATES_TO]->(s)
                    """, fault_id=fault_id, service=service)
                
                # Create indirect propagation relationships
                for service in indirect_services:
                    print(f"Creating INDIRECTLY_AFFECTS relationship from {fault_service} to {service}")
                    
                    # Create service node
                    session.run("""
                        MERGE (s:Service {name: $service})
                    """, service=service)
                    
                    # Create indirect propagation relationship
                    session.run("""
                        MATCH (f:Fault {id: $fault_id})
                        MATCH (s:Service {name: $service})
                        MERGE (f)-[:INDIRECTLY_AFFECTS]->(s)
                    """, fault_id=fault_id, service=service)

def main():
    # Neo4j connection details
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = __import__('os').environ['NEO4J_PASSWORD']

    # Data paths
    data_dir = "codes/preprocess/parsed_data/TT"
    
    # Initialize analyzer
    analyzer = FaultPropagationAnalyzer(uri, user, password)
    
    try:
        # Process each records file
        for i in range(9):  # records0.json to records8.json
            records_file = f"records{i}.json"
            traces_file = f"traces{i}.json"
            
            records_path = os.path.join(data_dir, records_file)
            traces_path = os.path.join(data_dir, traces_file)
            
            if os.path.exists(records_path) and os.path.exists(traces_path):
                try:
                    print(f"\nProcessing {records_file} and {traces_file}")
                    # Load data
                    fault_data = analyzer.load_fault_data(records_path)
                    trace_data = analyzer.load_trace_data(traces_path)
                    
                    # Find affected services during fault periods
                    affected_services = analyzer.find_affected_services(fault_data, trace_data)
                    
                    # Store fault propagation in Neo4j
                    analyzer.store_fault_propagation(affected_services)
                    
                    print(f"Successfully processed {records_file}")
                except Exception as e:
                    print(f"Error processing {records_file}: {e}")
    
    except Exception as e:
        print(f"Error: {e}")
    finally:
        analyzer.close()

if __name__ == "__main__":
    main() 