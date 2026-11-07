import json
import os
from neo4j import GraphDatabase
from util import Info

class FaultPropagationAnalyzer:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.info = Info("TrainTicket")
        self.valid_call_edges = {
            (caller, callee)
            for caller, callees in self.info.edge_info.items()
            for callee in callees
        }

    def close(self):
        self.driver.close()

    def delete_fault_related_data(self):
        with self.driver.session() as session:
            # Delete all fault-related nodes and relationships
            session.run("""
                MATCH (n:Fault)
                DETACH DELETE n
            """)
            # Also delete existing propagation relationships
            session.run("""
                MATCH ()-[r:PROPAGATES_TO]->() DELETE r
            """)
            session.run("""
                MATCH ()-[r:INDIRECTLY_AFFECTS]->() DELETE r
            """)
            session.run("""
                MATCH ()-[r:PROPAGATES_THROUGH]->() DELETE r
            """)
            session.run("""
                MATCH ()-[r:AFFECTS]->() DELETE r
            """)
            session.run("""
                MATCH ()-[r:INJECTED_IN]->() DELETE r
            """)
            print("Successfully deleted fault-related nodes and relationships")

    def load_fault_data(self, fault_file):
        with open(fault_file, 'r') as f:
            return json.load(f)

    def load_trace_data(self, trace_file):
        try:
            with open(trace_file, 'r') as f:
                file_content = f.read().strip()
                if not file_content:
                    print(f"\nWARNING: Trace file {trace_file} is empty!")
                    return {}
                
                try:
                    data = json.loads(file_content)
                    print(f"\nLoaded trace data from {trace_file}")
                    print(f"Number of timestamps: {len(data)}")
                    if len(data) > 0:
                        print("First timestamp example:")
                        first_timestamp = next(iter(data))
                        print(f"Timestamp: {first_timestamp}")
                        first_services = list(data[first_timestamp].keys())[:3]  # Show first 3 services
                        print(f"Sample services: {first_services}")
                        print(f"Total services in first timestamp: {len(data[first_timestamp])}")
                    else:
                        print("WARNING: Trace file contains no timestamps!")
                    return data
                except json.JSONDecodeError as e:
                    print(f"ERROR: Invalid JSON in {trace_file}: {e}")
                    print(f"First 100 characters: {file_content[:100]}")
                    return {}
        except Exception as e:
            print(f"ERROR: Failed to read trace file {trace_file}: {e}")
            return {}

    def get_standard_service_name(self, full_name):
        # Remove the numeric ID from service name and standardize format
        # e.g., "ts-ticketinfo-service-20" -> "ts-ticketinfo-service"
        print(f"  Standardizing service name: {full_name}")
        parts = full_name.split('-')
        if len(parts) >= 3:
            # Remove the numeric suffix if it exists
            if parts[-1].isdigit():
                parts = parts[:-1]
            standard_name = '-'.join(parts)
            print(f"  Standardized to: {standard_name}")
            return standard_name
        
        print(f"  Keeping original name: {full_name}")
        return full_name

    def find_affected_services(self, fault_data, trace_data):
        affected_services = {}  # {fault_service: {affected_services: set(), all_calls: []}}
        print(f"\nProcessing fault data:")
        
        print(f"Total faults: {len(fault_data['faults'])}")
        print(f"Trace data contains {len(trace_data)} timestamps")
        
        for fault in fault_data['faults']:
            # Extract service name correctly from fault record
            # Original format example: "ts-auth-service"
            fault_service_raw = fault['service']  # e.g., "ts-auth-service"
            fault_start = fault['s']
            fault_end = fault['e']
            fault_type = fault['fault_type']
            print(f"\nProcessing fault: {fault_service_raw} ({fault_type}) from {fault_start} to {fault_end}")
            
            # Initialize for this fault
            if fault_service_raw not in affected_services:
                affected_services[fault_service_raw] = {
                    'affected_services': set(),  # 所有受影响的服务
                    'all_calls': []  # 故障期间的所有调用关系
                }
            
            # Find traces during fault period
            matching_timestamps = []
            possible_fault_services = []  # List of possible matching service names in traces
            
            # First scan to find all possible service names that could match the fault service
            for timestamp_str, services_data in trace_data.items():
                for service_name in services_data.keys():
                    std_service_name = self.get_standard_service_name(service_name)
                    if std_service_name == fault_service_raw and service_name not in possible_fault_services:
                        possible_fault_services.append(service_name)
            
            print(f"Possible fault service matches in traces: {possible_fault_services}")
            
            for timestamp_str, services_data in trace_data.items():
                timestamp = int(timestamp_str)
                if fault_start <= timestamp <= fault_end:
                    matching_timestamps.append(timestamp)
                    print(f"Found trace at timestamp {timestamp} during fault period")
                    
                    # First check if we have the fault service in this trace
                    fault_service_found = False
                    matched_service_name = None
                    
                    for service_name in services_data.keys():
                        std_service_name = self.get_standard_service_name(service_name)
                        if std_service_name == fault_service_raw:
                            fault_service_found = True
                            matched_service_name = service_name
                            break
                    
                    if not fault_service_found:
                        print(f"  Warning: Fault service {fault_service_raw} not found in this trace, skipping")
                        continue
                    
                    print(f"  Found fault service as {matched_service_name}")
                    
                    # Print sample of services in this trace
                    sample_services = list(services_data.keys())[:3]  # First 3 services
                    print(f"  Sample services in this trace: {sample_services}")
                    print(f"  Total services in this trace: {len(services_data)}")
                    
                    # Collect all services in this trace
                    trace_services = []
                    for service in services_data.keys():
                        std_service_name = self.get_standard_service_name(service)
                        trace_services.append(std_service_name)
                        
                        # Add to affected services (only if not the fault service)
                        if std_service_name != fault_service_raw:
                            affected_services[fault_service_raw]['affected_services'].add(std_service_name)
                            print(f"  Added affected service: {std_service_name}")
                    
                    # Create call relationships only when they are present in
                    # the service dependency graph. This matches Algorithm 1's
                    # recordCall step and avoids turning a trace snapshot into
                    # a fully-connected graph.
                    trace_service_set = set(trace_services)
                    for caller, callee in self.valid_call_edges:
                        if caller in trace_service_set and callee in trace_service_set:
                            affected_services[fault_service_raw]['all_calls'].append((caller, callee))
                            print(f"  Found call: {caller} -> {callee}")
            
            print(f"Total matching timestamps for this fault: {len(matching_timestamps)}")
            print(f"Total affected services for {fault_service_raw}: {len(affected_services[fault_service_raw]['affected_services'])}")
            if len(affected_services[fault_service_raw]['affected_services']) > 0:
                print(f"Affected services: {affected_services[fault_service_raw]['affected_services']}")
        
        return affected_services

    def store_fault_propagation(self, fault_data, affected_services):
        with self.driver.session() as session:
            # Create fault nodes and relationships
            for fault in fault_data['faults']:
                # Get the fault service name directly from the fault record
                fault_service = fault['service']
                fault_type = fault['fault_type']
                fault_start = fault['s']
                
                print(f"\nCreating fault node: {fault_service} with type {fault_type}")
                
                # Create fault node with name property as the primary identifier
                result = session.run("""
                    MERGE (f:Fault {name: $service})
                    SET f.type = $type,
                        f.start_time = $start_time
                    RETURN f
                """, service=fault_service, type=fault_type, start_time=fault_start)
                
                print(f"Created fault node: {result.single()}")
                
                # Handle affected services
                if fault_service in affected_services:
                    data = affected_services[fault_service]
                    
                    # 1. Create service node for the fault service
                    session.run("""
                        MERGE (s:Service {name: $service_name})
                    """, service_name=fault_service)
                    
                    # 2. Create service nodes for all affected services
                    for service in data['affected_services']:
                        session.run("""
                            MERGE (s:Service {name: $service_name})
                        """, service_name=service)
                        
                        # Create PROPAGATES_TO relationship from fault to service
                        session.run("""
                            MATCH (f:Fault {name: $fault_service})
                            MATCH (s:Service {name: $service_name})
                            MERGE (f)-[:PROPAGATES_TO]->(s)
                        """, fault_service=fault_service, service_name=service)
                        
                        print(f"Created propagation: {fault_service} -> {service}")
                    
                    # 3. Create fault-service AFFECTS edges and service CALLS edges
                    for service in data['affected_services']:
                        session.run("""
                            MATCH (s1:Service {name: $fault_service})
                            MATCH (s2:Service {name: $service_name})
                            MERGE (s1)-[:AFFECTS]->(s2)
                        """, fault_service=fault_service, service_name=service)

                    unique_calls = set()
                    for caller, callee in data['all_calls']:
                        if (caller, callee) not in unique_calls:
                            unique_calls.add((caller, callee))
                            
                            session.run("""
                                MATCH (s1:Service {name: $caller})
                                MATCH (s2:Service {name: $callee})
                                MERGE (s1)-[:CALLS]->(s2)
                            """, caller=caller, callee=callee)
                            
                            print(f"Created call relationship: {caller} -> {callee}")
                    
                    # 4. Link the fault event to the injected service without
                    # overloading the paper's Service-[:AFFECTS]->Service edge.
                    session.run("""
                        MATCH (f:Fault {name: $fault_service})
                        MATCH (s:Service {name: $fault_service})
                        MERGE (f)-[:INJECTED_IN]->(s)
                    """, fault_service=fault_service)
                    
                    # Create summary of propagation
                    affected_count = len(data['affected_services'])
                    call_count = len(unique_calls)
                    
                    print(f"Fault {fault_service} ({fault_type}):")
                    print(f"  - Affects: {affected_count} services")
                    print(f"  - Total calls during fault: {call_count}")

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
        # First, delete existing fault data
        analyzer.delete_fault_related_data()
        
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
                    
                    # Find affected services
                    affected_services = analyzer.find_affected_services(fault_data, trace_data)
                    
                    # Store propagation paths
                    analyzer.store_fault_propagation(fault_data, affected_services)
                    
                    print(f"Successfully processed {records_file}")
                except Exception as e:
                    print(f"Error processing {records_file}: {e}")
                    import traceback
                    traceback.print_exc()
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        analyzer.close()

if __name__ == "__main__":
    main() 
