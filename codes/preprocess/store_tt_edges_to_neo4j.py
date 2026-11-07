from neo4j import GraphDatabase
import os

class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def delete_fault_related_data(self):
        with self.driver.session() as session:
            # Delete all fault-related nodes and relationships
            session.run("""
                MATCH (n:Fault)
                DETACH DELETE n
            """)
            print("Successfully deleted fault-related nodes and relationships")

    def create_service_relationships(self):
        with self.driver.session() as session:
            # Define the service call relationships
            edge_info = {
                "preserve": ["preserve", "seat", "security", "food", "order", "ticketinfo", "travel", "contacts", "notification", "user", "station"],
                "seat": ["seat", "order", "config", "travel"],   
                "cancel": ["inside-payment", "order-other", "order"],
                "security": ["security", "order-other", "order"],
                "food": ["travel", "food-map", "station"],
                "travel": ["travel", "order", "ticketinfo", "train", "route"],
                "inside-payment": ["payment", "order"],
                "ticketinfo": ["ticketinfo", "basic"],
                "basic": ["basic", "route", "price", "train", "station"],
                "order-other": ["station"],
                "order": ["order", "station", "assurance"],
                "auth": ["auth", "verification-code"]
            }

            # Create constraints
            session.run("CREATE CONSTRAINT service_name IF NOT EXISTS FOR (s:Service) REQUIRE s.name IS UNIQUE")

            # Create services and relationships
            for source, targets in edge_info.items():
                source_service = f"ts-{source}-service"
                # Create source service node if not exists
                session.run(
                    "MERGE (s:Service {name: $source_name})",
                    source_name=source_service
                )
                
                # Create target services and relationships
                for target in targets:
                    target_service = f"ts-{target}-service"
                    session.run(
                        "MERGE (t:Service {name: $target_name})",
                        target_name=target_service
                    )
                    session.run(
                        "MATCH (s:Service {name: $source_name}), (t:Service {name: $target_name}) "
                        "MERGE (s)-[:CALLS]->(t)",
                        source_name=source_service,
                        target_name=target_service
                    )

def main():
    # Neo4j connection details
    uri = "bolt://localhost:7687"  # Default Neo4j bolt port
    user = "neo4j"
    password = __import__('os').environ['NEO4J_PASSWORD']  # Updated password

    # Create connection and store relationships
    conn = Neo4jConnection(uri, user, password)
    try:
        # First delete fault-related data
        conn.delete_fault_related_data()
        # Then create service relationships
        conn.create_service_relationships()
        print("Successfully stored TT service call relationships in Neo4j")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main() 