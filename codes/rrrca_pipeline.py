#!/usr/bin/env python
import argparse
import json
import os
import re
import sys

from neo4j import GraphDatabase

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codes.preprocess.util import Info
from codes.rrrca_slow_path import SlowThinkingCorrector


def parse_probabilities(text):
    probs = {}
    for node, prob in re.findall(r"Node\s+(\d+)\s+fault probability:\s*([0-9.]+)", text or ""):
        probs[int(node)] = float(prob)
    return probs


def load_alpaca_predictions(path):
    with open(path, "r", encoding="utf-8") as fr:
        rows = json.load(fr)

    samples = []
    for idx, row in enumerate(rows):
        probs = parse_probabilities(row.get("output", "") or row.get("original_output", ""))
        if not probs:
            continue
        samples.append(
            {
                "sample_id": idx,
                "input": row.get("input", ""),
                "fast_scores": probs,
                "raw": row,
            }
        )
    return samples


def load_graph_info(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            call_edges = [
                (record["caller"], record["callee"])
                for record in session.run(
                    """
                    MATCH (s1:Service)-[:CALLS]->(s2:Service)
                    RETURN s1.name AS caller, s2.name AS callee
                    """
                )
            ]
            propagation_edges = [
                (record["fault_service"], record["affected_service"])
                for record in session.run(
                    """
                    MATCH (f:Fault)-[:PROPAGATES_TO]->(s:Service)
                    RETURN f.name AS fault_service, s.name AS affected_service
                    """
                )
            ]
            affect_edges = [
                (record["fault_service"], record["affected_service"])
                for record in session.run(
                    """
                    MATCH (s1:Service)-[:AFFECTS]->(s2:Service)
                    RETURN s1.name AS fault_service, s2.name AS affected_service
                    """
                )
            ]
            affect_edges.extend(
                [
                    (record["fault_service"], record["affected_service"])
                    for record in session.run(
                        """
                        MATCH (f:Fault)-[:AFFECTS]->(s:Service)
                        RETURN f.name AS fault_service, s.name AS affected_service
                        """
                    )
                ]
            )
    finally:
        driver.close()

    return {
        "call_edges": call_edges,
        "propagation_edges": propagation_edges,
        "affect_edges": affect_edges,
        "calls": [f"{src} calls {dst}" for src, dst in call_edges],
        "faults": [f"Fault service {src} affects {dst}" for src, dst in propagation_edges],
    }


def run(args):
    info = Info(args.bench)
    samples = load_alpaca_predictions(args.predictions)
    graph_info = load_graph_info(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
    corrector = SlowThinkingCorrector.from_graph_info(
        info.service_names,
        graph_info,
        alpha=args.fast_weight,
        beta=args.consistency_weight,
        gamma=args.rule_weight,
        distance_decay=args.distance_decay,
        max_hops=args.max_hops,
        edge_alpha=args.edge_alpha,
    )

    results = []
    for sample in samples:
        corrected = corrector.correct(sample["fast_scores"])
        observed = max(sample["fast_scores"].items(), key=lambda item: item[1])[0]
        corrected["propagation_path"] = corrector.propagation_path(
            observed,
            corrected["fault_node"],
            fast_scores=sample["fast_scores"],
        )
        results.append(
            {
                "sample_id": sample["sample_id"],
                "input": sample["input"],
                "fast_scores": sample["fast_scores"],
                "corrected_result": corrected,
            }
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fw:
        json.dump(results, fw, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} corrected samples to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Run the RRRCA slow-thinking correction pipeline.")
    parser.add_argument("--predictions", default="../result/llama_factory_dataset/alpaca/dev.json")
    parser.add_argument("--output", default="../result/rag_diagnosis/rrrca_corrected.json")
    parser.add_argument("--bench", default="TrainTicket")
    parser.add_argument("--neo4j_uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j_user", default="neo4j")
    parser.add_argument("--neo4j_password", default="buaanlsde")
    parser.add_argument("--fast_weight", type=float, default=0.6)
    parser.add_argument("--consistency_weight", type=float, default=0.25)
    parser.add_argument("--rule_weight", type=float, default=0.15)
    parser.add_argument("--distance_decay", type=float, default=0.6)
    parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--edge_alpha", type=float, default=0.5)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
