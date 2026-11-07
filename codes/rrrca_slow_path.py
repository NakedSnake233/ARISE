import heapq
import math
from collections import defaultdict, deque


class SlowThinkingCorrector:
    """RRRCA slow-thinking correction over service and propagation graphs.

    The implementation follows the paper-level scoring:
    S_final = alpha * S_fast + beta * S_consistency + gamma * S_rule
    and uses -log(edge_score) shortest paths for propagation explanations.
    """

    def __init__(
        self,
        service_names,
        calls=None,
        propagations=None,
        affects=None,
        alpha=0.6,
        beta=0.25,
        gamma=0.15,
        distance_decay=0.6,
        max_hops=3,
        edge_alpha=0.5,
        anomaly_threshold=None,
    ):
        total = alpha + beta + gamma
        if total <= 0:
            raise ValueError("alpha + beta + gamma must be positive")

        self.service_names = list(service_names)
        self.service2index = {service: idx for idx, service in enumerate(self.service_names)}
        self.index2service = {idx: service for service, idx in self.service2index.items()}
        self.alpha = alpha / total
        self.beta = beta / total
        self.gamma = gamma / total
        self.distance_decay = distance_decay
        self.max_hops = max_hops
        self.edge_alpha = edge_alpha
        self.anomaly_threshold = anomaly_threshold

        self.calls = set(self._normalize_edges(calls or []))
        self.propagations = set(self._normalize_edges(propagations or []))
        self.affects = set(self._normalize_edges(affects or []))
        self.edges = set(self.calls) | set(self.propagations) | set(self.affects)
        self.outgoing = defaultdict(set)
        self.incoming = defaultdict(set)
        for src, dst in self.edges:
            self.outgoing[src].add(dst)
            self.incoming[dst].add(src)

    def _normalize_service(self, value):
        if isinstance(value, int):
            return self.index2service.get(value)
        if value in self.service2index:
            return value
        if isinstance(value, str):
            parts = value.split("-")
            if parts and parts[-1].isdigit():
                candidate = "-".join(parts[:-1])
                if candidate in self.service2index:
                    return candidate
        return None

    def _normalize_edges(self, edges):
        normalized = []
        for edge in edges:
            if isinstance(edge, dict):
                src = edge.get("src") or edge.get("source") or edge.get("caller") or edge.get("fault_service")
                dst = edge.get("dst") or edge.get("target") or edge.get("callee") or edge.get("affected_service")
            else:
                src, dst = edge
            src = self._normalize_service(src)
            dst = self._normalize_service(dst)
            if src is not None and dst is not None:
                normalized.append((src, dst))
        return normalized

    @classmethod
    def from_graph_info(
        cls,
        service_names,
        graph_info,
        alpha=0.6,
        beta=0.25,
        gamma=0.15,
        distance_decay=0.6,
        max_hops=3,
        edge_alpha=0.5,
        anomaly_threshold=None,
    ):
        calls = graph_info.get("call_edges", [])
        propagations = graph_info.get("propagation_edges", [])
        affects = graph_info.get("affect_edges", [])

        if not calls:
            calls = cls._parse_call_texts(graph_info.get("calls", []))
        if not propagations and not affects:
            propagations = cls._parse_fault_texts(graph_info.get("faults", []))

        return cls(
            service_names,
            calls=calls,
            propagations=propagations,
            affects=affects,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            distance_decay=distance_decay,
            max_hops=max_hops,
            edge_alpha=edge_alpha,
            anomaly_threshold=anomaly_threshold,
        )

    @staticmethod
    def _parse_call_texts(lines):
        edges = []
        for line in lines:
            if " calls " not in line:
                continue
            src, dst = line.split(" calls ", 1)
            edges.append((src.strip(), dst.strip()))
        return edges

    @staticmethod
    def _parse_fault_texts(lines):
        edges = []
        for line in lines:
            if "Fault service " not in line or " affects " not in line:
                continue
            src = line.split("Fault service ", 1)[1].split(" (", 1)[0].strip()
            dst = line.rsplit(" affects ", 1)[-1].strip()
            edges.append((src, dst))
        return edges

    def _normalize_scores(self, scores):
        if isinstance(scores, dict):
            raw = {
                self._normalize_service(node): float(score)
                for node, score in scores.items()
            }
        else:
            raw = {
                self.index2service[idx]: float(score)
                for idx, score in enumerate(scores)
                if idx in self.index2service
            }
        raw = {service: max(0.0, score) for service, score in raw.items() if service is not None}
        for service in self.service_names:
            raw.setdefault(service, 0.0)
        total = sum(raw.values())
        if total <= 0:
            return {service: 1.0 / len(self.service_names) for service in self.service_names}
        return {service: score / total for service, score in raw.items()}

    def _anomaly_nodes(self, scores):
        if self.anomaly_threshold is not None:
            threshold = self.anomaly_threshold
        else:
            mean = sum(scores.values()) / len(scores)
            var = sum((score - mean) ** 2 for score in scores.values()) / len(scores)
            threshold = max(mean, mean + math.sqrt(var) * 0.5)
        nodes = {service for service, score in scores.items() if score >= threshold}
        if not nodes:
            nodes = {max(scores.items(), key=lambda item: item[1])[0]}
        return nodes

    def _shortest_distances(self, start):
        distances = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if distances[node] >= self.max_hops:
                continue
            neighbours = self.outgoing.get(node, set()) | self.incoming.get(node, set())
            for nxt in neighbours:
                if nxt not in distances:
                    distances[nxt] = distances[node] + 1
                    queue.append(nxt)
        return distances

    def _structural_consistency(self, scores):
        consistency = {}
        for service in self.service_names:
            distances = self._shortest_distances(service)
            value = 0.0
            norm = 0.0
            for neighbour, distance in distances.items():
                weight = self.distance_decay ** distance
                value += weight * scores.get(neighbour, 0.0)
                norm += weight
            consistency[service] = value / norm if norm > 0 else 0.0
        return self._minmax(consistency)

    def _minmax(self, values):
        lo = min(values.values()) if values else 0.0
        hi = max(values.values()) if values else 0.0
        if hi - lo <= 1e-12:
            return {key: 0.0 for key in values}
        return {key: (value - lo) / (hi - lo) for key, value in values.items()}

    def _rule_scores(self, scores, anomaly_nodes):
        max_out = max((len(self.outgoing.get(service, set())) for service in self.service_names), default=0)
        rule_scores = {}
        for service in self.service_names:
            upstream = self.incoming.get(service, set())
            upstream_healthy = bool(upstream) and all(src not in anomaly_nodes for src in upstream)
            if not upstream:
                upstream_healthy = True

            downstream = self.outgoing.get(service, set())
            affected_downstream = sum(1 for dst in downstream if dst in anomaly_nodes)
            fanout_ratio = len(downstream) / max_out if max_out > 0 else 0.0
            fanout_rule = fanout_ratio >= 0.5 or affected_downstream >= 2

            neighbourhood = downstream | upstream | {service}
            temporal_alignment = any(node in anomaly_nodes for node in neighbourhood)

            rule_scores[service] = (
                int(upstream_healthy) + int(fanout_rule) + int(temporal_alignment)
            ) / 3.0
        return rule_scores

    def correct(self, fast_scores, llm_scores=None):
        fast = self._normalize_scores(fast_scores)
        llm = self._normalize_scores(llm_scores) if llm_scores else {}
        anomaly_nodes = self._anomaly_nodes(fast)
        consistency = self._structural_consistency(llm or fast)
        rules = self._rule_scores(fast, anomaly_nodes)

        final_scores = {}
        explanations = {}
        for service in self.service_names:
            base_score = llm.get(service, fast[service])
            fast_component = max(fast[service], base_score)
            final_scores[service] = (
                self.alpha * fast_component
                + self.beta * consistency.get(service, 0.0)
                + self.gamma * rules.get(service, 0.0)
            )
            explanations[service] = {
                "fast_score": fast[service],
                "llm_score": llm.get(service),
                "structural_consistency": consistency.get(service, 0.0),
                "rule_score": rules.get(service, 0.0),
            }

        total = sum(final_scores.values())
        if total > 0:
            final_scores = {service: score / total for service, score in final_scores.items()}

        ranked_services = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)
        return {
            "fault_node": self.service2index[ranked_services[0][0]],
            "fault_service": ranked_services[0][0],
            "fault_probability": ranked_services[0][1],
            "all_probabilities": {
                self.service2index[service]: score for service, score in final_scores.items()
            },
            "ranked_services": ranked_services,
            "anomaly_services": sorted(anomaly_nodes),
            "explanations": explanations,
        }

    def _edge_base_score(self, src, dst, attention=None, llm_edge_scores=None):
        structural = 0.1
        if (src, dst) in self.calls:
            structural = max(structural, 0.5)
        if (src, dst) in self.affects:
            structural = max(structural, 0.7)
        if (src, dst) in self.propagations:
            structural = max(structural, 0.85)
        if attention and (src, dst) in attention:
            structural = max(structural, float(attention[(src, dst)]))

        semantic = structural
        if llm_edge_scores and (src, dst) in llm_edge_scores:
            semantic = float(llm_edge_scores[(src, dst)])

        return self.edge_alpha * structural + (1.0 - self.edge_alpha) * semantic

    def _edge_rule_factor(self, src, dst, scores):
        dst_health = 1.0 - scores.get(dst, 0.0)
        out_degree = len(self.outgoing.get(dst, set()))
        fanout = out_degree / (out_degree + 1.0)
        target_anomaly = 1.0 if dst in self._anomaly_nodes(scores) else 0.0
        return max(0.05, target_anomaly * (1.0 - dst_health) * max(fanout, 0.1))

    def propagation_path(self, observed_node, root_node, fast_scores=None, attention=None, llm_edge_scores=None, reverse=True):
        start = self._normalize_service(observed_node)
        end = self._normalize_service(root_node)
        if start is None or end is None:
            return {"path": [], "score": 0.0, "edge_scores": []}
        if start == end:
            return {"path": [start], "score": 1.0, "edge_scores": []}

        scores = self._normalize_scores(fast_scores or {})
        pq = [(0.0, start, [])]
        visited = {}

        while pq:
            cost, node, path_edges = heapq.heappop(pq)
            if node in visited and visited[node] <= cost:
                continue
            visited[node] = cost
            if node == end:
                path = [start] + [dst for _, dst, _ in path_edges]
                edge_scores = [
                    {"src": src, "dst": dst, "score": score}
                    for src, dst, score in path_edges
                ]
                path_score = 1.0
                for edge in edge_scores:
                    path_score *= edge["score"]
                return {"path": path, "score": path_score, "edge_scores": edge_scores}

            neighbours = self.incoming.get(node, set()) if reverse else self.outgoing.get(node, set())
            for nxt in neighbours:
                src, dst = (nxt, node) if reverse else (node, nxt)
                base = self._edge_base_score(src, dst, attention=attention, llm_edge_scores=llm_edge_scores)
                rule_factor = self._edge_rule_factor(src, dst, scores)
                edge_score = max(1e-6, min(1.0, base * rule_factor))
                edge_cost = -math.log(edge_score)
                heapq.heappush(pq, (cost + edge_cost, nxt, path_edges + [(node, nxt, edge_score)]))

        return {"path": [], "score": 0.0, "edge_scores": []}
