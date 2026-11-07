"""Exact default Drain matching via inverted token bitsets.

SS creates tens of thousands of templates in a common prefix leaf. This keeps
the original similarity/parameter-count/first-candidate ordering, while adding
matching counts in parallel with Python integer bit operations. No parser
threshold, message, template-update or training-order rule changes.
"""
from __future__ import annotations
import collections
from contextlib import contextmanager
import inspect
import textwrap
import weakref


class WidthIndex:
    def __init__(self):
        self.rows = {}; self.ids = []; self.templates = {}; self.literal = {}
        self.wildcards = []; self.parameters = []; self.leaves = collections.OrderedDict()

    def register(self, cluster, param):
        cid = cluster.cluster_id; tokens = tuple(cluster.log_template_tokens)
        if cid not in self.rows:
            self.rows[cid] = len(self.ids); self.ids.append(cid)
        row = self.rows[cid]; bit = 1 << row; old = self.templates.get(cid)
        if old == tokens: return
        while len(self.wildcards) < len(tokens): self.wildcards.append(0)
        for pos, token in enumerate(old or ()):
            if token == param: self.wildcards[pos] &= ~bit
            else:
                key = (pos, token); updated = self.literal_bits(key) & ~bit
                if updated: self.store_literal(key, updated)
                else: del self.literal[key]
        for pos, token in enumerate(tokens):
            if token == param: self.wildcards[pos] |= bit
            else:
                key = (pos, token); self.store_literal(key, self.literal_bits(key) | bit)
        count = tokens.count(param)
        while len(self.parameters) < count.bit_length(): self.parameters.append(0)
        for level in range(len(self.parameters)):
            self.parameters[level] &= ~bit
            if count & (1 << level): self.parameters[level] |= bit
        self.templates[cid] = tokens

    def literal_bits(self, key):
        entry = self.literal.get(key)
        return entry[1] << entry[0] if entry else 0

    def store_literal(self, key, bits):
        # Trim leading zero positions: singleton rare tokens remain tiny even
        # in a corpus with hundreds of thousands of clusters. Logical bitsets
        # and candidate ordering are unchanged.
        offset = (bits & -bits).bit_length()-1
        self.literal[key] = (offset, bits >> offset)

    def leaf(self, ids):
        key = id(ids); entry = self.leaves.get(key)
        if entry is None or entry[0] is not ids or len(ids) < entry[1]:
            entry = [ids, 0, 0, True, -1]
        for cid in ids[entry[1]:]:
            if cid not in self.rows: continue
            row = self.rows[cid]; entry[2] |= 1 << row
            entry[3] &= row > entry[4]; entry[4] = row
        entry[1] = len(ids); self.leaves[key] = entry; self.leaves.move_to_end(key)
        if len(self.leaves) > 64: self.leaves.popitem(last=False)
        return entry[2], entry[3]


def maximum(mask, planes):
    count = 0
    for level in range(len(planes)-1, -1, -1):
        positive = mask & planes[level]
        if positive: mask = positive; count |= 1 << level
    return mask, count


@contextmanager
def accelerated():
    from drain3.drain import Drain
    original_match = Drain.fast_match; original_add = Drain.add_log_message
    original_prefix = Drain.add_seq_to_prefix_tree; states = weakref.WeakKeyDictionary()

    def index(self):
        entry = states.get(self)
        if entry is None or entry[0] is not self.id_to_cluster:
            groups = {}
            for cluster in self.id_to_cluster.values():
                groups.setdefault(len(cluster.log_template_tokens), WidthIndex()).register(cluster, self.param_str)
            entry = (self.id_to_cluster, groups); states[self] = entry
        return entry[1]

    def fast_match(self, cluster_ids, tokens, sim_th, include_params):
        if type(self.id_to_cluster) is not dict or self.max_clusters is not None:
            return original_match(self, cluster_ids, tokens, sim_th, include_params)
        groups = index(self); current = groups.get(len(tokens))
        if current is None: return None
        mask, ascending = current.leaf(cluster_ids)
        if not mask: return None
        planes = []
        for pos, token in enumerate(tokens):
            carry = current.literal_bits((pos, token))
            if include_params: carry |= current.wildcards[pos]
            level = 0
            while carry:
                if len(planes) == level: planes.append(0)
                previous = planes[level]; planes[level] ^= carry; carry &= previous; level += 1
        best, count = maximum(mask, planes)
        similarity = float(count) / len(tokens) if tokens else 1.
        if similarity < sim_th: return None
        best, _ = maximum(best, current.parameters)
        if ascending:
            row = (best & -best).bit_length()-1
            return self.id_to_cluster[current.ids[row]]
        for cid in cluster_ids:
            row = current.rows.get(cid)
            if row is not None and best & (1 << row): return self.id_to_cluster[cid]
        return None

    def add(self, content):
        result = original_add(self, content)
        if type(self.id_to_cluster) is dict and self.max_clusters is None:
            cluster, change = result
            if change != "none":
                index(self).setdefault(len(cluster.log_template_tokens), WidthIndex()).register(cluster, self.param_str)
        return result

    # The default unlimited dictionary never evicts a cluster. Removing its
    # repeated leaf cleanup scan preserves the exact candidate IDs/order and
    # keeps the list stable for an incrementally maintained membership bitset.
    source = textwrap.dedent(inspect.getsource(original_prefix))
    cleanup = """            # clean up stale clusters before adding a new one.
            new_cluster_ids = []
            for cluster_id in cur_node.cluster_ids:
                if cluster_id in self.id_to_cluster:
                    new_cluster_ids.append(cluster_id)
            new_cluster_ids.append(cluster.cluster_id)
            cur_node.cluster_ids = new_cluster_ids"""
    if source.count(cleanup) != 1: raise ValueError("installed Drain prefix code differs from witnessed version")
    namespace = dict(original_prefix.__globals__)
    exec(compile(source.replace(cleanup, "            cur_node.cluster_ids.append(cluster.cluster_id)"), "<exact unlimited Drain prefix>", "exec"), namespace)
    append_prefix = namespace["add_seq_to_prefix_tree"]

    def prefix(self, root, cluster):
        return (append_prefix if type(self.id_to_cluster) is dict and self.max_clusters is None else original_prefix)(self, root, cluster)

    Drain.fast_match = fast_match; Drain.add_log_message = add; Drain.add_seq_to_prefix_tree = prefix
    try: yield
    finally:
        Drain.fast_match = original_match; Drain.add_log_message = original_add; Drain.add_seq_to_prefix_tree = original_prefix
