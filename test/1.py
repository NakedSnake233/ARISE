# ACM: 决策树(信息增益、二叉特征) —— 满足样例
import sys, math

def entropy(cnt0, cnt1):
    n = cnt0 + cnt1
    if n == 0: return 0.0
    e = 0.0
    if cnt0:
        p = cnt0 / n
        e -= p * math.log2(p)
    if cnt1:
        p = cnt1 / n
        e -= p * math.log2(p)
    return e

def majority_label(idxs, y):
    if not idxs: return 0
    c1 = sum(y[i] for i in idxs)
    c0 = len(idxs) - c1
    return 1 if c1 > c0 else 0  # tie -> 0

def info_gain(f, idxs, X, y):
    # 计算按特征f划分的信息增益，并返回左右子集
    cnt0 = sum(1 for i in idxs if y[i] == 0)
    cnt1 = len(idxs) - cnt0
    H_before = entropy(cnt0, cnt1)

    L = [i for i in idxs if X[i][f] == 0]
    R = [i for i in idxs if X[i][f] == 1]

    l0 = sum(1 for i in L if y[i] == 0); l1 = len(L) - l0
    r0 = sum(1 for i in R if y[i] == 0); r1 = len(R) - r0

    n = len(idxs)
    H_after = (len(L)/n) * entropy(l0, l1) + (len(R)/n) * entropy(r0, r1)
    return H_before - H_after, L, R

class Node:
    __slots__ = ("leaf","label","feat","left","right")
    def __init__(self): self.leaf=True; self.label=0; self.feat=-1; self.left=None; self.right=None

def build(idxs, feats_avail, X, y, parent_major):
    node = Node()
    if not idxs:             # 空分支：用父节点多数类
        node.label = parent_major
        return node

    # 纯叶
    first = y[idxs[0]]
    if all(y[i] == first for i in idxs):
        node.label = first
        return node

    if not feats_avail:
        node.label = majority_label(idxs, y)
        return node

    # 选信息增益最大的特征；并列取索引小的
    best_gain, best_f, best_L, best_R = -1.0, -1, None, None
    for f in feats_avail:
        g, L, R = info_gain(f, idxs, X, y)
        if g > best_gain or (abs(g - best_gain) <= 1e-12 and f < best_f):
            best_gain, best_f, best_L, best_R = g, f, L, R

    if best_gain <= 0:
        node.label = majority_label(idxs, y)
        return node

    node.leaf = False
    node.feat = best_f
    next_feats = [f for f in feats_avail if f != best_f]
    maj = majority_label(idxs, y)
    node.left  = build(best_L, next_feats, X, y, maj)
    node.right = build(best_R, next_feats, X, y, maj)
    return node

def predict(root, x):
    cur = root
    while not cur.leaf:
        cur = cur.right if x[cur.feat] == 1 else cur.left
    return cur.label

def main():
    data = sys.stdin.read().strip().split()
    it = iter(data)
    n = int(next(it)); m = int(next(it))
    X, y = [], []
    for _ in range(n):
        row = [int(next(it)) for _ in range(m+1)]
        X.append(row[:m]); y.append(row[-1])
    q = int(next(it))
    Q = [[int(next(it)) for _ in range(m)] for _ in range(q)]

    root = build(list(range(n)), list(range(m)), X, y, parent_major=majority_label(range(n), y))
    out = [str(predict(root, x)) for x in Q]
    sys.stdout.write("\n".join(out))

if __name__ == "__main__":
    main()
