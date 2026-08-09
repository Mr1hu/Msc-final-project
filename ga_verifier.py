"""
ga_verifier.py — 遗传算法独立验证

在完全不知道闭式解答案的情况下，用进化搜索寻找最优分配。
若收敛到与闭式解相同的结果，则证明闭式解为全局最优。

与 closed_form_solver.py 使用完全相同的问题设定（同一个 makespan 定义）。
"""
import random

# ---- 问题设定：必须与闭式解完全一致 ----
NODES = [
    # (名称, 吞吐 samples/s, 带宽 MB/s, 固定延迟 s)
    ("RTX4070",  936.0, 111.0, 0.985),
    ("RTX3080", 1073.2, 111.0, 4.470),
    ("RTX5080", 1719.9, 111.0, 1.138),
]
TOTAL   = 50000
SIZE_MB = 0.003

# ---- GA 超参数 ----
POP_SIZE      = 50
GENERATIONS   = 100
MUTATION_RATE = 0.10
SEED          = 42


def makespan(ratios):
    """适应度函数：给定分配比例，返回总完成时间（最慢节点）。越小越好。"""
    times = []
    for (_, c, r, L), frac in zip(NODES, ratios):
        d = TOTAL * frac
        times.append(L + d * SIZE_MB / r + d / c)
    return max(times)


def normalise(v):
    """归一化，保证比例之和为 1"""
    s = sum(v)
    return [x / s for x in v]


def genetic_search(pop_size=POP_SIZE, generations=GENERATIONS,
                   mutation=MUTATION_RATE, seed=SEED, trace=False):
    random.seed(seed)
    n = len(NODES)

    # 初始种群：随机分配方案
    pop = [normalise([random.random() for _ in range(n)]) for _ in range(pop_size)]
    history = []

    for g in range(generations):
        pop.sort(key=makespan)                       # 按适应度排序（小=好）
        history.append(makespan(pop[0]))
        survivors = pop[: pop_size // 2]             # 择优保留前 50%

        children = []
        while len(children) < pop_size - len(survivors):
            p1, p2 = random.sample(survivors, 2)
            w = random.random()
            child = normalise([w * a + (1 - w) * b for a, b in zip(p1, p2)])   # 加权平均交叉
            if random.random() < mutation:                                      # 变异
                j = random.randrange(n)
                child[j] *= random.uniform(0.8, 1.2)
                child = normalise(child)
            children.append(child)

        pop = survivors + children

    best = min(pop, key=makespan)
    return (best, history) if trace else best


if __name__ == "__main__":
    best, hist = genetic_search(trace=True)
    print("GA convergence (every 20 generations):")
    for g in range(0, GENERATIONS, 20):
        print(f"  gen {g:3d}: makespan = {hist[g]:.2f}s")
    print(f"  gen {GENERATIONS:3d}: makespan = {makespan(best):.2f}s")
    print("\nGA allocation:")
    for (name, *_), frac in zip(NODES, best):
        print(f"  {name}: {frac*100:5.2f}%")
    print(f"  makespan = {makespan(best):.2f}s")
