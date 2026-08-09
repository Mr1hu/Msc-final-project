"""
refine_loop.py — 迭代精化循环

问题：要算分配需要知道吞吐，但真实吞吐又取决于分到多少数据 —— 互相依赖。
解法：迭代。每轮按当前分配跑一次训练、就地测出真实吞吐、用闭式解重解，
      直到分配比例不再变化。

关键使能条件（导师指出）：一次训练迭代的耗时与该节点分到的总数据量无关，
因此不必等完整训练结束，跑一轮即可测得真实吞吐，每轮精化代价极低。

本文件提供两种 measure 后端：
  simulated — 用负载惩罚模型模拟吞吐随分配量变化，用于验证循环逻辑
  from_log  — 从真实训练日志中按 (图片数 / 计算时间) 读出吞吐（本研究实际采用）
"""
from closed_form_solver import closed_form_allocate

TOTAL = 50000
NAMES = ["RTX4070", "RTX3080", "RTX5080"]
LAT   = [0.985, 4.470, 1.138]        # 固定延迟（含通信 latency + 聚合开销）
TOL   = 0.001                        # 收敛阈值：分配比例最大变化 < 0.1%


# ---------------- 后端一：模拟测量 ----------------
IDEAL_C = [941.5, 1078.0, 1716.7]


def measure_simulated(i, d_i, total=TOTAL):
    """模拟：分到的数据越多，内存/IO 压力越大，吞吐越低"""
    load = d_i / total
    return IDEAL_C[i] * (1 - 0.30 * load)


# ---------------- 后端二：从真实日志读取 ----------------
def measure_from_log(lines):
    """
    lines: 各节点训练输出，格式 'name|n_images|comm|compute|total'
    返回 list[float]：各节点在真实负载下的吞吐
    """
    out = []
    for ln in lines:
        parts = ln.strip().split("|")
        n, compute = float(parts[1]), float(parts[3])
        out.append(n / compute)
    return out


# ---------------- 精化循环 ----------------
def refine(c0, lat=LAT, total=TOTAL, measure=measure_simulated,
           tol=TOL, max_rounds=8, verbose=True):
    d, T = closed_form_allocate(c0, lat, total)
    if verbose:
        print(f"{'Round':<8}" + "".join(f"{n:>11}" for n in NAMES) + f"{'max chg':>10}")
        print("-" * (8 + 11 * len(NAMES) + 10))
        print(f"{'init':<8}" + "".join(f"{di/total*100:>10.2f}%" for di in d) + f"{'-':>10}")

    for r in range(1, max_rounds + 1):
        c_new = [measure(i, d[i]) for i in range(len(d))]     # 就地测量真实吞吐
        d_new, T = closed_form_allocate(c_new, lat, total)    # 用新参数重解
        delta = max(abs(d_new[i] - d[i]) / total for i in range(len(d)))
        if verbose:
            print(f"{'r'+str(r):<8}" + "".join(f"{di/total*100:>10.2f}%" for di in d_new)
                  + f"{delta*100:>9.2f}%")
        d = d_new
        if delta < tol:
            if verbose:
                print(f"\n>>> Converged at round {r} (change < {tol*100:.1f}%)")
            break
    return d, T


if __name__ == "__main__":
    d, T = refine([736.0, 741.8, 1069.0])     # 从小样本画像的估计出发
    print(f"\nFinal split: " + "  ".join(
        f"{NAMES[i]}={d[i]/TOTAL*100:.2f}%" for i in range(len(d))))
    print(f"Predicted makespan = {T:.2f}s")
