"""
closed_form_solver.py — 闭式解数据分配算法（核心贡献）

目标：令所有节点的 (传输时间 + 计算时间) 相等 → 同时完成 → makespan 最小。

模型：  T_i = L_i + d_i * k_i ,   k_i = s/r_i + 1/c_i
求解：  令所有 T_i = T 且 sum(d_i) = D
        =>  T   = ( D + sum(L_i/k_i) ) / sum(1/k_i)
        =>  d_i = (T - L_i) / k_i

用法：
    python closed_form_solver.py <c1> <c2> <c3> <L1> <L2> <L3> <D>
例（本研究 P2）：
    python closed_form_solver.py 936.0 1073.2 1719.9 0.985 4.470 1.138 50000
"""
import sys

# ---- 常量：CIFAR-10 单图 32x32x3 ≈ 3KB；带宽三节点相同 ----
SIZE_MB = 0.003      # s : 每样本数据量 (MB)
RATE    = 111.0      # r : 数据速率 (MB/s)，三节点相同以保证公平
NAMES   = ["RTX4070", "RTX3080", "RTX5080"]


def closed_form_allocate(c, L, D, size_mb=SIZE_MB, rate=RATE):
    """
    c : list[float]  各节点吞吐 (samples/s)
    L : list[float]  各节点固定延迟 (s)，含通信 latency 与任何与数据量无关的开销
    D : int          总样本数
    返回 (d, T)：各节点分配的样本数，以及共同完成时间
    """
    n = len(c)
    # k_i：在节点 i 上每一条数据的代价（传输 + 计算）
    k = [size_mb / rate + 1.0 / c[i] for i in range(n)]

    R = sum(1.0 / k[i] for i in range(n))        # 集群总处理速率 (samples/s)
    W = sum(L[i] / k[i] for i in range(n))       # 延迟折算成的等价工作量 (samples)
    T = (D + W) / R                              # 共同完成时间 (s)

    d = [(T - L[i]) / k[i] for i in range(n)]    # 倒推各节点分配量

    # 健壮性：若某节点 d_i < 0，说明其固定延迟过大、不值得分配数据，剔除后重解
    if any(di < 0 for di in d):
        keep = [i for i, di in enumerate(d) if di >= 0]
        sub_d, T = closed_form_allocate([c[i] for i in keep],
                                        [L[i] for i in keep], D, size_mb, rate)
        d = [0.0] * n
        for idx, i in enumerate(keep):
            d[i] = sub_d[idx]
    return d, T


def node_time(c_i, L_i, d_i, size_mb=SIZE_MB, rate=RATE):
    """单节点总时间 = 固定延迟 + 传输 + 计算"""
    return L_i + d_i * size_mb / rate + d_i / c_i


if __name__ == "__main__":
    c = [float(x) for x in sys.argv[1:4]]
    L = [float(x) for x in sys.argv[4:7]]
    D = int(sys.argv[7])

    d, T = closed_form_allocate(c, L, D)
    for i in range(len(NAMES)):
        print(f"  {NAMES[i]}: {d[i]/D*100:5.2f}%  ({d[i]:.0f} imgs)  "
              f"c={c[i]:.1f}  L={L[i]:.3f}s  T_i={node_time(c[i], L[i], d[i]):.2f}s")
    print(f"Predicted makespan = {T:.2f}s")
    print("FRACTIONS " + " ".join(f"{di/D:.4f}" for di in d))
