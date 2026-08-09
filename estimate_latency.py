"""
estimate_latency.py — 从测量中学习隐藏的 latency

导师设定：T_latency 的真实值只有实验设计者知道，算法不可见，必须自己学出来。

原理：T_comm = L + size/rate 对 size 是线性的。
      传输若干种不同大小的数据，对 (size, time) 观测点做最小二乘拟合：
          截距     = L        （size = 0 时的固定耗时）
          斜率倒数 = rate     （带宽）

本文件包含两种模式：
  simulate  — 注入已知真值 + 噪声，验证估计方法本身的正确性
  real      — 用真实文件传输测量（需要 scp 可达的目标主机）
"""
import random
import subprocess
import time
import os

SIZES_MB = [5, 10, 20, 50, 100]   # 探测用的数据大小
REPEATS  = 3                       # 每种大小重复次数


# ---------------- 最小二乘直线拟合 ----------------
def fit_line(points):
    """拟合 t = a + b*s，返回 (a=截距, b=斜率)"""
    n   = len(points)
    sx  = sum(p[0] for p in points)
    sy  = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    a = (sy - b * sx) / n
    return a, b


def estimate(points):
    """由观测点估计 (latency 秒, 带宽 MB/s)"""
    a, b = fit_line(points)
    return max(0.0, a), 1.0 / b


# ---------------- 模式一：模拟验证 ----------------
TRUE_PARAMS = {                       # 注入的真值，估计环节不可见
    "Node1": {"latency_ms": 0.0,   "bw": 111.0},
    "Node2": {"latency_ms": 50.0,  "bw": 111.0},
    "Node3": {"latency_ms": 150.0, "bw": 111.0},
}


def simulate_transfer(latency_ms, bw, size_mb, noise=0.02):
    t = latency_ms / 1000 + size_mb / bw
    return t * (1 + random.uniform(-noise, noise))


def run_simulation():
    print(f"{'Node':<8}{'true L':>10}{'est L':>10}{'err':>9}"
          f"{'true bw':>10}{'est bw':>10}")
    print("-" * 57)
    for name, p in TRUE_PARAMS.items():
        pts = [(s, simulate_transfer(p["latency_ms"], p["bw"], s))
               for s in SIZES_MB for _ in range(REPEATS)]
        L, bw = estimate(pts)
        print(f"{name:<8}{p['latency_ms']:>8.1f}ms{L*1000:>8.1f}ms"
              f"{abs(L*1000 - p['latency_ms']):>7.1f}ms"
              f"{p['bw']:>10.1f}{bw:>10.1f}")


# ---------------- 模式二：真实传输测量 ----------------
def real_transfer(host, path, remote_dir, inject_ms=0.0):
    """真实 scp 计时 + 注入隐藏延迟"""
    t0 = time.time()
    subprocess.run(["scp", "-q", path, f"{host}:{remote_dir}/probe_recv"], check=True)
    elapsed = time.time() - t0
    if inject_ms > 0:
        time.sleep(inject_ms / 1000)
    return elapsed + inject_ms / 1000


def make_probe(size_mb, tmp="/tmp"):
    path = f"{tmp}/probe_{size_mb}mb"
    if not os.path.exists(path):
        subprocess.run(f"dd if=/dev/zero of={path} bs=1M count={size_mb}",
                       shell=True, capture_output=True)
    return path


def run_real(workers, remote_dir="/local/data/txfz0982"):
    """workers: dict  name -> {"host": "user@ip", "inject_ms": float}"""
    print(f"{'Node':<10}{'inject':>10}{'est L':>10}{'est bw':>12}")
    print("-" * 42)
    for name, w in workers.items():
        pts = []
        for s in SIZES_MB[:4]:
            f = make_probe(s)
            for _ in range(2):
                pts.append((s, real_transfer(w["host"], f, remote_dir, w["inject_ms"])))
        L, bw = estimate(pts)
        print(f"{name:<10}{w['inject_ms']:>8.1f}ms{L*1000:>8.1f}ms{bw:>10.1f}MB/s")


if __name__ == "__main__":
    run_simulation()
    # 真实测量示例（按需取消注释并填入自己的主机）：
    # run_real({"RTX4070": {"host": "user@129.11.146.207", "inject_ms": 50.0},
    #           "RTX5080": {"host": "user@129.11.144.252", "inject_ms": 150.0}})
