"""
stats3.py — 统计重复实验结果，给出 makespan 的均值与误差范围

读取 repeat3.sh 产生的 $SHARE/repeat3_raw.txt
每行格式：strategy|rep|node1_line;node2_line;node3_line
"""
import os
import statistics

path = os.path.join(os.environ.get("SHARE", os.path.expanduser("~/dist_work")),
                    "repeat3_raw.txt")

rows = {}
for line in open(path):
    label, rep, payload = line.strip().split("|", 2)
    times = [float(r.split("|")[4]) for r in payload.split(";") if "|" in r]
    rows.setdefault(label, []).append((max(times), max(times) - min(times)))

print(f"{'Strategy':<14}{'Makespan (mean±range)':>26}{'Spread (mean)':>16}")
print("-" * 56)
for label, vals in rows.items():
    mk = [a for a, _ in vals]
    sp = [b for _, b in vals]
    half_range = (max(mk) - min(mk)) / 2
    print(f"{label:<14}{statistics.mean(mk):>14.2f}s ±{half_range:>5.2f}"
          f"{statistics.mean(sp):>14.2f}s")
