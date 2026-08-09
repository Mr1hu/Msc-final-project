# 代码包说明

分布式深度学习中的数据分配与通信感知优化 — 实验代码

## 目录结构

```
core/       算法本体（纯 Python 标准库，任何机器上都能直接跑）
  closed_form_solver.py    闭式解分配算法（核心贡献）
  ga_verifier.py           遗传算法独立验证
  estimate_latency.py      隐藏 latency 的学习（最小二乘拟合）
  refine_loop.py           迭代精化循环

pipeline/   在集群上实际运行的代码（需要 PyTorch + CUDA）
  profile_100.py           Phase 1 小样本画像（100 张，重复 3 次取中位数）
  train_cifar.py           单节点计时训练，报告通信/计算时间
  train_fedavg.py          FedAvg 本地训练步（不重叠分片 + 权重保存）
  aggregate.py             FedAvg 加权聚合 + 测试集评估
  stats3.py                重复实验的统计
  run_strategies.sh        多策略对比驱动
  repeat3.sh               各策略重复 3 次
  run_converge.sh          FedAvg 收敛实验驱动
```

## 与论文章节的对应

| 论文内容 | 代码 |
|---|---|
| 通信成本模型 | `closed_form_solver.py` 中的 `node_time()` |
| 闭式解推导（算法 1） | `closed_form_solver.py` 中的 `closed_form_allocate()` |
| 隐藏 latency 学习（算法 2） | `estimate_latency.py` |
| 迭代精化（算法 3） | `refine_loop.py` |
| 两阶段流程（算法 4） | `profile_100.py` → `closed_form_solver.py` → `run_strategies.sh` |
| GA 验证（算法 5） | `ga_verifier.py` |
| 计时实验结果 | `run_strategies.sh` + `repeat3.sh` + `stats3.py` |
| FedAvg 收敛结果 | `run_converge.sh` + `train_fedavg.py` + `aggregate.py` |

## 完整实验流程

**Step 1 — 小样本画像**

各节点运行 `profile_100.py`，得到吞吐中位数。本研究结果：
736.0 / 741.8 / 1069.0 samples/s（4070 / 3080 / 5080）

**Step 2 — 求初始分配 P0**

```
python core/closed_form_solver.py 736.0 741.8 1069.0 0.3246 0.2500 0.4183 60000
```
得到 P0 = 29.03 / 29.34 / 41.63 %

**Step 3 — 应用到完整数据集并对比基线**

```
export CO1=0.2890 CO2=0.2913 CO3=0.4197
export P01=0.2903 P02=0.2934 P03=0.4163
bash pipeline/run_strategies.sh
```

**Step 4 — 精化**

从上一步日志中按「图片数 ÷ 计算时间」读出真实吞吐（941.5 / 1078.0 / 1716.7），
重新求解得到 P1 = 25.49 / 29.21 / 45.29 %

**Step 5 — 重复实验取误差范围**

```
bash pipeline/repeat3.sh && python pipeline/stats3.py
```

**Step 6 — FedAvg 收敛对比**

```
bash pipeline/run_converge.sh JointP2 0.2733 0.2396 0.4871 10
bash pipeline/run_converge.sh Equal   0.3333 0.3333 0.3334 10
```

## 环境说明

- 容器：Apptainer，镜像基于 `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime`
  （RTX 5080 为 Blackwell 架构 sm_120，必须使用该版本或更新）
- CPU 核数限制：`taskset -c`
- 共享存储：`/uolstore` 网络存储，三节点均可读写，用于放脚本与检查点
- 数据集：CIFAR-10，训练集 50,000 + 测试集 10,000

## 集群配置

驱动脚本顶部的主机地址、容器路径、各节点固定延迟需按实际环境修改。
本研究的配置为：

| 节点 | GPU | CPU 核 | 固定延迟 |
|---|---|---|---|
| worker 1 | RTX 4070 | 10 | 324.6 ms |
| worker 2 | RTX 3080 | 10 | 250.0 ms |
| worker 3 | RTX 5080 | 16 | 418.3 ms |

在 FedAvg 实验中，固定延迟还需加上每轮的权重写入开销
（0.66 / 4.22 / 0.72 s），即 0.985 / 4.470 / 1.138 s。
