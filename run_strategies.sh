#!/bin/bash
# run_strategies.sh — 在完整 60k CIFAR-10 上对比多种分配策略（计时实验）
#
# 用法：先导出各策略的比例，再运行
#   export CO1=0.2890 CO2=0.2913 CO3=0.4197     # 纯算力分配
#   export P01=0.2903 P02=0.2934 P03=0.4163     # 联合优化 P0
#   bash run_strategies.sh

# ---- 集群配置 ----
PC_4070=txfz0982@129.11.146.207
PC_3080=txfz0982@129.11.145.34
PC_5080=txfz0982@129.11.144.252
WORK=/local/data/txfz0982
SHARE=$HOME/dist_work
SIF=$WORK/pytorch_new.sif
E="apptainer exec --nv --bind $WORK:/data --bind $SHARE:/share $SIF"

# ---- 各节点固定延迟 (ms) ----
L_4070=324.6
L_3080=250.0
L_5080=418.3

run() {   # $1=标签 $2=f4070 $3=f3080 $4=f5080
  echo "=== $1 ==="
  R1=$(ssh $PC_4070 "taskset -c 0-9  $E python /share/train_cifar.py $2 $L_4070 RTX4070" 2>/dev/null)
  R2=$(ssh $PC_3080 "taskset -c 0-9  $E python /share/train_cifar.py $3 $L_3080 RTX3080" 2>/dev/null)
  R3=$(ssh $PC_5080 "taskset -c 0-15 $E python /share/train_cifar.py $4 $L_5080 RTX5080" 2>/dev/null)
  echo "$R1"; echo "$R2"; echo "$R3"
  python3 -c "
ts=[float(r.split('|')[4]) for r in ['$R1','$R2','$R3'] if '|' in r]
print(f'  >>> Makespan = {max(ts):.2f}s   (spread {max(ts)-min(ts):.2f}s)')"
  echo ""
}

echo "######### CIFAR-10 60k - allocation strategies #########"
echo ""
run "Equal split"   0.3333 0.3333 0.3334
run "Compute-only"  $CO1 $CO2 $CO3
run "Joint"         $P01 $P02 $P03
