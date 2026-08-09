#!/bin/bash
# run_converge.sh — FedAvg 分布式训练收敛实验
#
# 用法：run_converge.sh <标签> <f4070> <f3080> <f5080> <轮数>
# 例：  bash run_converge.sh JointP2 0.2733 0.2396 0.4871 10

PC_4070=txfz0982@129.11.146.207
PC_3080=txfz0982@129.11.145.34
PC_5080=txfz0982@129.11.144.252
WORK=/local/data/txfz0982
SHARE=$HOME/dist_work
SIF=$WORK/pytorch_new.sif
E="apptainer exec --nv --bind $WORK:/data --bind $SHARE:/share $SIF"

TAG=$1; F1=$2; F2=$3; F3=$4; R=$5
mkdir -p $SHARE/ckpt
rm -f $SHARE/ckpt/*.pt                      # 从随机初始化开始

# 由比例算出不重叠分片的边界
B1=$(python3 -c "print(f'{$F1:.4f}')")
B2=$(python3 -c "print(f'{$F1+$F2:.4f}')")
LOG=$SHARE/converge_$TAG.txt; : > $LOG

for ((r=1; r<=R; r++)); do
  # 各节点在自己的分片上本地训练一轮
  A=$(ssh $PC_4070 "taskset -c 0-9  $E python /share/train_fedavg.py 0    $B1 RTX4070 $r" 2>/dev/null)
  B=$(ssh $PC_3080 "taskset -c 0-9  $E python /share/train_fedavg.py $B1  $B2 RTX3080 $r" 2>/dev/null)
  C=$(ssh $PC_5080 "taskset -c 0-15 $E python /share/train_fedavg.py $B2  1.0 RTX5080 $r" 2>/dev/null)

  MK=$(python3 -c "
ts=[float(x.split('|')[2]) for x in ['$A','$B','$C'] if '|' in x]
print(f'{max(ts):.2f} {max(ts)-min(ts):.2f}')")

  # 聚合 + 评估
  EV=$(ssh $PC_5080 "$E python /share/aggregate.py $r $F1 $F2 $F3" 2>/dev/null)
  echo "$EV | makespan_spread $MK" | tee -a $LOG
done
