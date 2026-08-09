#!/bin/bash
# repeat3.sh — 各策略重复 3 次，用于给出误差范围
# 结果写入 $SHARE/repeat3_raw.txt，随后用 stats3.py 统计

PC_4070=txfz0982@129.11.146.207
PC_3080=txfz0982@129.11.145.34
PC_5080=txfz0982@129.11.144.252
WORK=/local/data/txfz0982
SHARE=$HOME/dist_work
SIF=$WORK/pytorch_new.sif
E="apptainer exec --nv --bind $WORK:/data --bind $SHARE:/share $SIF"

OUT=$SHARE/repeat3_raw.txt
: > $OUT

one() {   # $1=标签 $2=f4070 $3=f3080 $4=f5080 $5=重复序号
  R1=$(ssh $PC_4070 "taskset -c 0-9  $E python /share/train_cifar.py $2 324.6 RTX4070" 2>/dev/null)
  R2=$(ssh $PC_3080 "taskset -c 0-9  $E python /share/train_cifar.py $3 250.0 RTX3080" 2>/dev/null)
  R3=$(ssh $PC_5080 "taskset -c 0-15 $E python /share/train_cifar.py $4 418.3 RTX5080" 2>/dev/null)
  echo "$1|$5|$R1;$R2;$R3" >> $OUT
  echo "  rep$5 done: $1"
}

for REP in 1 2 3; do
  echo "--- repetition $REP ---"
  one "Equal"        0.3333 0.3333 0.3334 $REP
  one "ComputeOnly"  0.2890 0.2913 0.4197 $REP
  one "JointP0"      0.2903 0.2934 0.4163 $REP
  one "JointP1"      0.2549 0.2921 0.4529 $REP
done
echo "raw -> $OUT"
