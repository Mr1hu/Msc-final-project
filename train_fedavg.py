"""
train_fedavg.py — FedAvg 本地训练步（收敛实验用）

与 train_cifar.py 的区别：
  1. 数据分片不重叠，三节点的并集等于完整训练集
  2. 每轮开始加载全局模型，结束保存本地模型 —— 这是真正的分布式训练
  3. 只用 50,000 张训练集，10,000 张测试集留给独立评估

在各 worker 的容器内运行：
    python train_fedavg.py <start_frac> <end_frac> <node_name> <round>
输出：
    node_name|n_images|elapsed|avg_train_loss
"""
import sys
import os
import time

import torch
import torchvision
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader, Subset

s_f  = float(sys.argv[1])        # 本节点分片起点（占全集比例）
e_f  = float(sys.argv[2])        # 本节点分片终点
node = sys.argv[3]
rnd  = int(sys.argv[4])

BATCH = 128
CK    = "/share/ckpt"            # 共享网络存储上的检查点目录

tf = T.Compose([
    T.Resize(224), T.RandomHorizontalFlip(), T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
full = torchvision.datasets.CIFAR10('/share/cifar', train=True,
                                    download=False, transform=tf)
N      = len(full)                                   # 50,000
lo, hi = int(N * s_f), int(N * e_f)
shard  = Subset(full, list(range(lo, hi)))           # 不重叠分片

model = models.resnet18(num_classes=10).cuda()
g = f"{CK}/global.pt"
if os.path.exists(g):                                # 第 2 轮起加载全局模型
    model.load_state_dict(torch.load(g, map_location="cuda"))

opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
lf  = torch.nn.CrossEntropyLoss()
model.train()
ld  = DataLoader(shard, batch_size=BATCH, num_workers=4, shuffle=True, pin_memory=True)

t0, tot, nb = time.time(), 0.0, 0
for xb, yb in ld:
    opt.zero_grad()
    loss = lf(model(xb.cuda()), yb.cuda())
    loss.backward(); opt.step()
    tot += loss.item(); nb += 1
torch.cuda.synchronize()

torch.save(model.state_dict(), f"{CK}/{node}.pt")    # 写入共享存储（构成每轮固定开销）
print(f"{node}|{hi - lo}|{time.time() - t0:.2f}|{tot / nb:.4f}")
