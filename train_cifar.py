"""
train_cifar.py — Phase 2 单节点训练（计时实验用）

在分到的数据分片上训练，报告通信时间与计算时间。
通信时间按导师定义的模型计算：T_comm = L + DataSize / DataRate

在各 worker 的容器内运行：
    python train_cifar.py <fraction> <latency_ms> <node_name>
输出：
    node_name|n_images|comm_time|compute_time|total_time
"""
import sys
import time

import torch
import torchvision
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader, Subset, ConcatDataset

frac    = float(sys.argv[1])     # 本节点分到的比例
lat_ms  = float(sys.argv[2])     # 该节点的固定延迟 (ms)
node    = sys.argv[3]

EPOCHS      = 1
BATCH       = 128
MB_PER_IMG  = 0.003              # CIFAR 32x32x3 ≈ 3KB
DATA_RATE   = 111.0              # MB/s，三节点相同

tf = T.Compose([
    T.Resize(224), T.RandomHorizontalFlip(), T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
tr   = torchvision.datasets.CIFAR10('/share/cifar', train=True,  download=False, transform=tf)
te   = torchvision.datasets.CIFAR10('/share/cifar', train=False, download=False, transform=tf)
full = ConcatDataset([tr, te])                        # 完整 60,000 张

n     = int(len(full) * frac)
shard = Subset(full, list(range(n)))

# 通信时间 = 固定延迟 + 数据量 / 带宽
comm = lat_ms / 1000 + n * MB_PER_IMG / DATA_RATE

ld    = DataLoader(shard, batch_size=BATCH, num_workers=4, shuffle=True, pin_memory=True)
model = models.resnet18(num_classes=10).cuda()
opt   = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
lf    = torch.nn.CrossEntropyLoss()
model.train()

# 计算时间 = 真实训练耗时
t0 = time.time()
for _ in range(EPOCHS):
    for xb, yb in ld:
        opt.zero_grad(); lf(model(xb.cuda()), yb.cuda()).backward(); opt.step()
torch.cuda.synchronize()
compute = time.time() - t0

print(f"{node}|{n}|{comm:.2f}|{compute:.2f}|{comm + compute:.2f}")
