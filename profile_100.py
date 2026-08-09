"""
profile_100.py — Phase 1 小样本画像

用极小样本（100 张）快速估计各节点吞吐。重复 3 次取中位数，
以避免单次测量在不稳定节点上误导后续精化。

在各 worker 的容器内运行：
    python profile_100.py <node_name>
输出：
    node_name|median_throughput|[三次原始值]
"""
import sys
import time
import statistics

import torch
import torchvision
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader, Subset

node   = sys.argv[1]
N      = 100      # 画像样本数
BATCH  = 20
REPS   = 3        # 重复次数，取中位数

tf = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
ds    = torchvision.datasets.CIFAR10('/share/cifar', train=True,
                                     download=False, transform=tf)
shard = Subset(ds, list(range(N)))

model = models.resnet18(num_classes=10).cuda()
opt   = torch.optim.SGD(model.parameters(), lr=0.01)
lf    = torch.nn.CrossEntropyLoss()
model.train()


def run_once():
    ld = DataLoader(shard, batch_size=BATCH, num_workers=2, shuffle=True)
    for xb, yb in ld:                                   # 预热一个 batch
        opt.zero_grad(); lf(model(xb.cuda()), yb.cuda()).backward(); opt.step()
        break
    torch.cuda.synchronize()
    t0 = time.time()
    for xb, yb in ld:
        opt.zero_grad(); lf(model(xb.cuda()), yb.cuda()).backward(); opt.step()
    torch.cuda.synchronize()
    return N / (time.time() - t0)


vals = [run_once() for _ in range(REPS)]
print(f"{node}|{statistics.median(vals):.1f}|{[round(v, 1) for v in vals]}")
