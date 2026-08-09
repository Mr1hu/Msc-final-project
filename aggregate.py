"""
aggregate.py — FedAvg 加权聚合 + 测试集评估

按各节点的数据量加权平均模型参数，写回全局模型，
再用独立的 10,000 张测试集评估。

运行：
    python aggregate.py <round> <w_4070> <w_3080> <w_5080>
输出：
    ROUND n | test_loss x | test_acc y%
"""
import sys

import torch
import torchvision
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader

rnd = int(sys.argv[1])
w   = [float(x) for x in sys.argv[2:5]]
w   = [x / sum(w) for x in w]                        # 按数据量归一化的聚合权重
names = ["RTX4070", "RTX3080", "RTX5080"]
CK    = "/share/ckpt"

# ---- 加权平均 ----
sds = [torch.load(f"{CK}/{n}.pt", map_location="cpu") for n in names]
avg = {}
for k in sds[0]:
    if sds[0][k].dtype.is_floating_point:
        avg[k] = sum(w[i] * sds[i][k].float() for i in range(3)).to(sds[0][k].dtype)
    else:
        avg[k] = sds[0][k]                           # 整型缓冲（如 num_batches_tracked）不平均
torch.save(avg, f"{CK}/global.pt")

# ---- 独立测试集评估 ----
tf = T.Compose([
    T.Resize(224), T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
te = torchvision.datasets.CIFAR10('/share/cifar', train=False,
                                  download=False, transform=tf)
ld = DataLoader(te, batch_size=256, num_workers=4)

m = models.resnet18(num_classes=10).cuda()
m.load_state_dict(avg)
m.eval()
lf = torch.nn.CrossEntropyLoss()

correct = total = 0
loss_sum = 0.0
with torch.no_grad():
    for xb, yb in ld:
        xb, yb = xb.cuda(), yb.cuda()
        out = m(xb)
        loss_sum += lf(out, yb).item() * yb.size(0)
        correct  += (out.argmax(1) == yb).sum().item()
        total    += yb.size(0)

print(f"ROUND {rnd} | test_loss {loss_sum/total:.4f} | test_acc {correct/total*100:.2f}%")
