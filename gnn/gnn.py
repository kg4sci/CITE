import argparse
import os.path as osp
import time

import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score
from model import GCN, SAGE, GAT

parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='citeseer')
parser.add_argument('--model', type=str, default='GCN')
parser.add_argument('--num_layers', type=int, default='3')
parser.add_argument('--dropout', type=float, default='0.5')


parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--use_gdc', action='store_true', help='Use GDC')
parser.add_argument('--st', action='store_true', help='Use ST Embeddings')
parser.add_argument('--wandb', action='store_true', help='Track experiment')
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

init_wandb(
    name=f'GCN-{args.dataname}',
    lr=args.lr,
    epochs=args.epochs,
    hidden_channels=args.hidden_channels,
    device=device,
)

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
# dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
dataname = args.dataname
data = torch.load(f'../../datasets/{dataname}.pt').to(device)
if args.st:
    data.x = torch.load(f'st_embeddings/{dataname}_st.pt').to(device)
data.edge_index = to_undirected(data.edge_index)

# if len(data.train_mask)==10:
#     data.train_mask = data.train_mask[0]
#     data.val_mask = data.val_mask[0]
#     data.test_mask = data.test_mask[0]

# 随机种子，保证每次划分相同
torch.manual_seed(42)

# 加载数据
data = torch.load(f'../../datasets/{dataname}.pt').to(device)
if args.st:
    data.x = torch.load(f'st_embeddings/{dataname}_st.pt').to(device)
data.edge_index = to_undirected(data.edge_index)

# 获取所有节点的数量
num_nodes = data.x.shape[0]

# 生成一个随机排列的节点索引
random_idx = torch.randperm(num_nodes)

# 计算每个部分的大小
train_size = int(0.6 * num_nodes)
val_size = int(0.2 * num_nodes)
test_size = num_nodes - train_size - val_size  # 剩下的就是测试集大小

# 划分数据集
train_idx = random_idx[:train_size]
val_idx = random_idx[train_size:train_size + val_size]
test_idx = random_idx[train_size + val_size:]

# 创建mask并分配到train_mask, val_mask, test_mask
data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

data.train_mask[train_idx] = True
data.val_mask[val_idx] = True
data.test_mask[test_idx] = True

if args.use_gdc:
    transform = T.GDC(
        self_loop_weight=1,
        normalization_in='sym',
        normalization_out='col',
        diffusion_kwargs=dict(method='ppr', alpha=0.05),
        sparsification_kwargs=dict(method='topk', k=128, dim=0),
        exact=True,
    )
    data = transform(data)





if args.model == 'GCN':
    model = GCN(
    in_channels=data.x.shape[1],
    hidden_channels=args.hidden_channels,
    out_channels=len(data.label_name),
    num_layers=args.num_layers,
    dropout=args.dropout
    ).to(device)
elif args.model == 'SAGE':
    model = SAGE(
    in_channels=data.x.shape[1],
    hidden_channels=args.hidden_channels,
    out_channels=len(data.label_name),
    num_layers=args.num_layers,
    dropout=args.dropout
    ).to(device)
elif args.model == 'GAT':
    model = SAGE(
    in_channels=data.x.shape[1],
    hidden_channels=args.hidden_channels,
    out_channels=len(data.label_name),
    num_layers=args.num_layers,
    dropout=args.dropout
    ).to(device)
# optimizer = torch.optim.Adam([
#     dict(params=model.conv1.parameters(), weight_decay=5e-4),
#     dict(params=model.conv2.parameters(), weight_decay=0)
# ], lr=args.lr)  # Only perform weight-decay on first convolution.

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_attr).argmax(dim=-1)

    accs = []
    macro_f1 = []
    micro_f1 = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        acc = int((pred[mask] == data.y[mask]).sum()) / int(mask.sum())
        accs.append(acc)
        # 计算 F1 Macro
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        f1 = f1_score(data.y[mask].cpu().numpy(), pred[mask].cpu().numpy(), average='macro')
        macro_f1.append(f1)
        f1 = f1_score(data.y[mask].cpu().numpy(), pred[mask].cpu().numpy(), average='micro')
        micro_f1.append(f1)
    return accs, macro_f1, micro_f1


best_val_acc = test_acc = 0
best_val_f1 = test_f1 = 0
times = []
import csv
with open('results.csv', mode='a', newline='') as file:

    writer = csv.writer(file)
    
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        loss = train()
        train_acc, val_acc, tmp_test_acc = test()[0]
        train_f1, val_f1, tmp_test_f1 = test()[1]
        train_f1mi, val_f1mi, tmp_test_f1mi = test()[2]
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            test_f1 = tmp_test_f1
        log(Epoch=epoch, Loss=loss, Train_acc=train_acc, Val_acc=val_acc, Test_acc=test_acc, Train_f1=train_f1, Val_f1=val_f1, Test_f1=test_f1,Train_f1mi=train_f1mi, Val_f1mi=val_f1mi, Test_f1mi=tmp_test_f1mi)
        log(Test_acc=test_acc,Test_f1=test_f1,Test_f1mi=tmp_test_f1mi)
        times.append(time.time() - start)

    writer.writerow([args.model, args.dataname, args.num_layers, args.hidden_channels, args.dropout, test_acc, test_f1])

print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')