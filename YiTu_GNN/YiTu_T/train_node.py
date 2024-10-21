import argparse
import os
import hashlib

parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--config', type=str, default='', help='path to config file')
parser.add_argument('--batch_size', type=int, default=4000)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--dim', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--gpu', type=str, default='0', help='which GPU to use')
parser.add_argument('--model', type=str, default='', help='name of stored model to load')
parser.add_argument('--posneg', default=False, action='store_true', help='for positive negative detection, whether to sample negative nodes')
args=parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if args.data == 'WIKI' or args.data == 'REDDIT':
    args.posneg = True

import torch
import time
import random
import dgl
import numpy as np
import pandas as pd
from modules import *
from sampler import *
from utils import *
from tqdm import tqdm
from sklearn.metrics import average_precision_score, f1_score

ldf = pd.read_csv('DATA/{}/labels.csv'.format(args.data))
role = ldf['ext_roll'].values
# train_node_end = ldf[ldf['ext_roll'].gt(0)].index[0]
# val_node_end = ldf[ldf['ext_roll'].gt(1)].index[0]
labels = ldf['label'].values.astype(np.int64)

emb_file_name = hashlib.md5(str(torch.load(args.model, map_location=torch.device('cpu'))).encode('utf-8')).hexdigest() + '.pt'
if not os.path.isdir('embs'):
    os.mkdir('embs')
if not os.path.isfile('embs/' + emb_file_name):
    print('Generating temporal embeddings..')

    node_feats, edge_feats = load_feat(args.data)
    g, df = load_graph(args.data)
    sample_param, memory_param, gnn_param, train_param = parse_config(args.config)
    train_edge_end = df[df['ext_roll'].gt(0)].index[0]
    val_edge_end = df[df['ext_roll'].gt(1)].index[0]

    gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
    gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]
    combine_first = False
    if 'combine_neighs' in train_param and train_param['combine_neighs']:
        combine_first = True
    model = GeneralModel(gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param, combined=combine_first).cuda()
    mailbox = MailBox(memory_param, g['indptr'].shape[0] - 1, gnn_dim_edge) if memory_param['type'] != 'none' else None
    creterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_param['lr'])
    if 'all_on_gpu' in train_param and train_param['all_on_gpu']:
        if node_feats is not None:
            node_feats = node_feats.cuda()
        if edge_feats is not None:
            edge_feats = edge_feats.cuda()
        if mailbox is not None:
            mailbox.move_to_gpu()

    sampler = None
    if not ('no_sample' in sample_param and sample_param['no_sample']):
        sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                                sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
                                sample_param['strategy']=='recent', sample_param['prop_time'],
                                sample_param['history'], float(sample_param['duration']))
    neg_link_sampler = NegLinkSampler(g['indptr'].shape[0] - 1)

    model.load_state_dict(torch.load(args.model))

    processed_edge_id = 0

    def forward_model_to(time):
        global processed_edge_id
        if processed_edge_id >= len(df):
            return
        while df.time[processed_edge_id] < time:
            rows = df[processed_edge_id:min(processed_edge_id + train_param['batch_size'], len(df))]
            if processed_edge_id < train_edge_end:
                model.train()
            else:
                model.eval()
            root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))]).astype(np.int32)
            ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
            if sampler is not None:
                if 'no_neg' in sample_param and sample_param['no_neg']:
                    pos_root_end = root_nodes.shape[0] * 2 // 3
                    sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                else:
                    sampler.sample(root_nodes, ts)
                ret = sampler.get_ret()
            if gnn_param['arch'] != 'identity':
                mfgs = to_dgl_blocks(ret, sample_param['history'])
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts)
            mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs[0])
            with torch.no_grad():
                pred_pos, pred_neg = model(mfgs)
                if mailbox is not None:
                    eid = rows['Unnamed: 0'].values
                    mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
                    block = None
                    if memory_param['deliver_to'] == 'neighbors':
                        block = to_dgl_blocks(ret, sample_param['history'], reverse=True)[0][0]
                    mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block)
                    mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, model.memory_updater.last_updated_ts)
            processed_edge_id += train_param['batch_size']
            if processed_edge_id >= len(df):
                return

    def get_node_emb(root_nodes, ts):
        forward_model_to(ts[-1])
        if sampler is not None:
            sampler.sample(root_nodes, ts)
            ret = sampler.get_ret()
        if gnn_param['arch'] != 'identity':
            mfgs = to_dgl_blocks(ret, sample_param['history'])
        else:
            mfgs = node_to_dgl_blocks(root_nodes, ts)
        mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
        if mailbox is not None:
            mailbox.prep_input_mails(mfgs[0])
        with torch.no_grad():
            ret = model.get_emb(mfgs)
        return ret.detach().cpu()

    emb = list()
    for _, rows in tqdm(ldf.groupby(ldf.index // args.batch_size)):
        emb.append(get_node_emb(rows.node.values.astype(np.int32), rows.time.values.astype(np.float32)))
    emb = torch.cat(emb, dim=0)
    torch.save(emb, 'embs/' + emb_file_name)
    print('Saved to embs/' + emb_file_name)
else:
    print('Loading temporal embeddings from embs/' + emb_file_name)
    emb = torch.load('embs/' + emb_file_name)

model = NodeClassificationModel(emb.shape[1], args.dim, labels.max() + 1).cuda()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
labels = torch.from_numpy(labels).type(torch.int32)
role = torch.from_numpy(role).type(torch.int32)
emb = emb

class NodeEmbMinibatch():

    def __init__(self, emb, role, label, batch_size):
        self.role = role
        self.label = label
        self.batch_size = batch_size
        self.train_emb = emb[role == 0]
        self.val_emb = emb[role == 1]
        self.test_emb = emb[role == 2]
        self.train_label = label[role == 0]
        self.val_label = label[role == 1]
        self.test_label = label[role == 2]
        self.mode = 0
        self.s_idx = 0

    def shuffle(self):
        perm = torch.randperm(self.train_emb.shape[0])
        self.train_emb = self.train_emb[perm]
        self.train_label = self.train_label[perm]

    def set_mode(self, mode):
        if mode == 'train':
            self.mode = 0
        elif mode == 'val':
            self.mode = 1
        elif mode == 'test':
            self.mode = 2
        self.s_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.mode == 0:
            emb = self.train_emb
            label = self.train_label
        elif self.mode == 1:
            emb = self.val_emb
            label = self.val_label
        else:
            emb = self.test_emb
            label = self.test_label
        if self.s_idx >= emb.shape[0]:
            raise StopIteration
        else:
            end = min(self.s_idx + self.batch_size, emb.shape[0])
            curr_emb = emb[self.s_idx:end]
            curr_label = label[self.s_idx:end]
            self.s_idx += self.batch_size
            return curr_emb.cuda(), curr_label.cuda()

if args.posneg:
    role = role[labels == 1]
    emb_neg = emb[labels == 0].cuda()
    emb = emb[labels == 1]
    labels = torch.ones(emb.shape[0], dtype=torch.int64).cuda()
    labels_neg = torch.zeros(emb_neg.shape[0], dtype=torch.int64).cuda()
    neg_node_sampler = NegLinkSampler(emb_neg.shape[0])

minibatch = NodeEmbMinibatch(emb, role, labels, args.batch_size)
if not os.path.isdir('models'):
    os.mkdir('models')
save_path = 'models/node_' + args.model.split('/')[-1]
best_e = 0
best_acc = 0
for e in range(args.epoch):
    minibatch.set_mode('train')
    minibatch.shuffle()
    model.train()
    for emb, label in minibatch:
        optimizer.zero_grad()
        if args.posneg:
            neg_idx = neg_node_sampler.sample(emb.shape[0])
            emb = torch.cat([emb, emb_neg[neg_idx]], dim=0)
            label = torch.cat([label, labels_neg[neg_idx]], dim=0)
        pred = model(emb)
        loss = loss_fn(pred, label.long())
        loss.backward()
        optimizer.step()
    minibatch.set_mode('val')
    model.eval()
    accs = list()
    with torch.no_grad():
        for emb, label in minibatch:
            if args.posneg:
                neg_idx = neg_node_sampler.sample(emb.shape[0])
                emb = torch.cat([emb, emb_neg[neg_idx]], dim=0)
                label = torch.cat([label, labels_neg[neg_idx]], dim=0)
            pred = model(emb)
            if args.posneg:
                acc = average_precision_score(label.cpu(), pred.softmax(dim=1)[:, 1].cpu())
            else:
                acc = f1_score(label.cpu(), torch.argmax(pred, dim=1).cpu(), average="micro")
            accs.append(acc)
        acc = float(torch.tensor(accs).mean())
    print('Epoch: {}\tVal acc: {:.4f}'.format(e, acc))
    if acc > best_acc:
        best_e = e
        best_acc = acc
        torch.save(model.state_dict(), save_path)
print('Loading model at epoch {}...'.format(best_e))
model.load_state_dict(torch.load(save_path))
minibatch.set_mode('test')
model.eval()
accs = list()
with torch.no_grad():
    for emb, label in minibatch:
        if args.posneg:
            neg_idx = neg_node_sampler.sample(emb.shape[0])
            emb = torch.cat([emb, emb_neg[neg_idx]], dim=0)
            label = torch.cat([label, labels_neg[neg_idx]], dim=0)
        pred = model(emb)
        if args.posneg:
            acc = average_precision_score(label.cpu(), pred.softmax(dim=1)[:, 1].cpu())
        else:
            acc = f1_score(label.cpu(), torch.argmax(pred, dim=1).cpu(), average="micro")
        accs.append(acc)
    acc = float(torch.tensor(accs).mean())
print('Testing acc: {:.4f}'.format(acc))