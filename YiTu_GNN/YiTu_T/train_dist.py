import argparse
import os

parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--config', type=str, help='path to config file')
parser.add_argument('--seed', type=int, default=0, help='random seed to use')
parser.add_argument('--num_gpus', type=int, default=4, help='number of gpus to use')
parser.add_argument('--omp_num_threads', type=int, default=8)
parser.add_argument("--local_rank", type=int, default=-1)
args=parser.parse_args()


# set which GPU to use
if args.local_rank < args.num_gpus:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.local_rank)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = str(args.omp_num_threads)
os.environ['MKL_NUM_THREADS'] = str(args.omp_num_threads)

import torch
import dgl
import datetime
import random
import math
import threading
import numpy as np
from tqdm import tqdm
from dgl.utils.shared_mem import create_shared_mem_array, get_shared_mem_array
from sklearn.metrics import average_precision_score, roc_auc_score
from modules import *
from sampler import *
from utils import *

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(args.seed)
torch.distributed.init_process_group(backend='gloo', timeout=datetime.timedelta(0, 3600))
nccl_group = torch.distributed.new_group(ranks=list(range(args.num_gpus)), backend='nccl')

if args.local_rank == 0:
    _node_feats, _edge_feats = load_feat(args.data)
dim_feats = [0, 0, 0, 0, 0, 0]
if args.local_rank == 0:
    if _node_feats is not None:
        dim_feats[0] = _node_feats.shape[0]
        dim_feats[1] = _node_feats.shape[1]
        dim_feats[2] = _node_feats.dtype
        node_feats = create_shared_mem_array('node_feats', _node_feats.shape, dtype=_node_feats.dtype)
        node_feats.copy_(_node_feats)
        del _node_feats
    else:
        node_feats = None
    if _edge_feats is not None:
        dim_feats[3] = _edge_feats.shape[0]
        dim_feats[4] = _edge_feats.shape[1]
        dim_feats[5] = _edge_feats.dtype
        edge_feats = create_shared_mem_array('edge_feats', _edge_feats.shape, dtype=_edge_feats.dtype)
        edge_feats.copy_(_edge_feats)
        del _edge_feats
    else: 
        edge_feats = None
torch.distributed.barrier()
torch.distributed.broadcast_object_list(dim_feats, src=0)
if args.local_rank > 0 and args.local_rank < args.num_gpus:
    node_feats = None
    edge_feats = None
    if os.path.exists('DATA/{}/node_features.pt'.format(args.data)):
        node_feats = get_shared_mem_array('node_feats', (dim_feats[0], dim_feats[1]), dtype=dim_feats[2])
    if os.path.exists('DATA/{}/edge_features.pt'.format(args.data)):
        edge_feats = get_shared_mem_array('edge_feats', (dim_feats[3], dim_feats[4]), dtype=dim_feats[5])
sample_param, memory_param, gnn_param, train_param = parse_config(args.config)
orig_batch_size = train_param['batch_size']
if args.local_rank == 0:
    if not os.path.isdir('models'):
        os.mkdir('models')
    path_saver = ['models/{}_{}.pkl'.format(args.data, time.time())]
else:
    path_saver = [None]
torch.distributed.broadcast_object_list(path_saver, src=0)
path_saver = path_saver[0]

if args.local_rank == args.num_gpus:
    g, df = load_graph(args.data)
    num_nodes = [g['indptr'].shape[0] - 1]
else:
    num_nodes = [None]
torch.distributed.barrier()
torch.distributed.broadcast_object_list(num_nodes, src=args.num_gpus)
num_nodes = num_nodes[0]

mailbox = None
if memory_param['type'] != 'none':
    if args.local_rank == 0:
        node_memory = create_shared_mem_array('node_memory', torch.Size([num_nodes, memory_param['dim_out']]), dtype=torch.float32)
        node_memory_ts = create_shared_mem_array('node_memory_ts', torch.Size([num_nodes]), dtype=torch.float32)
        mails = create_shared_mem_array('mails', torch.Size([num_nodes, memory_param['mailbox_size'], 2 * memory_param['dim_out'] + dim_feats[4]]), dtype=torch.float32)
        mail_ts = create_shared_mem_array('mail_ts', torch.Size([num_nodes, memory_param['mailbox_size']]), dtype=torch.float32)
        next_mail_pos = create_shared_mem_array('next_mail_pos', torch.Size([num_nodes]), dtype=torch.long)
        update_mail_pos = create_shared_mem_array('update_mail_pos', torch.Size([num_nodes]), dtype=torch.int32)
        torch.distributed.barrier()
        node_memory.zero_()
        node_memory_ts.zero_()
        mails.zero_()
        mail_ts.zero_()
        next_mail_pos.zero_()
        update_mail_pos.zero_()
    else:
        torch.distributed.barrier()
        node_memory = get_shared_mem_array('node_memory', torch.Size([num_nodes, memory_param['dim_out']]), dtype=torch.float32)
        node_memory_ts = get_shared_mem_array('node_memory_ts', torch.Size([num_nodes]), dtype=torch.float32)
        mails = get_shared_mem_array('mails', torch.Size([num_nodes, memory_param['mailbox_size'], 2 * memory_param['dim_out'] + dim_feats[4]]), dtype=torch.float32)
        mail_ts = get_shared_mem_array('mail_ts', torch.Size([num_nodes, memory_param['mailbox_size']]), dtype=torch.float32)
        next_mail_pos = get_shared_mem_array('next_mail_pos', torch.Size([num_nodes]), dtype=torch.long)
        update_mail_pos = get_shared_mem_array('update_mail_pos', torch.Size([num_nodes]), dtype=torch.int32)
    mailbox = MailBox(memory_param, num_nodes, dim_feats[4], node_memory, node_memory_ts, mails, mail_ts, next_mail_pos, update_mail_pos)

class DataPipelineThread(threading.Thread):
    
    def __init__(self, my_mfgs, my_root, my_ts, my_eid, my_block, stream):
        super(DataPipelineThread, self).__init__()
        self.my_mfgs = my_mfgs
        self.my_root = my_root
        self.my_ts = my_ts
        self.my_eid = my_eid
        self.my_block = my_block
        self.stream = stream
        self.mfgs = None
        self.root = None
        self.ts = None
        self.eid = None
        self.block = None

    def run(self):
        with torch.cuda.stream(self.stream):
            # print(args.local_rank, 'start thread')
            nids, eids = get_ids(self.my_mfgs[0], node_feats, edge_feats)
            mfgs = mfgs_to_cuda(self.my_mfgs[0])
            prepare_input(mfgs, node_feats, edge_feats, pinned=True, nfeat_buffs=pinned_nfeat_buffs, efeat_buffs=pinned_efeat_buffs, nids=nids, eids=eids)
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs[0], use_pinned_buffers=True)
                if memory_param['deliver_to'] == 'neighbors':
                    self.block = self.my_block[0]
            self.mfgs = mfgs
            self.root = self.my_root[0]
            self.ts = self.my_ts[0]
            self.eid = self.my_eid[0]
            # print(args.local_rank, 'finished')

    def get_stream(self):
        return self.stream

    def get_mfgs(self):
        return self.mfgs
    
    def get_root(self):
        return self.root
    
    def get_ts(self):
        return self.ts

    def get_eid(self):
        return self.eid

    def get_block(self):
        return self.block


if args.local_rank < args.num_gpus:
    # GPU worker process
    model = GeneralModel(dim_feats[1], dim_feats[4], sample_param, memory_param, gnn_param, train_param).cuda()
    find_unused_parameters = True if sample_param['history'] > 1 else False
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], process_group=nccl_group, output_device=args.local_rank, find_unused_parameters=find_unused_parameters)
    creterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_param['lr'])
    pinned_nfeat_buffs, pinned_efeat_buffs = get_pinned_buffers(sample_param, train_param['batch_size'], node_feats, edge_feats)
    if mailbox is not None:
        mailbox.allocate_pinned_memory_buffers(sample_param, train_param['batch_size'])
    tot_loss = 0
    prev_thread = None
    while True:
        my_model_state = [None]
        model_state = [None] * (args.num_gpus + 1)
        torch.distributed.scatter_object_list(my_model_state, model_state, src=args.num_gpus)
        if my_model_state[0] == -1:
            break
        elif my_model_state[0] == 4:
            continue
        elif my_model_state[0] == 2:
            torch.save(model.state_dict(), path_saver)
            continue
        elif my_model_state[0] == 3:
            model.load_state_dict(torch.load(path_saver, map_location=torch.device('cuda:0')))
            continue
        elif my_model_state[0] == 5:
            torch.distributed.gather_object(float(tot_loss), None, dst=args.num_gpus)
            tot_loss = 0
            continue
        elif my_model_state[0] == 0:
            if prev_thread is not None:
                my_mfgs = [None]
                multi_mfgs = [None] * (args.num_gpus + 1)
                my_root = [None]
                multi_root = [None] * (args.num_gpus + 1)
                my_ts = [None]
                multi_ts = [None] * (args.num_gpus + 1)
                my_eid = [None]
                multi_eid = [None] * (args.num_gpus + 1)
                my_block = [None]
                multi_block = [None] * (args.num_gpus + 1)
                torch.distributed.scatter_object_list(my_mfgs, multi_mfgs, src=args.num_gpus)
                if mailbox is not None:
                    torch.distributed.scatter_object_list(my_root, multi_root, src=args.num_gpus)
                    torch.distributed.scatter_object_list(my_ts, multi_ts, src=args.num_gpus)
                    torch.distributed.scatter_object_list(my_eid, multi_eid, src=args.num_gpus)
                    if memory_param['deliver_to'] == 'neighbors':
                        torch.distributed.scatter_object_list(my_block, multi_block, src=args.num_gpus)
                stream = torch.cuda.Stream()
                curr_thread = DataPipelineThread(my_mfgs, my_root, my_ts, my_eid, my_block, stream)
                curr_thread.start()
                prev_thread.join()
                # with torch.cuda.stream(prev_thread.get_stream()):
                mfgs = prev_thread.get_mfgs()
                model.train()
                optimizer.zero_grad()
                pred_pos, pred_neg = model(mfgs)
                loss = creterion(pred_pos, torch.ones_like(pred_pos))
                loss += creterion(pred_neg, torch.zeros_like(pred_neg))
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    tot_loss += float(loss)
                if mailbox is not None:
                    with torch.no_grad():
                        eid = prev_thread.get_eid()
                        mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
                        root_nodes = prev_thread.get_root()
                        ts = prev_thread.get_ts()
                        block = prev_thread.get_block()
                        mailbox.update_mailbox(model.module.memory_updater.last_updated_nid, model.module.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block)
                        mailbox.update_memory(model.module.memory_updater.last_updated_nid, model.module.memory_updater.last_updated_memory, root_nodes, model.module.memory_updater.last_updated_ts)
                        if memory_param['deliver_to'] == 'neighbors':
                            torch.distributed.barrier(group=nccl_group)
                            if args.local_rank == 0:
                                mailbox.update_next_mail_pos()
                prev_thread = curr_thread
            else:
                my_mfgs = [None]
                multi_mfgs = [None] * (args.num_gpus + 1)
                my_root = [None]
                multi_root = [None] * (args.num_gpus + 1)
                my_ts = [None]
                multi_ts = [None] * (args.num_gpus + 1)
                my_eid = [None]
                multi_eid = [None] * (args.num_gpus + 1)
                my_block = [None]
                multi_block = [None] * (args.num_gpus + 1)
                torch.distributed.scatter_object_list(my_mfgs, multi_mfgs, src=args.num_gpus)
                if mailbox is not None:
                    torch.distributed.scatter_object_list(my_root, multi_root, src=args.num_gpus)
                    torch.distributed.scatter_object_list(my_ts, multi_ts, src=args.num_gpus)
                    torch.distributed.scatter_object_list(my_eid, multi_eid, src=args.num_gpus)
                    if memory_param['deliver_to'] == 'neighbors':
                        torch.distributed.scatter_object_list(my_block, multi_block, src=args.num_gpus)
                stream = torch.cuda.Stream()
                prev_thread = DataPipelineThread(my_mfgs, my_root, my_ts, my_eid, my_block, stream)
                prev_thread.start()
        elif my_model_state[0] == 1:
            if prev_thread is not None:
                # finish last training mini-batch
                prev_thread.join()
                mfgs = prev_thread.get_mfgs()
                model.train()
                optimizer.zero_grad()
                pred_pos, pred_neg = model(mfgs)
                loss = creterion(pred_pos, torch.ones_like(pred_pos))
                loss += creterion(pred_neg, torch.zeros_like(pred_neg))
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    tot_loss += float(loss)
                if mailbox is not None:
                    with torch.no_grad():
                        eid = prev_thread.get_eid()
                        mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
                        root_nodes = prev_thread.get_root()
                        ts = prev_thread.get_ts()
                        block = prev_thread.get_block()
                        mailbox.update_mailbox(model.module.memory_updater.last_updated_nid, model.module.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block)
                        mailbox.update_memory(model.module.memory_updater.last_updated_nid, model.module.memory_updater.last_updated_memory, root_nodes, model.module.memory_updater.last_updated_ts)
                        if memory_param['deliver_to'] == 'neighbors':
                            torch.distributed.barrier(group=nccl_group)
                            if args.local_rank == 0:
                                mailbox.update_next_mail_pos()
                prev_thread = None
            my_mfgs = [None]
            multi_mfgs = [None] * (args.num_gpus + 1)
            torch.distributed.scatter_object_list(my_mfgs, multi_mfgs, src=args.num_gpus)
            mfgs = mfgs_to_cuda(my_mfgs[0])
            prepare_input(mfgs, node_feats, edge_feats, pinned=True, nfeat_buffs=pinned_nfeat_buffs, efeat_buffs=pinned_efeat_buffs)
            model.eval()
            with torch.no_grad():
                if mailbox is not None:
                    mailbox.prep_input_mails(mfgs[0])
                pred_pos, pred_neg = model(mfgs)
                if mailbox is not None:
                    my_root = [None]
                    multi_root = [None] * (args.num_gpus + 1)
                    my_ts = [None]
                    multi_ts = [None] * (args.num_gpus + 1)
                    my_eid = [None]
                    multi_eid = [None] * (args.num_gpus + 1)
                    torch.distributed.scatter_object_list(my_root, multi_root, src=args.num_gpus)
                    torch.distributed.scatter_object_list(my_ts, multi_ts, src=args.num_gpus)
                    torch.distributed.scatter_object_list(my_eid, multi_eid, src=args.num_gpus)
                    eid = my_eid[0]
                    mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
                    root_nodes = my_root[0]
                    ts = my_ts[0]
                    block = None
                    if memory_param['deliver_to'] == 'neighbors':
                        my_block = [None]
                        multi_block = [None] * (args.num_gpus + 1)
                        torch.distributed.scatter_object_list(my_block, multi_block, src=args.num_gpus)
                        block = my_block[0]
                    mailbox.update_mailbox(model.module.memory_updater.last_updated_nid, model.module.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block)
                    mailbox.update_memory(model.module.memory_updater.last_updated_nid, model.module.memory_updater.last_updated_memory, root_nodes, model.module.memory_updater.last_updated_ts)
                    if memory_param['deliver_to'] == 'neighbors':
                        torch.distributed.barrier(group=nccl_group)
                        if args.local_rank == 0:
                            mailbox.update_next_mail_pos()
                y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
                y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
                ap = average_precision_score(y_true, y_pred)
                auc = roc_auc_score(y_true, y_pred)
                torch.distributed.gather_object(float(ap), None, dst=args.num_gpus)
                torch.distributed.gather_object(float(auc), None, dst=args.num_gpus)
else:
    # hosting process
    train_edge_end = df[df['ext_roll'].gt(0)].index[0]
    val_edge_end = df[df['ext_roll'].gt(1)].index[0]
    sampler = None
    if not ('no_sample' in sample_param and sample_param['no_sample']):
        sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                                  sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
                                  sample_param['strategy']=='recent', sample_param['prop_time'],
                                  sample_param['history'], float(sample_param['duration']))
    neg_link_sampler = NegLinkSampler(g['indptr'].shape[0] - 1)

    def eval(mode='val'):
        if mode == 'val':
            eval_df = df[train_edge_end:val_edge_end]
        elif mode == 'test':
            eval_df = df[val_edge_end:]
        elif mode == 'train':
            eval_df = df[:train_edge_end]
        ap_tot = list()
        auc_tot = list()
        train_param['batch_size'] = orig_batch_size
        itr_tot = max(len(eval_df) // train_param['batch_size'] // args.num_gpus, 1) * args.num_gpus
        train_param['batch_size'] = math.ceil(len(eval_df) / itr_tot)
        multi_mfgs = list()
        multi_root = list()
        multi_ts = list()
        multi_eid = list()
        multi_block = list()
        for _, rows in eval_df.groupby(eval_df.index // train_param['batch_size']):
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
                mfgs = to_dgl_blocks(ret, sample_param['history'], cuda=False)
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts, cuda=False)
            multi_mfgs.append(mfgs)
            multi_root.append(root_nodes)
            multi_ts.append(ts)
            multi_eid.append(rows['Unnamed: 0'].values)
            if mailbox is not None and memory_param['deliver_to'] == 'neighbors':
                multi_block.append(to_dgl_blocks(ret, sample_param['history'], reverse=True, cuda=False)[0][0])
            if len(multi_mfgs) == args.num_gpus:
                model_state = [1] * (args.num_gpus + 1)
                my_model_state = [None]
                torch.distributed.scatter_object_list(my_model_state, model_state, src=args.num_gpus)
                multi_mfgs.append(None)
                my_mfgs = [None]
                torch.distributed.scatter_object_list(my_mfgs, multi_mfgs, src=args.num_gpus)
                if mailbox is not None:
                    multi_root.append(None)
                    multi_ts.append(None)
                    multi_eid.append(None)
                    my_root = [None]
                    my_ts = [None]
                    my_eid = [None]
                    torch.distributed.scatter_object_list(my_root, multi_root, src=args.num_gpus)
                    torch.distributed.scatter_object_list(my_ts, multi_ts, src=args.num_gpus)
                    torch.distributed.scatter_object_list(my_eid, multi_eid, src=args.num_gpus)
                    if memory_param['deliver_to'] == 'neighbors':
                        multi_block.append(None)
                        my_block = [None]
                        torch.distributed.scatter_object_list(my_block, multi_block, src=args.num_gpus)
                gathered_ap = [None] * (args.num_gpus + 1)
                gathered_auc = [None] * (args.num_gpus + 1)
                torch.distributed.gather_object(float(0), gathered_ap, dst=args.num_gpus)
                torch.distributed.gather_object(float(0), gathered_auc, dst=args.num_gpus)
                ap_tot += gathered_ap[:-1]
                auc_tot += gathered_auc[:-1]
                multi_mfgs = list()
                multi_root = list()
                multi_ts = list()
                multi_eid = list()
                multi_block = list()
            pbar.update(1)
        ap = float(torch.tensor(ap_tot).mean())
        auc = float(torch.tensor(auc_tot).mean())
        return ap, auc

    best_ap = 0
    best_e = 0
    tap = 0
    tauc = 0
    for e in range(train_param['epoch']):
        print('Epoch {:d}:'.format(e))
        time_sample = 0
        time_tot = 0
        if sampler is not None:
            sampler.reset()
        if mailbox is not None:
            mailbox.reset()
        # training
        train_param['batch_size'] = orig_batch_size
        itr_tot = train_edge_end // train_param['batch_size'] // args.num_gpus * args.num_gpus
        train_param['batch_size'] = math.ceil(train_edge_end / itr_tot)
        multi_mfgs = list()
        multi_root = list()
        multi_ts = list()
        multi_eid = list()
        multi_block = list()
        group_indexes = list()
        group_indexes.append(np.array(df[:train_edge_end].index // train_param['batch_size']))
        if 'reorder' in train_param:
            # random chunk shceduling
            reorder = train_param['reorder']
            group_idx = list()
            for i in range(reorder):
                group_idx += list(range(0 - i, reorder - i))
            group_idx = np.repeat(np.array(group_idx), train_param['batch_size'] // reorder)
            group_idx = np.tile(group_idx, train_edge_end // train_param['batch_size'] + 1)[:train_edge_end]
            group_indexes.append(group_indexes[0] + group_idx)
            base_idx = group_indexes[0]
            for i in range(1, train_param['reorder']):
                additional_idx = np.zeros(train_param['batch_size'] // train_param['reorder'] * i) - 1
                group_indexes.append(np.concatenate([additional_idx, base_idx])[:base_idx.shape[0]])
        with tqdm(total=itr_tot + max((val_edge_end - train_edge_end) // train_param['batch_size'] // args.num_gpus, 1) * args.num_gpus) as pbar:
            for _, rows in df[:train_edge_end].groupby(group_indexes[random.randint(0, len(group_indexes) - 1)]):
                t_tot_s = time.time()
                root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))]).astype(np.int32)
                ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
                if sampler is not None:
                    if 'no_neg' in sample_param and sample_param['no_neg']:
                        pos_root_end = root_nodes.shape[0] * 2 // 3
                        sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                    else:
                        sampler.sample(root_nodes, ts)
                    ret = sampler.get_ret()
                    time_sample += ret[0].sample_time()
                if gnn_param['arch'] != 'identity':
                    mfgs = to_dgl_blocks(ret, sample_param['history'], cuda=False)
                else:
                    mfgs = node_to_dgl_blocks(root_nodes, ts, cuda=False)
                multi_mfgs.append(mfgs)
                multi_root.append(root_nodes)
                multi_ts.append(ts)
                multi_eid.append(rows['Unnamed: 0'].values)
                if mailbox is not None and memory_param['deliver_to'] == 'neighbors':
                    multi_block.append(to_dgl_blocks(ret, sample_param['history'], reverse=True, cuda=False)[0][0])
                if len(multi_mfgs) == args.num_gpus:
                    model_state = [0] * (args.num_gpus + 1)
                    my_model_state = [None]
                    torch.distributed.scatter_object_list(my_model_state, model_state, src=args.num_gpus)
                    multi_mfgs.append(None)
                    my_mfgs = [None]
                    torch.distributed.scatter_object_list(my_mfgs, multi_mfgs, src=args.num_gpus)
                    if mailbox is not None:
                        multi_root.append(None)
                        multi_ts.append(None)
                        multi_eid.append(None)
                        my_root = [None]
                        my_ts = [None]
                        my_eid = [None]
                        torch.distributed.scatter_object_list(my_root, multi_root, src=args.num_gpus)
                        torch.distributed.scatter_object_list(my_ts, multi_ts, src=args.num_gpus)
                        torch.distributed.scatter_object_list(my_eid, multi_eid, src=args.num_gpus)
                        if memory_param['deliver_to'] == 'neighbors':
                            multi_block.append(None)
                            my_block = [None]
                            torch.distributed.scatter_object_list(my_block, multi_block, src=args.num_gpus)
                    multi_mfgs = list()
                    multi_root = list()
                    multi_ts = list()
                    multi_eid = list()
                    multi_block = list()
                pbar.update(1)
                time_tot += time.time() - t_tot_s
            print('Training time:',time_tot)
            model_state = [5] * (args.num_gpus + 1)
            my_model_state = [None]
            torch.distributed.scatter_object_list(my_model_state, model_state, src=args.num_gpus)
            gathered_loss = [None] * (args.num_gpus + 1)
            torch.distributed.gather_object(float(0), gathered_loss, dst=args.num_gpus)
            total_loss = np.sum(np.array(gathered_loss) * train_param['batch_size'])
            ap, auc = eval('val')
            if ap > best_ap:
                best_e = e
                best_ap = ap
                model_state = [4] * (args.num_gpus + 1)
                model_state[0] = 2
                my_model_state = [None]
                torch.distributed.scatter_object_list(my_model_state, model_state, src=args.num_gpus)
                # for memory based models, testing after validation is faster
                tap, tauc = eval('test')
        print('\ttrain loss:{:.4f}  val ap:{:4f}  val auc:{:4f}'.format(total_loss, ap, auc))
        print('\ttotal time:{:.2f}s sample time:{:.2f}s'.format(time_tot, time_sample))

    print('Best model at epoch {}.'.format(best_e))
    print('\ttest ap:{:4f}  test auc:{:4f}'.format(tap, tauc))

    # let all process exit
    model_state = [-1] * (args.num_gpus + 1)
    my_model_state = [None]
    torch.distributed.scatter_object_list(my_model_state, model_state, src=args.num_gpus)