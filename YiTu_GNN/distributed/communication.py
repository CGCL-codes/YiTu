import logging
import pickle
import threading

import dgl
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp

from .threadsafe_queue import Queue


def pull_subgraph(block, world_size, device):
    """拉取所有节点上的子图拓扑, 用于第一层的模型并行计算
    """
    src_nodes = block.srcdata[dgl.NID]
    # 获取原始边
    src = src_nodes[block.edges()[0]].to(device)
    dst = src_nodes[block.edges()[1]].to(device)
    # 保存目标节点，防止构建block时删除没有入边的节点
    dst_nodes = block.dstdata[dgl.NID].to(device)
    # 异步交换第一层的block
    handler1, size1, dst_nodes_list1 = all_gather_async(dst_nodes, world_size, device)
    handler2, size2, src_list1 = all_gather_async(src, world_size, device)
    handler3, size3, dst_list1 = all_gather_async(dst, world_size, device)

    handler1.wait()
    dst_nodes_list = []
    for size, tensor in zip(size1, dst_nodes_list1):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        dst_nodes_list.append(pickle.loads(buffer).to(device))
    handler2.wait()
    src_list = []
    for size, tensor in zip(size2, src_list1):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        src_list.append(pickle.loads(buffer).to(device))
    handler3.wait()
    dst_list = []
    for size, tensor in zip(size3, dst_list1):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        dst_list.append(pickle.loads(buffer).to(device))
    new_blocks = []
    for i in range(len(dst_nodes_list1)):
        # 指定dst_nodes，否则一些入边为0的节点不存在
        b = dgl.to_block(
            dgl.graph((src_list[i], dst_list[i])), dst_nodes=dst_nodes_list[i]
        ).to(device)
        # b.srcdata["features"] = in_feats[b.srcdata[dgl.NID]].to(device)
        new_blocks.append(b)
    return new_blocks


def all_gather_tensors(data, device):
    world_size = dist.get_world_size()
    if world_size == 1:
        return [data]
    ## gather shapes first
    myshape = data.shape
    mycount = data.size
    shape_tensor = torch.Tensor(np.array(myshape)).to(device)
    all_shape = [torch.Tensor(np.array(myshape)).to(device) for i in range(world_size)]
    dist.all_gather(all_shape, shape_tensor)
    ## compute largest shapes
    all_shape = [x.cpu().numpy() for x in all_shape]
    all_count = [int(x.prod()) for x in all_shape]
    all_shape = [list(map(int, x)) for x in all_shape]
    max_count = max(all_count)
    ## padding tensors and gather them
    output_tensors = [torch.Tensor(max_count).to(device) for i in range(world_size)]
    padded_input_array = np.zeros(max_count)
    padded_input_array[:mycount] = data.reshape(-1)
    input_tensor = torch.Tensor(padded_input_array).to(device)
    dist.all_gather(output_tensors, input_tensor)
    ## unpadding gathered tensors
    padded_output = [x.cpu().numpy() for x in output_tensors]
    output = [
        x[: all_count[i]].reshape(all_shape[i]) for i, x in enumerate(padded_output)
    ]
    return output


def all_gather(data, world_size, device):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    # world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device)

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to(device)
    size_list = [torch.LongTensor([0]).to(device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    # logging.debug("all gather get size list: {}".format(size_list))
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to(device))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to(device)
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer).to(device))

    return data_list


def all_gather_async(data, world_size, device):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    # world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device)

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to(device)
    size_list = [torch.LongTensor([0]).to(device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to(device))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to(device)
        tensor = torch.cat((tensor, padding), dim=0)
    work = dist.all_gather(tensor_list, tensor, async_op=True)

    return work, size_list, tensor_list


class Communication(object):
    """负责模型并行和数据并行之间切换时的数据通信
    前向传播时, 由模型并行切换到数据并行, 将每个节点上部分激活值进行汇聚；
    反向传播时, 由数据并行切换到模型并行, 负责对梯度进行通信。
    """

    def __init__(self, rank, num_workers, device):
        self.rank = rank
        self.num_workers = num_workers
        self.device = device
        self.forward_receive_tensor = Queue()
        self.forward_send_tensor = Queue()
        self.backward_receive_tensor = Queue()
        self.backward_send_tensor = Queue()

    def start_comm_thread(self, num_iterations):
        # forward
        forward_args = (
            self.forward_send_tensor,
            self.forward_receive_tensor,
            num_iterations,
            self.rank,
            self.num_workers,
        )
        forward = threading.Thread(target=forward_comm, args=forward_args)
        forward.start()
        # backward
        backward_args = (
            self.backward_send_tensor,
            self.backward_receive_tensor,
            num_iterations,
            self.num_workers,
            self.rank,
            self.device,
        )
        backward = threading.Thread(target=backward_comm, args=backward_args)
        backward.start()

    def start_eval_comm_thread(self, num_iterations):
        # forward
        forward_args = (
            self.forward_send_tensor,
            self.forward_receive_tensor,
            num_iterations,
            self.rank,
            self.num_workers,
        )
        forward = threading.Thread(target=forward_comm, args=forward_args)
        forward.start()

    def send(self, tensor, forward=True):
        """将需要发送的tensor写入到对应的队列中"""
        if forward:
            self.forward_send_tensor.push(tensor)
        else:
            self.backward_send_tensor.push(tensor)

    def receive(self, forward=True):
        """从receive队列中读取接收到的tensor并返回"""
        if forward:
            return self.forward_receive_tensor.pop()
        else:
            return self.backward_receive_tensor.pop()


def forward_comm(
    forward_send_tensor, forward_receive_tensor, num_iterations, rank, num_workers
):
    logging.info("Rank:{}, Forward communication starts!".format(rank))
    for it in range(num_iterations):
        # num_workers个tensor concat在一起
        # nodes_offset: nodes_offset[i + 1] - nodes_offset[i]表示第i个worker在模型并行后获得的feature大小
        feats, nodes_offset = forward_send_tensor.pop()
        logging.debug(
            "Iteration: {}, Rank: {} send model forward output, nodes offset: {}, feats shape: {}".format(
                it, rank, nodes_offset, feats.shape
            )
        )
        handlers = []
        for i in range(num_workers):
            # 切片并不会拷贝，所以reduce最终会直接修改feats
            x = feats[
                nodes_offset[i] : nodes_offset[i + 1],
            ]
            handler = dist.reduce(x, i, op=ReduceOp.SUM, async_op=True)
            handlers.append((handler, i))
        for h, r in handlers:
            # 等待reduce操作完成
            h.wait()
            if r == rank:
                output = feats[
                    nodes_offset[rank] : nodes_offset[rank + 1],
                ]
                logging.debug(
                    "Iteration: {}, Rank: {} receive model forward output, feats shape: {}".format(
                        it, rank, output.shape
                    )
                )
                forward_receive_tensor.push(output)
    logging.info("Rank:{}, Forward communication has finished!".format(rank))


def backward_comm(
    backward_send_tensor,
    backward_receive_tensor,
    num_iterations,
    num_workers,
    rank,
    device,
):
    logging.info("Rank:{}, Backward communication starts!".format(rank))
    for it in range(num_iterations):
        grad = backward_send_tensor.pop()
        logging.debug(
            "Iteration: {} send data grad, grad shape: {}".format(it, grad.shape)
        )
        output = all_gather(grad, num_workers, device)
        grad_output = torch.cat(output, 0)
        logging.debug(
            "Iteration: {} receive data grad, grad shape: {}".format(
                it, grad_output.shape
            )
        )
        backward_receive_tensor.push(grad_output)
    logging.info("Rank:{}, Backward communication has finished!".format(rank))
