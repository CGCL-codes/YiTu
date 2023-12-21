import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from YiTu_GNN.communication import all_gather
from YiTu_GNN.communication import all_gather_tensors


def all_gather_gpu_test(rank, world_size):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    length = [(i + 1) * 2 for i in range(world_size)]
    # data = torch.tensor([length[rank]], dtype=torch.int32).to('cuda')
    # tensor_list = [torch.zeros(1, dtype=torch.int32).to('cuda') for _ in range(world_size)]
    # dist.all_gather(tensor_list, data)
    # print("rank:{}, tensor list{}".format(rank, tensor_list))
    data = torch.rand(length[rank]).to("cuda")
    print("rank:{}, data{}".format(rank, data))
    # lists = [torch.zeros(l.item()) for l in tensor_list]
    # lists = [torch.zeros(2 * world_size) for _ in range(world_size)]
    # print("rank:{}, data:{}, tensor list: {}".format(rank, data, lists))
    lists = all_gather(data, world_size)
    print("rank:{}, length:{}, tensor_list: {}".format(rank, length[rank], lists))

def all_gather_cpu_test(rank, world_size, data):
    device = torch.device('cpu')
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    result = all_gather(data[rank], world_size, device)
    for i in range(len(data)):
        assert torch.equal(data[i], result[i])

def main():
    world_size = 2
    for i in range(100):
        data1 = torch.rand(100000, 2)
        data2 = torch.rand(100000, 3)
        data = [data1, data2]
        mp.spawn(all_gather_cpu_test,
            args=(world_size, data),
            nprocs=world_size,
            join=True)

if __name__=="__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()
    print("finished!")
