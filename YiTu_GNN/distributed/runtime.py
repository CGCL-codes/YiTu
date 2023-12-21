import logging

import torch

from . import communication
from .optimizer import OptimizerWithWeightStashing


class StageRuntime:
    """分阶段执行运行时, 将一个GNN模型拆分成模型并行和数据并行两个阶段, 能够很好地实现流水线机制.

    Examples:
        # 1. 构造dataloader, 模型, 损失函数
        dataloader = ...
        model = ...
        loss_fcn = ...
        # 2. 利用YiTu_GNN api构造能够支持多个版本参数的优化器
        optim = optimizer.AdamWithWeightStashing(model, param_version, args.lr)
        # 3. 构造运行时
        runtime = StageRuntime(
            model, rank, world_size, device, optim, loss_fcn, num_warmup=num_warmup
        )
        # 4. 迭代训练
        for epoch in range(num_epochs):
            num_iter = (train_nid.shape[-1] + args.batch_size) // args.batch_size
            # 4.1 调用train()初始化训练阶段的一些数据: 迭代次数和dataloader
            runtime.train(num_iter, dataloader)
            # 4.2 warmup阶段: 在正式训练前启动几次模型并行的前向传播, 从而能够实现流水线机制
            runtime.forward()
            for it in range(num_iter):
                # 4.3 执行前向传播
                output, loss, target = runtime.forward()
                # 4.4 执行反向传播并优化参数
                runtime.backward_and_step()
    """

    def __init__(self, model, rank, num_workers, device, optim, loss_fn, num_warmup=2):
        self.model = model
        self.num_workers = num_workers
        self.rank = rank
        # 保存模型并行阶段的图结构，以用于数据并行
        self.blocks = []
        # 模型并行阶段的计算结果：(tensor, nodes_offset), 用于发送给数据并行阶段以及反向传播阶段
        self.model_result_tensors = []
        self.gradients = []
        self.device = device
        self.comm_handler = communication.Communication(rank, num_workers, device)
        assert isinstance(
            optim, OptimizerWithWeightStashing
        ), "optim must inherit OptimizerWithWeightStashing"
        self.optim = optim
        self.loss_fn = loss_fn
        self.num_warmup = num_warmup
        self.forward_count = 0
        # 比如label等数据，用于计算loss
        self.targets = []
        # 用于在forward阶段将target输出暴露给用户
        self.output_targets = []
        # 每次迭代的loss
        self.losses = []
        self.num_iter = 0
        self.outputs = []
        self._eval = False
        self.dataloader = None
        self.data_iter = None

    def zero_grad(self):
        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i].zero_grad()

    def train(self, num_iterations, dataloader):
        """用于初始化训练阶段的一些数据: 迭代次数, 数据加载器等.
        注意: 每轮epoch之前都需要调用该API用于初始化.
        """
        self.blocks = []
        self.model_result_tensors = []
        self.gradients = []
        # 启动发送和接收数据的线程，num_iterations为迭代的次数也就是send和recv的调用次数
        logging.info(
            "Rank {}, train stage starts communication thread, num_iterations: {}".format(
                self.rank, num_iterations
            )
        )
        if self.comm_handler is not None:
            self.comm_handler.start_comm_thread(num_iterations)
        self.model.train()
        self.num_iter = num_iterations
        self._eval = False
        self.dataloader = dataloader
        self.data_iter = iter(self.dataloader)

    def eval(self, num_iterations, dataloader):
        """用于初始化模型评估阶段的一些数据: 迭代次数, 数据加载器等.
        注意: 每轮epoch之前都需要调用该API用于初始化.
        """
        self.blocks = []
        self.model_result_tensors = []
        self.gradients = []
        logging.info(
            "Rank {}, eval stage starts communication thread, num_iterations: {}".format(
                self.rank, num_iterations
            )
        )
        if self.comm_handler is not None:
            self.comm_handler.start_eval_comm_thread(num_iterations)
        self.model.eval()
        self.num_iter = num_iterations
        self._eval = True
        self.dataloader = dataloader
        self.data_iter = iter(self.dataloader)

    def _receive_tensors_forward(self):
        output = self.comm_handler.receive()
        return output

    def _send_tensors_forward(self):
        self.comm_handler.send(self.model_result_tensors[-1])

    def _receive_tensors_backward(self):
        output = self.comm_handler.receive(forward=False)
        return output

    def _send_tensors_backward(self, grad):
        self.comm_handler.send(grad, forward=False)

    def _run_model_forward(self, blocks, x):
        """Run forward pass."""
        # 1. model forward
        # 2. all gather tensor
        # 反向传播时需要对feats执行操作，还需要保存该结果
        feats = self.model.model_forward(blocks, x)
        # append操作无法执行backward操作，将导致计算图中断，所以将list结果进行concat
        feats = torch.cat(feats, 0)
        nodes_offset = [0]
        for i in range(self.num_workers):
            nodes_offset.append(nodes_offset[-1] + len(blocks[i].dstnodes()))
        self.model_result_tensors.append((feats, nodes_offset))
        # 需要缓存图结构信息
        self.blocks.append(blocks[self.num_workers :])
        # 发送模型并行阶段的计算结果
        self._send_tensors_forward()

    def _run_data_forward(self):
        # 1. get tensor and blocks
        # 2. data forward
        feats = self._receive_tensors_forward()
        blocks = self.blocks.pop(0)
        feats.require_grad = True
        # assert feats.is_leaf, "features of data forward must be a leaf node"
        output = self.model.data_forward(blocks, feats)
        self.leaf_node = feats
        return output

    def _run_data_backward(self, loss):
        feats_grad = {}

        def save_grad(name):
            def hook(grad):
                feats_grad[name] = grad
                # print("features grad shape: {}".format(grad.shape))

            return hook

        feats = self.leaf_node
        feats.register_hook(save_grad("feats"))
        # 1. data backward
        # 2. all gather gradient
        # 设置retain_graph=True以避免释放计算图, 因为模型并行阶段的反向传播还需要使用该计算图
        torch.autograd.backward(loss, retain_graph=True)

        assert feats_grad["feats"] is not None, "feats grad is None"
        assert (
            feats_grad["feats"].shape == feats.shape
        ), "feats grad shape: {} is not equal to feats shape: {}".format(
            feats_grad["feats"].shape, feats.shape
        )
        # 发送梯度
        self._send_tensors_backward(feats_grad["feats"])

    def _run_model_backward(self):
        # 1. 获取输出梯度以及需要求梯度的叶子节点
        # 2. model backward
        grad = self._receive_tensors_backward()
        feats, _ = self.model_result_tensors.pop(0)
        logging.debug("run model backward, feats shape: {}".format(feats.shape))
        assert (
            grad.shape == feats.shape
        ), "grad.shape {} is not equal to feats.shape {}".format(
            grad.shape, feats.shape
        )
        # 设置retain_graph=False以释放掉计算图
        torch.autograd.backward(feats, grad_tensors=grad, retain_graph=False)

    def _data_backward_and_step(self, loss):
        self.optim.load_data_old_params()
        self.optim.data_zero_grad()
        self._run_data_backward(loss)
        self.optim.load_data_new_params()
        self.optim.data_step()

    def _model_backward_and_step(self):
        self.optim.load_model_old_params()
        self.optim.model_zero_grad()
        self._run_model_backward()
        self.optim.load_model_new_params()
        self.optim.model_step()

    def _get_data(self):
        try:
            data, target = next(self.data_iter)
            self.targets.append(target)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            data, target = next(self.data_iter)
            self.targets.append(target)
        return data

    def _get_target(self):
        return self.targets.pop(0)

    def _eval_forward(self):
        data = self._get_data()
        self._run_model_forward(*data)
        output = self._run_data_forward()
        target = self._get_target()
        loss = self.loss_fn(output, target)
        return output, loss, target

    def _pipeline_forward(self):
        res = None
        # 迭代num_iter+1次，最后一次啥也不执行
        # warmup阶段: 先执行num_warmup次forward操作，从而实现流水线机制
        if self.forward_count == 0:
            for _ in range(self.num_warmup):
                data = self._get_data()
                self._run_model_forward(*data)
            output = self._run_data_forward()
            self.outputs.append(output)
            # 计算loss
            target = self._get_target()
            self.output_targets.append(target)
            loss = self.loss_fn(output, target)
            self.losses.append(loss)
            self._data_backward_and_step(loss)
        elif (
            self.forward_count <= self.num_iter - 1
            and self.forward_count > self.num_iter - self.num_warmup
        ):
            output = self._run_data_forward()
            self.outputs.append(output)
            # 计算loss
            target = self._get_target()
            self.output_targets.append(target)
            loss = self.loss_fn(output, target)
            self.losses.append(loss)
            self._data_backward_and_step(loss)
            res = (self.outputs.pop(0), self.losses.pop(0), self.output_targets.pop(0))
        elif self.forward_count == self.num_iter:
            res = (self.outputs.pop(0), self.losses.pop(0), self.output_targets.pop(0))
        else:
            data = self._get_data()
            self._run_model_forward(*data)
            output = self._run_data_forward()
            self.outputs.append(output)
            # 计算loss
            target = self._get_target()
            self.output_targets.append(target)
            loss = self.loss_fn(output, target)
            self.losses.append(loss)
            self._data_backward_and_step(loss)
            res = (self.outputs.pop(0), self.losses.pop(0), self.output_targets.pop(0))
        return res

    def _train_forward(self):
        res = None
        # 串行执行
        if self.num_warmup <= 0:
            # 为了和流水线执行的API统一, warmup阶段啥也不执行
            if self.forward_count == 0:
                res = None
            else:
                data = self._get_data()
                self._run_model_forward(*data)
                output = self._run_data_forward()
                target = self._get_target()
                loss = self.loss_fn(output, target)
                self._data_backward_and_step(loss)
                res = (output, loss, target)
        # 流水线
        else:
            res = self._pipeline_forward()
        self.forward_count += 1
        if self.forward_count > self.num_iter:
            self.forward_count = 0
        return res

    def forward(self):
        """forward操作
        注意: 在训练阶段, forward api并不是单纯的执行前向传播操作, 还会执行数据并行阶段的反向传播.

        如下所示, num_warmup = 2, num_iter = 3, 执行过程如下, MF: 模型并行前向传播, DF: 数据并行前向传播, DB: 数据并行反向传播
        MB: 模型并行反向传播, iF表示第i次迭代调用forward()的执行操作
        iB表示第i次迭代调用backward()的执行操作。

        examples:

        1MF 2MF 1DF 1DB | 3MF 2DF 2DB | 1MB | 3DF 3DB | 2MB | empty | 3MB
        |     0F        |      1F     | 1B  |    2F   |  2B |   3F  | 3B
        """
        if self._eval:
            return self._eval_forward()
        else:
            return self._train_forward()

    def backward_and_step(self):
        """执行反向传播
        注意: 这里的backward api只执行模型并行的反向传播, 其中数据并行的反向传播在forward()中执行。

        如下所示, num_warmup = 2, num_iter = 3, 执行过程如下, MF: 模型并行前向传播, DF: 数据并行前向传播, DB: 数据并行反向传播
        MB: 模型并行反向传播, iF表示第i次迭代调用forward()的执行操作
        iB表示第i次迭代调用backward()的执行操作。

        examples:

        1MF 2MF 1DF 1DB | 3MF 2DF 2DB | 1MB | 3DF 3DB | 2MB | empty | 3MB
        |     0F        |      1F     | 1B  |    2F   |  2B |   3F  | 3B
        """
        self._model_backward_and_step()
