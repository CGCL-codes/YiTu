from collections import deque

import torch.optim


class OptimizerWithWeightStashing(torch.optim.Optimizer):
    """Wrapper class that adds weight stashing to a vanilla torch.optim.Optimizer.

    Arguments:
        - optim_name: the name of optimizer, required to create the corresponding
                      base_optimizer (torch.optim.{optim_name}).
        - optimizer_args: the keyword arguments passed to base_optimizer.
    """

    def __init__(self, optim_name, model, num_versions, **optimizer_args):
        child_modules = [child for child in model.children()]
        assert (
            len(child_modules) == 2
        ), "just need 2 modules: model module and data module"
        # 模型并行和数据并行的两个模块
        self.model_module = child_modules[0]
        self.data_module = child_modules[1]
        self.num_versions = num_versions
        # 复用pytorch的优化器
        # self.base_optimizer = getattr(torch.optim, optim_name)(
        #     model.parameters(), **optimizer_args
        # )
        self.model_optim = getattr(torch.optim, optim_name)(
            self.model_module.parameters(), **optimizer_args
        )
        self.data_optim = getattr(torch.optim, optim_name)(
            self.data_module.parameters(), **optimizer_args
        )
        self.initialize_queue()

    def __getattr__(self, key):
        """Relay the unknown key to base_optimizer."""
        return getattr(self.base_optimizer, key)

    def initialize_queue(self):
        # 使用两个队列来分别保存模型并行和数据并行的各个版本的参数
        # 最新版本的参数放在最后面
        self.model_queue = deque(maxlen=self.num_versions)
        self.data_queue = deque(maxlen=self.num_versions)
        for _ in range(self.num_versions):
            self.model_queue.append(self.get_model_params())
            self.data_queue.append(self.get_data_params())

    def get_model_params(self, clone=True):
        if clone:
            state_dict = self.model_module.state_dict()
            for key in state_dict:
                # 拷贝参数
                state_dict[key] = state_dict[key].clone()
        else:
            state_dict = self.model_module.state_dict()
        return state_dict

    def get_data_params(self, clone=True):
        if clone:
            state_dict = self.data_module.state_dict()
            for key in state_dict:
                # 拷贝参数
                state_dict[key] = state_dict[key].clone()
        else:
            state_dict = self.data_module.state_dict()
        return state_dict

    def set_data_params(self, state_dict):
        self.data_module.load_state_dict(state_dict)
        # for name, param in self.data_module.named_parameters():
        # param.data = state_dict[name]

    def set_model_params(self, state_dict):
        self.model_module.load_state_dict(state_dict)
        # for name, param in self.model_module.named_parameters():
        # param.data = state_dict[name]

    def load_data_old_params(self):
        """加载反向传播在数据并行阶段需要的参数, 只修改参数不修改梯度"""
        self.set_data_params(self.data_queue[0])

    def load_model_old_params(self):
        """加载反向传播在模型并行阶段需要的参数"""
        self.set_model_params(self.model_queue[0])

    def load_new_params(self):
        self.set_data_params(self.data_queue[-1])
        self.set_model_params(self.model_queue[-1])

    def load_model_new_params(self):
        # 只覆盖模型的参数而不覆盖梯度
        # self.set_data_params(self.data_queue[-1])
        self.set_model_params(self.model_queue[-1])

    def load_data_new_params(self):
        self.set_data_params(self.data_queue[-1])
        # self.set_model_params(self.model_queue[-1])

    def zero_grad(self):
        # self.base_optimizer.zero_grad()
        self.data_optim.zero_grad()
        self.model_optim.zero_grad()
        pass

    def data_zero_grad(self):
        self.data_optim.zero_grad()
        pass

    def model_zero_grad(self):
        self.model_optim.zero_grad()
        pass

    def data_step(self):
        self.data_optim.step()
        self.data_queue.append(self.get_data_params(clone=True))
        pass

    def model_step(self):
        self.model_optim.step()
        self.model_queue.append(self.get_model_params(clone=True))
        pass

    def step(self):
        """Performs a single optimization step."""
        # self.base_optimizer.step()
        # self.data_queue.append(self.get_data_params(clone=True))
        # self.model_queue.append(self.get_model_params(clone=True))
        self.data_optim.step()
        self.model_optim.step()
        pass


class AdamWithWeightStashing(OptimizerWithWeightStashing):
    """
    Adam optimizer with weight stashing.
    """

    def __init__(
        self,
        model,
        num_versions,
        lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
    ):
        super(AdamWithWeightStashing, self).__init__(
            optim_name="Adam",
            model=model,
            num_versions=num_versions,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
