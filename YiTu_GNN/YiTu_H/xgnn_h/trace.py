import torch


class Tracer(torch.Tensor):
    trace_begin = False
    previous_func = None
    previous_args = []
    previous_kwargs = {}

    @staticmethod
    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls, *args, **kwargs)
        return self

    def __repr__(self):
        return 'Tracer {}'.format(super().__repr__())

    def _set_previous(self, func_name, args, kwargs):
        self.previous_func = func_name
        if args is not None:
            self.previous_args = args
        if kwargs is not None:
            self.previous_kwargs = kwargs

    @classmethod
    def __torch_function__(cls, func, types, args, kwargs=None):
        ret = super().__torch_function__(func, types, args, kwargs)
        if isinstance(ret, Tracer):
            ret._set_previous(
                func.__name__, args, kwargs
            )
        return ret

    def to(self, *args, **kwargs):
        new_obj = Tracer()
        tempTensor = super().to(*args, **kwargs)
        new_obj.data = tempTensor.data
        new_obj.requires_grad = tempTensor.requires_grad
        return new_obj
