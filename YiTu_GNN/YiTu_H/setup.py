from pytest import ExitCode
from setuptools import setup, Command
from torch.utils import cpp_extension


class TestCommand(Command):
    description = "test"
    user_options = [
        ('run-forever', 'f', 'run-forever')
    ]

    def initialize_options(self):
        self.run_forever = None

    def finalize_options(self):
        pass

    def run(self):
        import pytest
        max_epoch = 1
        if self.run_forever:
            max_epoch = 0x7fffffff
        for epoch in range(max_epoch):
            print('epoch:', epoch)
            ret = pytest.main(['-x', 'test'])
            if ret != ExitCode.OK:
                break


setup(
    name='graph_ext',
    ext_modules=[
        cpp_extension.CUDAExtension(
            'graph_ext',
            sources=[
                'src/entry.cpp',
                'src/fc_cuda.cu',
                'src/spmm_cuda.cu',
                'src/sddmm_cuda.cu',
                'src/hfused_cuda.cu',
            ]
        )
    ],
    cmdclass={
        'test': TestCommand,
        'build_ext': cpp_extension.BuildExtension,
    }
)
