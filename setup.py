#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools
import sys
from setuptools import Extension
from setuptools.command.build_ext import build_ext as _build_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


class CustomBuildExt(_build_ext):
    """CustomBuildExt"""

    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy

        self.include_dirs.append(numpy.get_include())


class BuildExt(BuildExtension):
    """CustomBuildExt"""

    def finalize_options(self):
        super().finalize_options()
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy

        self.include_dirs.append(numpy.get_include())


compile_extra_args = ["-std=c++11", "-O3", "-fopenmp"]

link_extra_args = ["-fopenmp"]
if sys.platform.startswith("darwin"):
    compile_extra_args = ["-std=c++11", "-mmacosx-version-min=10.9"]
    link_extra_args = ["-stdlib=libc++", "-mmacosx-version-min=10.9"]

############################
# Check Platform
############################

assert sys.platform.startswith("linux") or sys.platform.startswith(
    "darwin"
), "Only Supported On Linux And Darwin"

# setup(
#     name="YiTu_GNN",
#     version="1.0",
#     description="A distributed GNN system",
#     author="xin ning",
#     author_email="ningxin1009@163.com",
#     packages=["YiTu_GNN"],
# )
setuptools.setup(
    name="YiTu_GNN",
    version="0.1",
    author="ning xin",
    description="A distributed GNN system",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=["scikit-learn>=0.24.2", "Cython"],
    zip_safe=False,
    cmdclass={"build_ext": BuildExt},  # "build_ext": CustomBuildExt,
    include_package_data=True,
    ext_modules=[
        Extension(
            "sample_kernel",
            ["YiTu_GNN/cpython/sample_kernel.pyx",],
            language="c++",
            extra_compile_args=compile_extra_args,
            extra_link_args=link_extra_args,
        ),
        CUDAExtension(
            name="YiTu_GNN_kernel",
            sources=["ccsrc/kernel/YiTu_GNN.cpp", "ccsrc/kernel/YiTu_GNN_kernel.cu"],
        ),
        CppExtension(
            name="rabbit",
            sources=["ccsrc/reorder/reorder.cpp"],
            extra_compile_args=["-O3", "-fopenmp", "-mcx16"],
            libraries=["numa", "tcmalloc_minimal"],
        ),
    ],
    ext_package="YiTu_GNN",
)
