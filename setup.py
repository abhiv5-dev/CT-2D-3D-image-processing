import os
import sys
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "ct",
        [
            "python/bindings.cpp",
            "src/image2d.cpp",
            "src/volume3d.cpp",
            "src/tiff_io.cpp",
            "src/preview.cpp",
            "src/filters/gaussian.cpp",
            "src/filters/median.cpp",
            "src/filters/bilateral.cpp",
            "src/filters/diffusion.cpp",
            "src/filters/nlm.cpp",
            "src/intensity.cpp",
        ],
        include_dirs=["include", "/opt/homebrew/include"],
        library_dirs=["/opt/homebrew/lib"],
        libraries=["tiff"],
        extra_compile_args=["-std=c++17", "-O3"],
    ),
]

setup(
    name="ct",
    version="1.0.0",
    author="Abhishek",
    description="2D/3D CT Image Processing and Filtering library",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    packages=["ct_eval"],
    package_dir={"": "python"},
)
