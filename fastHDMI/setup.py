from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension("src/fastHDMI.cython_fun", ["src/fastHDMI/cython_fun.pyx"])
]

setup(
    name="fastHDMI",
    version="0.1",
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    install_requires=["cython"],
)
