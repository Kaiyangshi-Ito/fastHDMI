from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

extensions = [
    Extension("fastHDMI.cython_fun", ["src/fastHDMI/cython_fun.pyx"]),
]

setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=cythonize(extensions),
)
