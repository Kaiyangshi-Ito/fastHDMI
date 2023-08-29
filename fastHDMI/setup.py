from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import subprocess
import os

def check_cpu_flag(flag):
    try:
        result = subprocess.run(['grep', flag, '/proc/cpuinfo'], stdout=subprocess.PIPE)
        return flag in result.stdout.decode('utf-8')
    except Exception:
        return False

# Check for SIMD support
simd_supported = os.cpu_count() > 1

# Check for AVX2 and AVX512 support
avx2_supported = check_cpu_flag('avx2')
avx512_supported = check_cpu_flag('avx512f') # 'avx512f' is the foundational AVX512 support

# Configure the macros and compile/link args based on the checks
macros = []
compile_args = []
link_args = []

if simd_supported:
    compile_args.append("-msse4.2")
    link_args.append("-msse4.2")

if avx2_supported:
    macros.append(('USE_AVX2', None))
    compile_args.append("-mavx2")
    link_args.append("-mavx2")

if avx512_supported:
    macros.append(('USE_AVX512', None))
    compile_args.append("-mavx512f")
    link_args.append("-mavx512f")

# Configure the extension
extensions = [
    Extension(
        "fastHDMI.cython_fun",
        ["src/fastHDMI/cython_fun.pyx"],
        define_macros=macros,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    )
]

setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=cythonize(extensions, language_level="3"),
)
