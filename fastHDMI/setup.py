from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import subprocess

# Check for AVX2 support
def has_avx2_support():
    try:
        result = subprocess.run(['grep', 'avx2', '/proc/cpuinfo'], stdout=subprocess.PIPE)
        return 'avx2' in result.stdout.decode('utf-8')
    except Exception:
        return False

avx2_supported = has_avx2_support()

# Configure the extension
extensions = [
    Extension(
        "fastHDMI.cython_fun",
        ["src/fastHDMI/cython_fun.pyx"],
        define_macros=[('USE_AVX2', None)] if avx2_supported else [],
        extra_compile_args=["-mavx2"] if avx2_supported else [],
        extra_link_args=["-mavx2"] if avx2_supported else [],
    )
]

setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=cythonize(extensions, language_level="3"),
)
