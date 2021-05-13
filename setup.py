from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

compile_flags = ["-std=c++11", "-fopenmp", "-O2"]
linker_flags = ["-fopenmp"]

module = Extension("target_mean",
                   ["target_mean.pyx"],
                   language="c++",
                   include_dirs=[numpy.get_include()],
                   extra_compile_args=compile_flags,
                   extra_link_args=linker_flags)

setup(
  name="target_mean",
  ext_modules=cythonize(module)
)
