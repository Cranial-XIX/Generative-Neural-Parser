from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
  name = 'parser',
  ext_modules = cythonize(
    "parser.pyx",
    language="c++",
  ),
  include_dirs=[numpy.get_include()]
)