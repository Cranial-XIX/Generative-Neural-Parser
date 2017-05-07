from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'gr',
  ext_modules = cythonize(
    "gr.pyx",
    sources=["gr.cpp"],
    language="c++",
  )
)