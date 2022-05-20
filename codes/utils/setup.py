from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext_modules = [Extension('ksmallest', ['ksmallest.pyx'], language='c++')]
setup(name='ksmallest', cmdclass={'build_ext': build_ext}, ext_modules=ext_modules)
