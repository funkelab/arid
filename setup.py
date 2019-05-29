from Cython.Build import cythonize
from distutils.sysconfig import get_python_inc, get_config_var
from setuptools import setup
from setuptools.extension import Extension
import numpy as np
import os

impl_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'arid',
    'impl')

include_dirs = [
    impl_dir,
    os.path.dirname(get_python_inc()),
    get_python_inc(),
    np.get_include()
]

library_dirs = [
    impl_dir,
    get_config_var("LIBDIR")
]

setup(
        name='arid',
        version='0.1',
        description='Ultra-Metric Learning.',
        url='https://github.com/funkelab/arid',
        author='Jan Funke',
        author_email='jfunke@iri.upc.edu',
        license='MIT',
        packages=[
            'arid',
            'arid.impl'
        ],
        ext_modules=cythonize([
            Extension(
                'arid.impl.wrappers',
                sources=[
                    'arid/impl/wrappers.pyx',
                    'arid/impl/connected_components.cpp'
                ],
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                extra_link_args=['-std=c++11'],
                extra_compile_args=['-O3', '-std=c++11'],
                language='c++')
        ])
)
