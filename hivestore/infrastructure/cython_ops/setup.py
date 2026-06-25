from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "hive_ops",
        ["hive_ops.pyx"],
        include_dirs=[np.get_include()]
    )
]

setup(
    name="hive_ops",
    ext_modules=cythonize(extensions, language_level=3)
)
