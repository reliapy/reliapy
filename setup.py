#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='reliapy',
    version="1.0.0",
    url='https://github.com/reliapy/reliapy',
    description="Structural Risk and Reliability with Python.",
    author="Ketson R. M. dos Santos",
    author_email="ketson.santos@epfl.ch",
    license='BSD',
    platforms=["OSX", "Windows", "Linux"],
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"": ["*.pdf"]},
    install_requires=[
        "numpy", "scipy", "matplotlib", "scikit-learn"
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
    ],
)
