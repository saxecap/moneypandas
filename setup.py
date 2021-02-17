from setuptools import setup, find_packages
from os import path
from codecs import open

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

install_requires = [
    "pandas", # for testing "pandas<1.0.0",
    "numpy", # for testing "numpy<1.20.0",
    "six",
    "iso4217parse",
    "money"
]
tests_require = install_requires + [
    "pytest",
    "hypothesis"
]

setup(
    name='moneypandas',
    version='0.9.5',
    setup_requires=['setuptools_scm'],
    description='Money type for pandas',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/flaxandteal/moneypandas',
    author='Phil Weir (moneypandas tweaks), Tom Augspurger (cyberpandas)',
    author_email='phil.weir@flaxandteal.co.uk',
    license="BSD",
    classifiers=[  # Optional
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    packages=find_packages(),
    install_requires=install_requires,
    tests_require=tests_require
)
