#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name = "pyhive",
    version = "0.1",
    packages = find_packages(),
    scripts = [],
    install_requires = [
            'aio-hs2>=0.2',
            'pandas>=0.13',
            'docutils>=0.3',
            ],
    package_data = {
        '': ['*.txt', '*.rst'],
    },

    # metadata for upload to PyPI
    author = "wabu",
    author_email = "daniel.waeber@gameduell.de",
    description = "asyncio based hive client for python",
    license = "MIT",
    keywords = "python hive hadoop ayncio aio-hs2",
    url = "https://github.com/wabu/pyhive",
)
