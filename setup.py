#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

requirements = ["jax", "funsor", "numpyro", "flowjax", "numba", "matplotlib", "seaborn"]

setup(
    name="rnpe",
    author="Daniel Ward",
    author_email="danielward27@outook.com",
    python_requires=">=3.9",
    install_requires=requirements,
    license="MIT license",
    include_package_data=True,
    packages=find_packages(),
    url="https://github.com/danielward27/rnpe",
    zip_safe=False,
)
