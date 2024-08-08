from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as file:
    requirements = [line.strip() for line in file if line.strip()]

setup(
    name="meros",
    version="0.0.3",
    packages=find_packages(),
    license="BSD-3-Clause",
    install_requires=requirements,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ryan Sfeila",
    author_email="ryansfeila@gmail.com",
    url="https://github.com/sfeilaryan/MeROS",
)
