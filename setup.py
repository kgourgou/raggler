from setuptools import find_packages, setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="raggler",
    packages=find_packages(),
    version="0.1.0",
    description="A simple RAG package.",
    author="Kostis Gourgoulias",
    license="MIT",
    install_requires=required,
)
