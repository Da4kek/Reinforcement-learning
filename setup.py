from setuptools import setup
from setuptools import find_packages

setup(name="gymrl",version="0.1",description="Reinforcement Learning Algorithms",author="DarK",
      url="https://github.com/The-DarK-os/gymrl",
      license="MIT",
      install_requires=["gym","numpy","matplotlib","tensorflow"],
      packages=find_packages())