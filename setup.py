#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="live-nst",
    version="0.0.1",
    description="Scientific project in the field of neural style transfer. It is coded for FOKA 2022 at the GUT.",
    author="Gradient PG",
    author_email="gradientpg@gmail.com",
    url="https://github.com/Gradient-PG/live-nst",
    install_requires=["pytorch-lightning"],
    packages=find_packages(),
)
