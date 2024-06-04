.PHONY: install setup all

install:
	pip install -r requirements.txt
	echo "from setuptools import setup, find_packages\n\nsetup(\n    name='src',\n    version='0.1',\n    packages=find_packages(),\n)" > setup.py
	pip install -e .

setup:
	echo "\nPROJECT_PATH=$(shell pwd)" >> .env

all: install setup