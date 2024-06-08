.PHONY: install setup all test

run:
	python3.9 main.py

install:
	pip install -r requirements.txt
	echo "from setuptools import setup, find_packages\n\nsetup(\n    name='src',\n    version='0.1',\n    packages=find_packages(),\n)" > setup.py
	pip install -e .

setup:
	echo "\nPROJECT_PATH=$(shell pwd)" >> .env

test:
	python -m unittest discover -s tests -p 'test*.py'

all: install setup