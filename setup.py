"""Compatibility shim for tooling that still invokes setup.py directly."""

from setuptools import setup


if __name__ == "__main__":
    setup()
