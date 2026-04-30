from setuptools import setup, find_packages

setup(
    name="sbi_hpm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "healpy",
        "scipy",
        "astropy",
        "torch",
        "sbi",
        "pymaster",
        "corner",
        "matplotlib",
    ],
)
