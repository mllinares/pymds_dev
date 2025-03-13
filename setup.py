from setuptools import setup, find_packages

setup(
    name="pymds",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pyro-ppl",
        "ruptures"
    ],
    author="Maureen Llinares",
    description="PyMDS - Algorithme d'inversion basé sur Pyro pour les données de chlore-36",
    url="https://github.com/mllinares/pymds_dev",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
)
