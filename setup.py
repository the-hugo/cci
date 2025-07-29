from setuptools import setup, find_packages

setup(
    name="cci",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy", "pandas", "spacy", "scipy", "matplotlib", "tqdm"],
)
