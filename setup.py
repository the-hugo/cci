from setuptools import setup, find_packages

setup(
    name="cci",
    version="0.1.0",
    description="Creative Convergence Index (CCI) - Measure collaborative discourse dynamics",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas", 
        "spacy>=3.4.0",
        "scipy",
        "matplotlib",
        "tqdm",
        "sentence-transformers>=2.0.0"
    ],
    extras_require={
        "coref": ["spacy-coref"]  # Optional coreference resolution
    },
    python_requires=">=3.9",
    author="CCI Research Team",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
