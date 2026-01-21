from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="soft-computing",
    version="1.0.0",
    author="Soft Computing Team",
    description="A comprehensive Python-based soft computing project integrating Neural Networks, Fuzzy Logic, and Genetic Algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=1.0.0",
        "tensorflow>=2.8.0",
        "pandas>=1.3.0",
        "seaborn>=0.11.0",
        "scikit-fuzzy>=0.4.2",
    ],
)
