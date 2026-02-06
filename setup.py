"""
Setup configuration for ToTf package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ToTf",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Cross-Library Compatible Library for PyTorch and TensorFlow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ToTf",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "tqdm>=4.65.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "torch": ["torch>=2.0.0"],
        "tensorflow": ["tensorflow>=2.13.0"],
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
)
