from setuptools import setup, find_packages

setup(
    name="ctdfjorder",
    version="0.1.6",
    author="Nikolas Yanek-Chrones",
    author_email="research@icarai.io",
    description="A package for processing and analyzing CTD data.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nikothomas/ctdfjorder",
    project_urls={
        "Homepage": "https://github.com/nikothomas/ctdfjorder",
        "Issues": "https://github.com/nikothomas/ctdfjorder/issues",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords=["CTD"],
    python_requires=">=3.11",
    install_requires=[
        "polars~=1.1.0",
        "psutil~=6.0.0",
        "enlighten~=1.12.4",
        "pandas~=2.2.2",
        "setuptools~=70.3.0",
        "openpyxl~=3.1.4",
        "tensorflow",
        "numpy~=1.26.4",
        "gsw~=3.6.18",
        "matplotlib~=3.9.1",
        "statsmodels~=0.14.2",
        "keras~=3.4.1",
        "scikit-learn~=1.5.1",
        "pyrsktools==0.1.9",
        "colorlog~=6.8.2",
        "pyarrow~=17.0.0",
        "fastexcel~=0.10.4"
    ],
    packages=find_packages("."),
    entry_points={
        "console_scripts": [
            "ctdfjorder-cli=cli:main",
        ],
    },
)
