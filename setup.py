from setuptools import setup, find_packages

setup(
    name="CTDFjorder",
    version="0.0.34",
    author="Nikolas Yanek-Chrones",
    author_email="nikojb1001@gmail.com",
    description="A package for processing and analyzing CTD data.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nikothomas/CTDFjorder",
    project_urls={
        "Homepage": "https://github.com/nikothomas/CTDFjorder",
        "Issues": "https://github.com/nikothomas/CTDFjorder/issues"
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords=["CTD"],
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.26.4",
        "pandas>=2.2.1",
        "gsw>=3.6.17",
        "matplotlib>=3.8.4",
        "statsmodels>=0.14.0",
        "tabulate>=0.9.0",
        "pyrsktools==0.1.9",
        "openpyxl>=3.1.2",
        "setuptools"
    ],
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'ctdfjorder-cli=CTDFjorder.CTDFjorder:main',
        ],
    },
)
