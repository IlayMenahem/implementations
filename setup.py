from setuptools import setup, find_packages

setup(
    name="implementations",
    version="0.1.0",
    author="ilay menahem",
    author_email="ilay.menachem1@gmail.com",
    description="implementations of useful algorithms",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ilaymenahem/implementations",
    packages=find_packages(),

    install_requires=[
        'gymnasium',
        'tqdm',
        'numpy',
        'matplotlib',
        'torch',
    ],

    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],

    python_requires=">=3.10",
)
