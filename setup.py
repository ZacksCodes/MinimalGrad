import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="minimalgrad",
    version="0.1.0",
    author="Zakaria Salmi",
    author_email="q60lw0@inf.elte.hu",
    description="A minimal autograd engine.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ZacksCodes/MinimalGrad",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)