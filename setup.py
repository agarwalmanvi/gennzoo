import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gennzoo",
    version="0.0.1",
    author="Manvi Agarwal",
    author_email="m.agarwal.1@student.rug.nl",
    description="Model Zoo for GeNN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/agarwalmanvi/gennzoo",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
