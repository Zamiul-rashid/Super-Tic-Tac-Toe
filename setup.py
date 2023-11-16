from setuptools import find_packages, setup

with open("app/README.md", "r") as f:
    long_description = f.read()

setup(
    name="Super-Tic-Tac-Toe",
    version="1.0",
    description="Python console-based game is an implementation of Super Tic Tac Toe.",
    package_dir={"": "app"},
    packages=find_packages(where="app"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Zamiul-rashid/Super-Tic-Tac-Toe",
    author="Zamiul,Proyash",
    author_email="zamiulrashid1@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=["bson >= 0.5.10"],
    extras_require={
        "dev": ["twine>=4.0.2"],
    },
    python_requires=">=3.10",
)