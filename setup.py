from setuptools import find_packages, setup

with open("sttt/README.md", "r") as f:
    long_description = f.read()

setup(
    name="STTT",
    version="0.8",
    description='CLI implementation of the game "Super Tic Tac Toe" written in python',
    packages=['sttt'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Zamiul-rashid/Super-Tic-Tac-Toe",
    author="Zamiul, Proyas",
    author_email="zamiulrashid1@gmail.com,abyashrirproyas@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'sttt = sttt.__main__:playgame'
    ]
    },
    python_requires=">=3.10",
)