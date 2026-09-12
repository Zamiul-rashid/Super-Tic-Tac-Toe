import os
from setuptools import Extension, find_packages, setup

with open("sttt/README.md", "r") as f:
    long_description = f.read()

ext_modules = []
if os.path.exists("cpp/src/python_module.cpp") and os.path.exists("cpp/src/sttt_c_api.cpp"):
    ext_sources = ["cpp/src/python_module.cpp", "cpp/src/sttt_c_api.cpp"]
    if os.path.exists("cpp/src/mcts.cpp"):
        ext_sources.append("cpp/src/mcts.cpp")
    ext_modules.append(
        Extension(
            "sttt_cpp",
            sources=ext_sources,
            include_dirs=["cpp/include"],
            extra_compile_args=["-O3", "-march=native", "-Wall", "-Wextra", "-Werror", "-std=c++17", "-fPIC", "-pthread"],
            extra_link_args=["-pthread"],
            language="c++",
        )
    )

setup(
    name="STTT",
    version="0.9",
    description='CLI implementation of the game "Super Tic Tac Toe" written in python',
    packages=['sttt'],
    ext_modules=ext_modules,
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
    extras_require={"ai": ["torch>=2.6", "numpy>=1.24", "matplotlib>=3.10,<4"]},
)
