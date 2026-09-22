import os
import subprocess
import sysconfig
from datetime import datetime, timezone
from setuptools import Extension, find_packages, setup

with open("sttt/README.md", "r") as f:
    long_description = f.read()

def get_git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"

git_rev = get_git_rev()
build_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{git_rev}"
version = "1.1.0"

cflags = ["-O3", "-march=native", "-Wall", "-Wextra", "-Werror", "-std=c++17", "-fPIC", "-pthread"]
py_include = sysconfig.get_path("include")
defines = [
    f"-DSTTT_CPP_VERSION=\"{version}\"",
    f"-DSTTT_BUILD_ID=\"{build_id}\"",
    f"-DSTTT_SOURCE_REVISION=\"{git_rev}\"",
    f"-DSTTT_COMPILER_FLAGS=\"{' '.join(cflags)}\"",
]

ext_modules = []
if os.path.exists("cpp/src/python_module.cpp") and os.path.exists("cpp/src/sttt_c_api.cpp"):
    ext_sources = ["cpp/src/python_module.cpp", "cpp/src/sttt_c_api.cpp"]
    if os.path.exists("cpp/src/mcts.cpp"):
        ext_sources.append("cpp/src/mcts.cpp")
    include_dirs = ["cpp/include"]
    if py_include and py_include not in include_dirs:
        include_dirs.append(py_include)
    ext_modules.append(
        Extension(
            "sttt_cpp",
            sources=ext_sources,
            include_dirs=include_dirs,
            extra_compile_args=cflags + defines,
            extra_link_args=["-pthread"],
            language="c++",
        )
    )

setup(
    name="STTT",
    version="1.1.0",
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
