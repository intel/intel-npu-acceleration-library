#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as build_ext_orig
import pathlib
import glob
import os
import re


def get_version():
    this_file_path = os.path.dirname(os.path.abspath(__file__))
    with open(
        os.path.join(this_file_path, "intel_npu_acceleration_library", "_version.py"),
        "rt",
    ) as fp:
        verstrline = fp.read()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    mo = re.search(VSRE, verstrline, re.M)
    if mo:
        verstr = mo.group(1)
    else:
        raise RuntimeError("Unable to find version string")
    return verstr


class CMakeExtension(Extension):
    def __init__(self, name):
        # don't invoke the original build_ext for this special extension
        headers = glob.glob("include/**/*.h")
        cpp_sources = glob.glob("src/*.cpp")
        requirements = glob.glob("*requirements.txt")
        sources = ["CMakeLists.txt"] + requirements + cpp_sources + headers
        super().__init__(name, sources=sources)


class build_ext(build_ext_orig):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        cwd = pathlib.Path().absolute()

        # these dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing
        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        extdir.mkdir(parents=True, exist_ok=True)

        # example of cmake args
        config = "Debug" if self.debug else "Release"
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={os.path.join(extdir.parent.absolute(), ext.name, "lib")}',
            "-DCMAKE_BUILD_TYPE=" + config,
        ]

        # example of build args
        build_args = [
            "--config",
            config,
        ]

        os.chdir(str(build_temp))
        self.spawn(["cmake", str(cwd)] + cmake_args)
        if not self.dry_run:
            self.spawn(["cmake", "--build", "."] + build_args)
        # Troubleshooting: if fail on line above then delete all possible
        # temporary CMake files including "CMakeCache.txt" in top level dir.
        os.chdir(str(cwd))


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as fh:
    requirements = fh.readlines()

with open("dev_requirements.txt") as fh:
    dev_requirements = fh.readlines()

setup(
    name="intel_npu_acceleration_library",
    version=get_version(),
    packages=[
        "intel_npu_acceleration_library",
        "intel_npu_acceleration_library.backend",
        "intel_npu_acceleration_library.nn",
        "intel_npu_acceleration_library.functional",
    ],
    author="Alessandro Palla",
    author_email="alessandro.palla@intel.com",
    description="Intel® NPU Acceleration Library",
    license="Apache License 2.0",
    url="https://github.com/intel/intel-npu-acceleration-library",
    ext_modules=[CMakeExtension("intel_npu_acceleration_library")],
    cmdclass={
        "build_ext": build_ext,
    },
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="intel-npu-acceleration-library, machine learning, llm, intel core ultra",
)
