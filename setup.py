import re
from os import path

from setuptools import setup

# read the contents of README file
this_directory = path.abspath(path.dirname(__file__))
# with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
#     long_description = f.read()

setup(
    name="diffusion-displacement-map",
    python_requires=">=3.6",
    packages=[
        "diffusion_displacement_map",
    ],
    install_requires=[
        "creativeai==0.1.1",
        "docopt",
        "tqdm",
        "schema",
    ],
    entry_points={
        "console_scripts": [
            "diffusion_displacement_map=diffusion_displacement_map.__main__:main",
        ]
    },
)
