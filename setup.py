import warnings

# Silence the Loky cores warning
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

# Silence the LightGBM feature-name warning
warnings.filterwarnings("ignore", message=".*valid feature names.*")

from setuptools import setup, find_packages, setup
from typing import List

Hyperlink = "-e ."


def get_requirements(file_path: str) -> list[str]:
    """
    Reads a requirements file and returns a list of packages.
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if Hyperlink in requirements:
            requirements.remove(Hyperlink)

    return requirements


setup(
    name="reg_housing_price_predictor",
    version="0.0.1",
    author="Abdiwahid Ali",
    author_email="maqbuul@outlook.com",
    description="A collection of Pydantic extensions for enhanced functionality.",
    packages=find_packages(exclude=["tests"]),
    install_requires=get_requirements("requirements.txt"),
)
