from typing import List
from setuptools import setup, find_packages

HYPHEN_E_DOT = "-e ." 

def get_requirement(filePath: str) -> list[str]:
    requirements = []
    with open(filePath) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", " ") for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

        return requirements
 


setup(
    name= "Marks Predictor",
    version= "0.0.1",
    author= "Khizar Abbasi",
    author_email="Khierabbasi144@gmail.com",
    packages= find_packages(),
    install_requires = get_requirement("requirements.txt")
)