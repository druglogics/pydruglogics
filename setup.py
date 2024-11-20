from setuptools import setup, find_packages

setup(
    name="pydruglogics",
    version="0.1.7",
    author="Laura Szekeres",
    author_email="szekereslaura98@gmail.com",
    description="PyDrugLogics: a Python package designed for constructing, optimizing Boolean models and performs in-silico perturbations of the models.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/druglogics/pydruglogics",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: Unix",
    ],
    python_requires=">=3.11",
    install_requires=[
        "pygad",
        "joblib",
        "matplotlib",
        "mpbn",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn"
    ]
)