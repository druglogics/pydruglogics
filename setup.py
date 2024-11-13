from setuptools import setup, find_packages

setup(
    name="pydruglogics",
    version="0.1.0",
    author="Laura Szekeres",
    author_email="szekereslaura98@gmail.com",
    description="PyDrugLogics: a Python package designed for constructing, optimizing Boolean models and performs in-silico perturbations of the models.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/druglogics/pydruglogics",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
    ],
    python_requires=">=3.10",
    install_requires=[
        "pygad",
        "joblib",
        "matplotlib",
        "mpbn",
        "numpy",
        "pandas",
        "pyboolnet",
        "scipy",
        "scikit-learn"]
)