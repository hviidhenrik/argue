from setuptools import find_packages, setup

setup(
    name="phd-argue",
    version="1.0.7",
    description="Python package with ARGUE anomaly detection model",
    author="Henrik Hviid Hansen",
    packages=find_packages(),
    classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent"],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "matplotlib",
        "numpy",
        "pandas",
        "plotly",
        "scikit_learn",
        "tensorflow-cpu",
        "tqdm",
        "statsmodels",
    ],
)
