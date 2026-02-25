"""Setup configuration for hawkes-order-flow package."""

from setuptools import setup, find_packages

setup(
    name="hawkes-order-flow",
    version="0.1.0",
    description="Multivariate Hawkes processes for order flow alpha",
    author="Quantitative Researcher",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "numba>=0.57.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.3.0",
        "requests>=2.31.0",
        "joblib>=1.3.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.9",
    zip_safe=False,
)
