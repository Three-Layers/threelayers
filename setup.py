from setuptools import find_packages, setup

with open("readme.md", "r") as f:
    long_description = f.read()

setup(
    name="threelayers",
    version="0.0.1",
    description="Data science library built through competition experience",
    package_dir={"": "threelayers"},
    packages=find_packages(where="threelayers"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    author="threelayers",
    author_email="zazaneryawan@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    install_requires=["pandas>=2.0.0", "scipy>=1.10.1", "scikit-learn>=1.1.0"],
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    python_requires=">=3.9",
)