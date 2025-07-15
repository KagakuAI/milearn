from setuptools import setup, find_packages

# Optional: read dependencies
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="milearn",
    version="1.0",
    author="KagakuAI",
    author_email="dvzankov@gmail.com.com",
    description="Multi-instance machine learning in Python",
    long_description_content_type="text/x-rst",
    url="https://github.com/KagakuAI/milearn",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # update if different
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
