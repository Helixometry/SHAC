from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="SHAC",
    version="1.0.0",
    author="Girish, Orchid Chetia Phukan, Mohd Mujtaba Akhtar, Bhavinkumar Vinodbhai Kuwar, Swarup Ranjan Behera, Arun Balaji Buduru",
    description="Detecting Neural Audio Codec Synthesized Heart Sounds",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Helixometry/SHAC",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "transformers>=4.30.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "python-speech-features>=0.6",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "PyYAML>=6.0",
        "tqdm>=4.60.0",
        "matplotlib>=3.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
)
