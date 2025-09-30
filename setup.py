"""
Setup script for AIRL Franka Cube Lift Project
"""

from setuptools import setup, find_packages

setup(
    name="airl_franka",
    version="0.1.0",
    description="AIRL implementation for Franka Cube Lift Task in Isaac Lab",
    author="Michael Schnidrig",
    author_email="mschnidrig@ethz.ch",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0", 
        "h5py>=3.7.0",
        "gymnasium>=0.26.0",
        "PyYAML>=6.0",
        "tensorboard>=2.10.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "pre-commit>=2.17.0",
        ],
        "isaac": [
            # Isaac Lab dependencies will be managed by Isaac Lab installation
            # These are just placeholders for documentation
        ]
    },
    entry_points={
        "console_scripts": [
            "train-airl=train_airl:main",
            "validate-expert-data=utils.validate_expert_data:main",
            "evaluate-policy=utils.evaluate_policy:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="reinforcement-learning, imitation-learning, robotics, isaac-sim",
    project_urls={
        "Documentation": "https://github.com/your-username/airl_franka",
        "Source": "https://github.com/your-username/airl_franka",
        "Bug Reports": "https://github.com/your-username/airl_franka/issues",
    },
)
