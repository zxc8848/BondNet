from setuptools import setup, find_packages

setup(
    name='bondnet',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'torch-geometric>=2.3.0',
        'rdkit',
        'ase',
        'numpy',
        'scipy',
        'scikit-learn',
        'tqdm',
        'pyyaml',
        'matplotlib',
    ],
    python_requires='>=3.9',
)
