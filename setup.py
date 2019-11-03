"""Install package."""
from setuptools import setup, find_packages

setup(
    name='paccmann_chemistry',
    version='0.0.1',
    description='Generative models of chemical data for PaccMann^RL',
    long_description=open('README.md').read(),
    url='https://github.com/PaccMann/paccmann_chemistry',
    author='Ali Oskooei, Matteo Manica, Jannis Born, Joris Cadow',
    author_email=(
        'ali.oskooei@gmail.com, drugilsberg@gmail.com, '
        'jab@zurich.ibm.com, joriscadow@gmail.com'
    ),
    install_requires=[
        'numpy',
        'torch>=1.0.0',
        'tensorflow>=1.10,<2.0',
    ],
    packages=find_packages('.'),
    zip_safe=False,
)
