"""Install package."""
import codecs
import os

from setuptools import find_packages, setup


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError('Unable to find version string.')


setup(
    name='paccmann_chemistry',
    version=get_version('paccmann_chemistry/__init__.py'),
    description='Generative models of chemical data for PaccMann^RL',
    long_description=open('README.md').read(),
    url='https://github.com/PaccMann/paccmann_chemistry',
    author=(
        'Ali Oskooei, Matteo Manica, Jannis Born, Joris Cadow, Nil Adell Mill'
    ),
    author_email=(
        'ali.oskooei@gmail.com, drugilsberg@gmail.com, '
        'jab@zurich.ibm.com, joriscadow@gmail.com, nila@ethz.ch'
    ),
    install_requires=[
        'numpy', 'torch>=1.0.0',
        'pytoda @ git+https://git@github.com/PaccMann/paccmann_datasets@0.2.4'
    ],
    packages=find_packages('.'),
    zip_safe=False,
)
