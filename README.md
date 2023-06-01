[![Build Status](https://github.com/PaccMann/paccmann_chemistry/actions/workflows/build.yml/badge.svg)](https://github.com/PaccMann/paccmann_chemistry/actions/workflows/build.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gradio demo](https://img.shields.io/website-up-down-green-red/https/hf.space/gradioiframe/GT4SD/paccmann_rl/+.svg?label=demo%20status)](https://huggingface.co/spaces/GT4SD/paccmann_rl)


# paccmann_chemistry

Generative models of chemical data for PaccMann<sup>RL</sup>. For example, a SMILES/SELFIES VAE using stack-augmented GRUs in both encoder and decoder. For details, see for example:

- [![DOI:10.1016/j.isci.2021.102269](http://img.shields.io/badge/DOI-10.1016/j.isci.2021.102269-094573.svg)](https://doi.org/10.1016/j.isci.2021.102269) [PaccMann<sup>RL</sup>: De novo generation of hit-like anticancer molecules from transcriptomic data via reinforcement learning](https://www.cell.com/iscience/fulltext/S2589-0042(21)00237-6) (_iScience_, 2021). 

- [![DOI:10.1088/2632-2153/abe808](http://img.shields.io/badge/DOI-10.1088/2632/2153/abe808-C1D4F4.svg)](https://doi.org/10.1088/2632-2153/abe808) [Data-driven molecular design for discovery and synthesis of novel ligands: a case study on SARS-CoV-2](https://iopscience.iop.org/article/10.1088/2632-2153/abe808) (_Machine Learning: Science and Technology_, 2021).





## Requirements

- `conda>=3.7`

## Installation

The library itself has few dependencies (see [setup.py](setup.py)) with loose requirements. 
To run the example training script we provide environment files under `examples/`.

Create a conda environment:

```sh
conda env create -f examples/conda.yml
```

Activate the environment:

```sh
conda activate paccmann_chemistry
```

Install in editable mode for development:

```sh
pip install -e .
```

## Example usage

In the `examples` directory is a training script [train_vae.py](./examples/train_vae.py) that makes use of `paccmann_chemistry`.

```console
(paccmann_chemistry) $ python examples/train_vae.py -h
usage: train_vae.py [-h]
                    train_smiles_filepath test_smiles_filepath
                    smiles_language_filepath model_path params_filepath
                    training_name

Chemistry VAE training script.

positional arguments:
  train_smiles_filepath
                        Path to the train data file (.smi).
  test_smiles_filepath  Path to the test data file (.smi).
  smiles_language_filepath
                        Path to SMILES language object.
  model_path            Directory where the model will be stored.
  params_filepath       Path to the parameter file.
  training_name         Name for the training.

optional arguments:
  -h, --help            show this help message and exit
```

`params_filepath` could point to [examples/example_params.json](examples/example_params.json), examples for other files can be downloaded from [here](https://ibm.box.com/v/paccmann-pytoda-data).

## References

If you use `paccmann_chemistry` in your projects, please cite the following:

```bib
@article{born2021datadriven,
  author = {Born, Jannis and Manica, Matteo and Cadow, Joris and Markert, Greta and Mill, Nil Adell and Filipavicius, Modestas and Janakarajan, Nikita and Cardinale, Antonio and Laino, Teodoro and {Rodr{\'{i}}guez Mart{\'{i}}nez}, Mar{\'{i}}a},
  doi = {10.1088/2632-2153/abe808},
  issn = {2632-2153},
  journal = {Machine Learning: Science and Technology},
  number = {2},
  pages = {025024},
  title = {{Data-driven molecular design for discovery and synthesis of novel ligands: a case study on SARS-CoV-2}},
  url = {https://iopscience.iop.org/article/10.1088/2632-2153/abe808},
  volume = {2},
  year = {2021}
}

@article{born2021paccmannrl,
  title = {PaccMann\textsuperscript{RL}: De novo generation of hit-like anticancer molecules from transcriptomic data via reinforcement learning},
  journal = {iScience},
  volume = {24},
  number = {4},
  pages = {102269},
  year = {2021},
  issn = {2589-0042},
  doi = {https://doi.org/10.1016/j.isci.2021.102269},
  url = {https://www.cell.com/iscience/fulltext/S2589-0042(21)00237-6},
  author = {Born, Jannis and Manica, Matteo and Oskooei, Ali and Cadow, Joris and Markert, Greta and {Rodr{\'{i}}guez Mart{\'{i}}nez}, Mar{\'{i}}a}
}
```
