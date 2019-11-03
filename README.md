[![Build Status](https://travis-ci.org/PaccMann/paccmann_chemistry.svg?branch=master)](https://travis-ci.org/PaccMann/paccmann_chemistry)
# paccmann_chemistry

Generative models of chemical data for PaccMann^RL.

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
@misc{born2019reinforcement,
    title={Reinforcement learning-driven de-novo design of anticancer compounds conditioned on biomolecular profiles},
    author={Jannis Born and Matteo Manica and Ali Oskooei and Maria Rodriguez Martinez},
    year={2019},
    eprint={1909.05114},
    archivePrefix={arXiv},
    primaryClass={q-bio.BM}
}
```
