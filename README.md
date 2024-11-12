# Multi-Marginal Flow Matching

Implementation of the Multi-Marginal Flow Matching model. 

## Installation

We recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) to create a new environement for installing the MMFM package.
Create the environment using the following command and activate it:
```sh
conda env create -f environment.yml

conda activate mmfm
```
After installing the dependencies, install the package:
```sh
pip install -e .
```
Keep the `-e` if you want to continue to edit the code.

## Usage

The following code shows how to train an MMFM model:

```python
from mmfm.conditional_flow_matching import MultiMarginalFlowMatcher

FM = MultiMarginalFlowMatcher()  # see code for default arguments

for batch in data_loader:
    x, targets, timepoints = batch
    optimizer.zero_grad()
    t, xt, ut, _, _ = FM.sample_location_and_conditional_flow(x, timepoints)
    vt = mmfm_model(torch.cat([xt, targets, t], dim=1))
    loss = torch.mean((vt - ut) ** 2)
    loss.backward()
    optimizer.step()

```

## Experiments

The code to reproduce the tables and figures of the synthetic and real-world experiments, please check out the code
in the benchmark folder. First, run the corresponding shell scripts to execute the gridsearch over a set of hyperparmeters.
Internally they call the `train_mmfm.py` scripts to to the actual model training. Second, run one of the analysis notebooks,
to reproduce the figures and tables.

