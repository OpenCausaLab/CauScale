## SERGIO Gene Expression Simulator

Gene expression data simulator based on SERGIO. The simulation pipeline is adapted from
[Targeted Cause Discovery](https://github.com/snu-mllab/Targeted-Cause-Discovery), which in turn
uses the SERGIO wrapper from the [AVICI repository](https://github.com/larslorch/avici/tree/full).

To run the codes, first navigate to this directory and install the dedicated conda environment
(requires JAX + TensorFlow, separate from the main training env):
```bash
cd data/data_generation/sergio
conda env create -f environment.yml
conda activate avici
```

To generate all datasets at once, run the provided bash scripts from this directory:
```bash
# Training data: nodes 10–150, 5 graph types × 3 degrees × 200 graphs
bash bash/data-train-sergio.sh

# Test data: nodes 100–200, er × 2 degrees × 5 graphs
bash bash/data-test-sergio.sh
```

To obtain a single dataset, run
```
python -B -m avici.experiment.data --data_config_path config/train/n{d}/[config file] --path_data [output path] --j [env index]
```
For example:
```
python -B -m avici.experiment.data --data_config_path config/train/n10/data_er_e2.yaml --path_data datasets/sergio/train/n10/er_e2_add --descr data_er_e2 --j 0
```
- Data configurations are in `config/train/n{d}/` and `config/test/n{d}/`.
- We train on `{er,sbm,sf,sf_in,sf_indirect}_e{2,4,6}` for nodes 10, 20, 30, 50, 80, 100, 150.
- We generate 200 environments (`--j 0~199`) for each training config.
- Test configs are `er_e{2,4}` for nodes 100 and 200, with 5 environments (`--j 1001~1005`).

The resulting files contain
1. ground-truth causal graph `DAG.npy`.
2. observational data:  
(1) discretized UMI count expression data `data.npy` (low fidelity).  
(2) clean data without any technical noise `clean.npy`.  
(3) continuous expression data with dropout `dropout.npy` (high fidelity).
3. interventional data `data_intv.npy`, `clean_intv.npy`, `dropout_intv.npy` and corresponding intervention mask `intv.npy`.
4. metadata `info.json`.
