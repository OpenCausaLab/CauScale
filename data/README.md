# Data

## Directory Structure

```
data/
├── data_csv/                        # Dataset index files (CSV)
│   ├── train-synthetic-stage-1.csv  # Training index: small graphs (10–100 nodes)
│   ├── train-synthetic-stage-2.csv  # Training index: large graphs (150–500 nodes)
│   ├── inference_synthetic.csv      # Test index: synthetic graphs (10–1000 nodes)
│   ├── train-sergio.csv             # Training index: SERGIO gene expression
│   └── inference_sergio.csv         # Test index: SERGIO gene expression
├── data_generation/
│   ├── synthetic/                   # Synthetic data generation scripts
│   │   ├── generate_data_continuous.py  # Main generation script
│   │   ├── dag_generator.py             # DAG sampling and FCM simulation
│   │   ├── causal_mechanisms.py         # Causal mechanism implementations
│   │   └── bash/                        # Shell scripts for bulk generation
│   │       ├── train-synthetic-stage1.sh
│   │       ├── train-synthetic-stage2.sh
│   │       └── test-synthetic.sh
│   └── sergio/                      # SERGIO gene expression generation scripts
│       ├── avici/                   # AVICI source (SERGIO wrapper)
│       ├── bash/
│       │   ├── data-train-sergio.sh
│       │   └── data-test-sergio.sh
│       ├── config/                  # Per-size YAML configs
│       │   ├── train/n{d}/          # d ∈ {10,20,30,50,80,100,150,200}
│       │   └── test/n{d}/           # d ∈ {100,200}
│       ├── environment.yml
│       └── README.md
└── datasets/                        # Generated data (not tracked by git)
    ├── synthetic/
    │   ├── train/
    │   └── test/
    └── sergio/
        ├── train/
        └── test/
```

## CSV Index Format

Each CSV has four columns:

| Column | Description |
|--------|-------------|
| `fp_data` | Path to interventional data `.npy`, shape `(N, d)` |
| `fp_graph` | Path to ground-truth adjacency matrix `.npy`, shape `(d, d)` |
| `fp_regime` | Path to intervention targets (`.csv` for synthetic, `.npy` for SERGIO) |
| `split` | `train` / `val` / `test` |

Paths are relative to the project root.

---

## Synthetic Data

### Folder Structure

Each generated folder follows the pattern:

```
data_p{d}_e{E}_n{N}_{mechanism}_{dag_type}_structural/
```

- `d`: number of nodes
- `E`: expected number of edges (`d × degree`)
- `N`: total data points per dataset
- `mechanism`: causal mechanism (e.g. `linear`, `nn`, `nn_add`)
- `dag_type`: graph structure (`er` or `sf`)

Each folder contains:
- `DAG{i}.npy` — adjacency matrix, `(d, d)` boolean
- `data_interv{i}.npy` — concatenated data across all environments, `(N, d)`
- `intervention{i}.csv` — per-row intervention targets (empty row = observational)
- `regime{i}.csv` — environment index per row (0 = observational, 1..K = interventional)

### Generating Synthetic Data

The scripts use absolute paths derived from their own location, so they can be run from any directory:

```bash
# Stage 1: small graphs (10, 20, 100 nodes), 600 DAGs each, 1000 points
bash data/data_generation/synthetic/bash/data-train-synthetic-stage1.sh

# Stage 2: large graphs (150, 200, 300, 500 nodes), 300 DAGs each, 10000 points
bash data/data_generation/synthetic/bash/data-train-synthetic-stage2.sh

# Test: graphs from 10 to 1000 nodes, 5 DAGs each, 1000 points
bash data/data_generation/synthetic/bash/data-test-synthetic.sh
```

Key parameters shared across all scripts:

| Parameter | Value |
|-----------|-------|
| Mechanisms | `linear`, `nn`, `nn_add` (train); `linear`, `nn`, `sigmoid_add`, `polynomial` (test) |
| DAG types | `er` (Erdős-Rényi), `sf` (Scale-Free) |
| Expected degrees | 1.0, 2.0, 3.0, 4.0 |
| Interventions | one per node (perfect single-target) |
| Intervention type | structural (do-intervention) |

To generate a single configuration manually:

```bash
python generate_data_continuous.py \
    --mechanism linear \
    --nb-nodes 20 \
    --expected-degree 2.0 \
    --nb-points 1000 \
    --nb-dag 10 \
    --intervention \
    --nb-interventions 20 \
    --max-nb-target 1 \
    --min-nb-target 1 \
    --cover \
    --obs-data \
    --dag-type er \
    --output_folder my_data \
    --output-dir .
```

---

## SERGIO Gene Expression Data

### Folder Structure

Each generated graph instance is stored under:

```
datasets/sergio/{train,test}/n{d}/{graph_type}_e{degree}_add/{j}/
```

- `d`: number of nodes
- `graph_type`: `er`, `sf`, `sbm`, `sf_indirect`, or `sf_in`
- `degree`: expected edges per node (2, 4, or 6)
- `j`: graph index

Each `{j}/` folder contains:
- `DAG.npy` — adjacency matrix, `(d, d)` boolean
- `data_intv.npy` — interventional data, `(N_int, d)`
- `intv.npy` — intervention mask per row, `(N_int, d)` boolean
- `data.npy` — observational data (UMI counts), `(150, d)`
- `clean.npy`, `dropout.npy` — observational data variants
- `clean_intv.npy`, `dropout_intv.npy` — interventional data variants
- `info.json` — metadata

### Generating SERGIO Data

See `data/data_generation/sergio/README.md` for environment setup and full instructions.

The scripts use absolute paths derived from their own location, so they can be run from any directory:

```bash
# Training: nodes 10–200, 5 graph types × 3 degrees × 200 graphs each
bash data/data_generation/sergio/bash/data-train-sergio.sh

# Test: nodes 100–200, er × 2 degrees × 5 graphs (j=1001–1005)
bash data/data_generation/sergio/bash/data-test-sergio.sh
```

**Training parameters:**

| Parameter | Values |
|-----------|--------|
| Nodes | 10, 20, 30, 50, 80, 100, 150, 200 |
| Graph types | `er`, `sf`, `sbm`, `sf_indirect`, `sf_in` |
| Expected degrees | 2, 4, 6 |
| Graphs per config | 200 (j = 0..199) |
| Interventional points | 5000 |

**Test parameters:**

| Parameter | Values |
|-----------|--------|
| Nodes | 100, 200 |
| Graph types | `er` |
| Expected degrees | 2, 4 |
| Graphs per config | 5 (j = 1001..1005) |
| Interventional points | 20000 |

