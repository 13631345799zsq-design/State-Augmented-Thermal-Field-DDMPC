# Supplementary Source Code

## Manuscript

**State-augmented transient thermal-field modeling for data-driven predictive thermal management of multi-heat-source electronic devices**

This archive contains the source code associated with the transient thermal-field models and the ANODE-based data-driven model predictive control (DD-MPC) framework described in the manuscript.

## Scope of this archive

During peer review, this supplementary archive provides source code only. It includes:

- training and evaluation code for ANODE, NODE, NODE-GRU, NODE-Delay, and ANODE-Direct;
- parameter-counting and loss-curve plotting utilities;
- the ANODE model used by the SLSQP-based MPC optimizer;
- one trained ANODE checkpoint used by the supplied MPC example; and
- scripts for generating predicted temperature histories, fan commands, optimization statistics, and diagnostic plots.

The complete transient dataset and processed experimental result files are not included in the review-stage archive. They will be released after acceptance. The physical experiments use a customized thermal-control and data-acquisition platform; hardware-interface and data-acquisition programs are therefore outside the scope of this archive.

## Directory structure

```text
Supplementary Material/
├── 1_Model_Train/
│   ├── 1_ANODE/
│   ├── 2_NODE/
│   ├── 3_NODE_GRU/
│   ├── 4_NODE_Delay/
│   └── 5_ANODE_Direct/
└── 2_ANODE_MPC_SLSQP/
    ├── _0_NODE_Model.py
    ├── _1_NODE_MPC_SLSQP.py
    └── models_1/
        └── model_epoch_980.pth
```

Each model directory contains the following files:

- `_0_*.py`: model definition;
- `_1_Transient_Model_train.py`: full-trajectory recursive training;
- `_2_Transient_Model_test.py`: recursive model evaluation and result export;
- `_3_parameter_count.py`: parameter-counting utility; and
- `_o_loss_plot_made.py`: training/validation loss plotting utility.

## Software requirements

The code requires a CUDA-capable GPU because the current scripts explicitly use CUDA tensors.

Main Python dependencies:

- Python 3;
- PyTorch with CUDA support;
- NumPy;
- SciPy;
- pandas;
- Matplotlib; and
- `pytorch_ssim`.

The SLSQP optimizer is provided by `scipy.optimize.minimize`. A recent PyTorch 2.x environment is recommended because the MPC script uses the `weights_only` option when loading the checkpoint.

## Dataset format

The training and testing scripts read numerically named `.pkl` files from the directory specified by `datapath`. The files are sorted by their numeric file name before dataset indices are applied.

Each `.pkl` file is expected to contain:

- `time_list`: physical time stamps for one transient trajectory;
- `flux_matrix`: heat-flux fields over time;
- `cooling_matrix`: cooling-input fields over time;
- `envtemp_list`: ambient-temperature sequence;
- `temp_matrix`: temperature fields over time.

The spatial fields are resampled to `128 × 128` by area interpolation. Temperature-related quantities use the normalization interval 20–85 °C, heat flux uses 0–0.03 W mm⁻², fan speed uses 0–6300 rpm, and time increments are normalized by 20 s in the model input.

## Dataset partition and repeated runs

The manuscript-level partition is:

- training trajectories: indices 0–119;
- validation trajectories: indices 120–159; and
- test trajectories: indices 160–199.

The distributed test scripts are currently configured with `test_indices = list(range(180, 200))` for a reduced 20-case evaluation. To reproduce the complete 40-case test reported in the manuscript, set:

```python
test_indices = list(range(160, 200))
```

The reported means and standard deviations are based on five independent runs with random seeds 42–46 and output directories `fold_result_1` through `fold_result_5`. For a fresh five-run training procedure, the training loop must cover:

```python
for n_fold in range(5):
```

The supplied ANODE-Direct training script is configured to resume the fifth run only (`range(4, 5)`). Change it to `range(5)` when reproducing all five independent runs from the beginning.

## Training-data-volume experiments

The variable `n` controls the learning-curve configuration:

| `n` | Training trajectories | Training epochs |
|---:|---:|---:|
| 1 | 120 | 1000 |
| 2 | 85 | 1414 |
| 3 | 60 | 2000 |
| 4 | 42 | 2828 |
| 5 | 30 | 4000 |

The five models use 120, 60, and 30 trajectories for the principal learning-curve comparison. ANODE additionally uses 85 and 42 trajectories.

## Running the model code

Edit `datapath` in the selected training or testing script so that it points to the transient-data directory. Run commands from within the corresponding model folder so that the local model import can be resolved.

Example:

```bash
python _1_Transient_Model_train.py
python _2_Transient_Model_test.py
python _3_parameter_count.py
python _o_loss_plot_made.py
```

Training creates `fold_result_*` directories containing checkpoints, loss arrays, and plots. Testing loads the checkpoints from these directories and exports per-fold and aggregated evaluation results.

## Running the MPC example

The MPC example is located in `2_ANODE_MPC_SLSQP`. Its main settings are placed at the beginning of `_1_NODE_MPC_SLSQP.py`, including:

- heat-source powers;
- ambient, initial, and target temperatures;
- prediction-horizon time steps;
- SLSQP iteration limit and tolerance; and
- heat, temperature-gradient, and cooling weights.

Run:

```bash
python _1_NODE_MPC_SLSQP.py
```

The script loads `models_1/model_epoch_980.pth`, performs repeated SLSQP optimization, and writes NumPy arrays and plots to a parameter-specific `SLSQP_results_*` directory. The supplied example uses a 40 °C ambient condition, a 40 °C initial field, and a 65 °C temperature target. Other target temperatures used in the manuscript can be evaluated by changing `target_temp`.

The MPC program represents the offline predictive-planning component. It does not communicate directly with the experimental fan controller or temperature-acquisition hardware.

## Notes on reproducibility

- Model training and testing require the transient dataset, which is not included at the review stage.
- The physical experiments require the customized experimental platform described in the manuscript.
- CUDA and library versions can cause small numerical differences, particularly in optimization time and floating-point results.
- Optimization time depends on the GPU and concurrent system workload; the SLSQP iteration count is the more hardware-independent computational indicator.
- The class name `PhysicsInformedCNN` is a legacy implementation name. The present framework uses physically motivated input and state construction rather than a conventional physics-informed neural network trained with governing-equation residuals.

## Availability

The source code in this archive is supplied for manuscript review. The complete dataset and processed experimental results will be released after acceptance to support further verification and reuse.
