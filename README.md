# Code Package Overview

Experimental code for **Data Distribution and Communication-Aware Optimisation in Distributed Deep Learning**.

## Directory Structure

```
core/       Core algorithms (Python standard library only; can run directly on any machine)
  closed_form_solver.py    Closed-form allocation algorithm (main contribution)
  ga_verifier.py           Independent verification using a genetic algorithm
  estimate_latency.py      Estimation of hidden latency using least-squares fitting
  refine_loop.py           Iterative refinement loop

pipeline/   Code executed on the cluster (requires PyTorch + CUDA)
  profile_100.py           Phase 1 small-sample profiling (100 images, repeated 3 times; median reported)
  train_cifar.py           Single-node timed training; reports communication/computation time
  train_fedavg.py          FedAvg local training step (non-overlapping partitions + weight saving)
  aggregate.py             FedAvg weighted aggregation + test-set evaluation
  stats3.py                Statistics for repeated experiments
  run_strategies.sh        Driver for comparing multiple allocation strategies
  repeat3.sh               Repeats each strategy 3 times
  run_converge.sh          Driver for FedAvg convergence experiments
```

## Mapping to the Dissertation

| Dissertation Content | Code |
|---|---|
| Communication cost model | `node_time()` in `closed_form_solver.py` |
| Closed-form derivation (Algorithm 1) | `closed_form_allocate()` in `closed_form_solver.py` |
| Hidden latency estimation (Algorithm 2) | `estimate_latency.py` |
| Iterative refinement (Algorithm 3) | `refine_loop.py` |
| Two-phase workflow (Algorithm 4) | `profile_100.py` → `closed_form_solver.py` → `run_strategies.sh` |
| GA verification (Algorithm 5) | `ga_verifier.py` |
| Timing experiment results | `run_strategies.sh` + `repeat3.sh` + `stats3.py` |
| FedAvg convergence results | `run_converge.sh` + `train_fedavg.py` + `aggregate.py` |

## Complete Experimental Workflow

**Step 1 — Small-Sample Profiling**

Run `profile_100.py` on each worker to obtain the median throughput. The measured results in this study were:
736.0 / 741.8 / 1069.0 samples/s (RTX 4070 / RTX 3080 / RTX 5080)

**Step 2 — Compute the Initial Allocation P0**

```
python core/closed_form_solver.py 736.0 741.8 1069.0 0.3246 0.2500 0.4183 60000
```
This produces P0 = 29.03 / 29.34 / 41.63 %.

**Step 3 — Apply the Allocation to the Full Dataset and Compare Against Baselines**

```
export CO1=0.2890 CO2=0.2913 CO3=0.4197
export P01=0.2903 P02=0.2934 P03=0.4163
bash pipeline/run_strategies.sh
```

**Step 4 — Refinement**

From the logs generated in the previous step, calculate the observed throughput as `number of images / computation time` (941.5 / 1078.0 / 1716.7 samples/s).
Re-solving the allocation problem gives P1 = 25.49 / 29.21 / 45.29 %.

**Step 5 — Repeat Experiments and Calculate Error Ranges**

```
bash pipeline/repeat3.sh && python pipeline/stats3.py
```

**Step 6 — Compare FedAvg Convergence**

```
bash pipeline/run_converge.sh JointP2 0.2733 0.2396 0.4871 10
bash pipeline/run_converge.sh Equal   0.3333 0.3333 0.3334 10
```

## Environment

- **Container:** Apptainer, using an image based on `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime`
  (The RTX 5080 uses the Blackwell `sm_120` architecture and therefore requires this version or newer.)
- **CPU core restriction:** `taskset -c`
- **Shared storage:** `/uolstore` network storage, accessible by all three workers and used for scripts and checkpoints
- **Dataset:** CIFAR-10, with 50,000 training images and 10,000 test images

## Cluster Configuration

The host addresses, container paths, and fixed latency values at the top of the driver scripts should be modified according to the target environment.
The configuration used in this study was:

| Worker | GPU | CPU Cores | Fixed Latency |
|---|---|---|---|
| worker 1 | RTX 4070 | 10 | 324.6 ms |
| worker 2 | RTX 3080 | 10 | 250.0 ms |
| worker 3 | RTX 5080 | 16 | 418.3 ms |

For the FedAvg experiments, the fixed latency also includes the per-round model-weight writing overhead
(0.66 / 4.22 / 0.72 s), resulting in effective fixed latency values of 0.985 / 4.470 / 1.138 s.
