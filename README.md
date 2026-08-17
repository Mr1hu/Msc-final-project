# Data Distribution and Communication-Aware Optimisation in Distributed Deep Learning

This repository contains the implementation and experimental code for the MSc project **“Data Distribution and Communication-Aware Optimisation in Distributed Deep Learning.”**

The project investigates workload allocation for synchronous distributed deep learning on heterogeneous workers. It jointly models **computation throughput**, **data-transfer cost**, and **fixed per-worker latency**, and derives a closed-form allocation that minimises the predicted makespan by balancing worker completion times.

The implementation was evaluated using CIFAR-10 and ResNet-18 on a heterogeneous three-worker GPU cluster.

## Overview

Equal data partitioning implicitly assumes that all workers have similar processing and communication characteristics. In a heterogeneous cluster, this can create a **straggler effect**, where faster workers finish early and remain idle while waiting for the slowest worker.

This project compares three main allocation strategies:

* **Equal** — the dataset is divided approximately equally among workers.
* **Compute-only** — data is allocated in proportion to measured worker throughput.
* **Joint communication-aware allocation** — computation throughput, transfer cost, and fixed latency are jointly considered.

The proposed method follows a **measure-model-solve-refine** workflow:

1. Profile worker performance using a small workload.
2. Estimate communication parameters and compute throughput.
3. Solve the workload allocation analytically.
4. Apply the allocation to the full workload.
5. Refine the allocation using observed full-load throughput.
6. Validate both system performance and learning convergence.

## Repository Structure

The current public repository stores the implementation files at the repository root. Conceptually, the code is divided into **core optimisation components** and **experimental pipeline components**.

```text
.
├── closed_form_solver.py   # Closed-form workload allocation (Algorithm 1)
├── estimate_latency.py     # Hidden latency estimation (Algorithm 2)
├── refine_loop.py          # Iterative allocation refinement (Algorithm 3)
├── ga_verifier.py          # Genetic-algorithm verification (Algorithm 5)
│
├── profile_100.py          # Small-sample worker profiling
├── train_cifar.py          # Per-worker timing experiment
├── train_fedavg.py         # FedAvg local training on disjoint shards
├── aggregate.py            # Sample-weighted FedAvg aggregation and evaluation
├── stats3.py               # Statistics for repeated timing experiments
│
├── run_strategies.sh       # Allocation-strategy comparison
├── repeat3.sh              # Repeated timing experiments
└── run_converge.sh         # FedAvg convergence experiments
```

## Method

For worker (i), completion time is represented using an affine cost model combining fixed and data-dependent costs:

```text
T_i = L_i + d_i * (s / r_i + 1 / c_i)
```

where:

* `d_i` is the number of samples assigned to worker `i`,
* `L_i` is its fixed communication/launch latency,
* `s` is the payload size per sample,
* `r_i` is its effective data-transfer rate,
* `c_i` is its measured computation throughput.

The optimisation problem is:

```text
minimise   max_i T_i
subject to sum_i d_i = D
           d_i >= 0
```

For active workers, the continuous optimum is obtained by equalising their predicted completion times.

The resulting closed-form solver requires no numerical optimisation library and can be recomputed after new profiling or refinement measurements.

## Experimental Environment

The final experiments use the following three workers:

| Worker   | GPU              | CPU Allocation | Fixed Latency |
| -------- | ---------------- | -------------: | ------------: |
| Worker 1 | GeForce RTX 4070 |       10 cores |      324.6 ms |
| Worker 2 | GeForce RTX 3080 |       10 cores |      250.0 ms |
| Worker 3 | GeForce RTX 5080 |       16 cores |      418.3 ms |

The coordinator is responsible for allocation, remote execution, logging, and aggregation and is not included as a training worker in the optimisation.

### Software

* Python
* PyTorch 2.7.0
* CUDA 12.8
* cuDNN 9
* torchvision
* Apptainer
* CIFAR-10
* ResNet-18

The container image used in the experiments is:

```text
pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
```

The CUDA 12.8 / PyTorch 2.7 environment provides support for the RTX 5080 Blackwell `sm_120` architecture while remaining usable on the RTX 30- and 40-series workers.

## Reproducing the Experimental Workflow

### 1. Small-Sample Profiling

Run `profile_100.py` on each worker.

The measured median throughputs used for the initial allocation were:

| GPU      |       Throughput |
| -------- | ---------------: |
| RTX 4070 |  736.0 samples/s |
| RTX 3080 |  741.8 samples/s |
| RTX 5080 | 1069.0 samples/s |

### 2. Compute the Initial Allocation

Run the closed-form solver using the measured throughput and latency parameters:

```bash
python closed_form_solver.py \
    736.0 741.8 1069.0 \
    0.3246 0.2500 0.4183 \
    60000
```

This produces the initial profiling-based allocation.

### 3. Compare Allocation Strategies

Configure the required allocation ratios in the shell driver and run:

```bash
bash run_strategies.sh
```

The experiment compares:

```text
Equal
Compute-only
Joint
```

under the same model, dataset, preprocessing, CPU-affinity, and measurement configuration.

### 4. Refine the Allocation

Full-load execution provides more representative effective throughput measurements than the short profiling stage.

`refine_loop.py` uses these measurements to re-estimate worker throughput and recompute the communication-aware allocation.

The refinement step corrects systematic differences between small-sample profiling and full-load execution while retaining the same analytical allocation model.

### 5. Repeated Timing Experiments

Run each strategy repeatedly:

```bash
bash repeat3.sh
python stats3.py
```

`stats3.py` reports summary statistics for the repeated makespan measurements.

### 6. FedAvg Convergence Evaluation

The convergence experiment uses genuine non-overlapping CIFAR-10 shards and sample-weighted aggregation.

Example:

```bash
bash run_converge.sh JointP2 0.2733 0.2396 0.4871 10
bash run_converge.sh Equal   0.3333 0.3333 0.3334 10
```

`train_fedavg.py` performs local training and writes worker checkpoints. `aggregate.py` combines the models using weights proportional to the number of samples represented by each worker and evaluates the resulting model on the CIFAR-10 test set.

## Code-to-Method Mapping

| Component                      | File                    | Responsibility                                                          |
| ------------------------------ | ----------------------- | ----------------------------------------------------------------------- |
| Closed-form allocation         | `closed_form_solver.py` | Algorithm 1; computes the common completion time and optimal allocation |
| Hidden latency estimation      | `estimate_latency.py`   | Algorithm 2; transfer probing and least-squares latency estimation      |
| Iterative refinement           | `refine_loop.py`        | Algorithm 3; updates throughput from representative measurements        |
| Genetic-algorithm verification | `ga_verifier.py`        | Algorithm 5; independent numerical verification of the model optimum    |
| Worker profiling               | `profile_100.py`        | Repeated 100-sample ResNet-18 profiling                                 |
| Timing experiment              | `train_cifar.py`        | Per-worker training and model-based communication timing                |
| Distributed learning           | `train_fedavg.py`       | Local training on non-overlapping CIFAR-10 shards                       |
| Aggregation                    | `aggregate.py`          | Sample-weighted FedAvg and test-set evaluation                          |
| Strategy orchestration         | `run_strategies.sh`     | Runs Equal, Compute-only and Joint strategies                           |
| Repeated experiments           | `repeat3.sh`            | Repeats timing experiments                                              |
| Statistics                     | `stats3.py`             | Summarises repeated timing measurements                                 |
| Convergence evaluation         | `run_converge.sh`       | Coordinates multi-round FedAvg experiments                              |

## Scope and Reproducibility Notes

The implementation represents a controlled three-worker experimental system rather than a production distributed-training framework.

The current timing experiments use independently measured worker execution times and modelled communication costs. The shell-based SSH orchestration does not provide simultaneous distributed timing of all workers, and the timing results therefore reconstruct the makespan from worker measurements.

The FedAvg convergence experiment, in contrast, uses genuine non-overlapping data shards and sample-weighted model aggregation.

Before running the code on another cluster, update:

* worker host names,
* container paths,
* dataset paths,
* CPU-affinity settings,
* shared-storage paths,
* communication latency parameters.

The current scripts contain environment-specific configuration values and are therefore not intended to provide fully push-button reproduction on arbitrary hardware.

## Dataset

The experiments use **CIFAR-10**:

* 50,000 training images
* 10,000 test images

The profiling, timing, and convergence experiments use **ResNet-18**.

## Project Scope

The project focuses on:

* synchronous distributed training,
* static heterogeneous workers,
* computation and communication-aware data allocation,
* minimisation of training makespan,
* iterative allocation refinement,
* and validation of learning convergence.

Dynamic worker membership, non-IID allocation constraints, energy optimisation, and model/pipeline parallelism are outside the implemented scope.
