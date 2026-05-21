# OrthoBO

A research implementation and benchmark suite for the orthogonalized Bayesian optimization algorithm introduced in [ORTHOBO: Orthogonal Bayesian Hyperparameter Optimization](https://arxiv.org/abs/2605.06454).

<!--This repository contains the full benchmark infrastructure used to develop and validate an [OptunaHub](https://hub.optuna.org/samplers/orthobo/) submission of the OrthoBO sampler. If you just want to use OrthoBO in your own Optuna study, install it directly from OptunaHub (you don't need this repo).-->

## Table of Contents
- [Description](#description)
- [Repository Structure](#repository-structure)
- [OptunaHub Usage](#optunahub-usage)
- [Benchmark Results](#benchmark-results)
- [Reference](#reference)
<!--[Using OrthoBO via OptunaHub](using-orthobo-via-optunahub)-->

## Description
Standard Bayesian optimization (BO) fits a probabilistic surrogate model (typically a Gaussian process (GP)) to observed data and maximizes an acquisition function (typically Expected Improvement (EI)) to select the next evaluation point. In practice, computing this acquisition function across the model's posterior distribution is complex, so it is approximated using finite Monte Carlo (MC) sampling. However, this finite sampling introduces MC estimation noise, which can flip candidate rankings and lead to suboptimal point selection. OrthoBO reduces this noise by subtracting an orthogonal correction derived from posterior distribution gradients (score functions) to cancel out the variance components introduced during the MC evaluation. This results in more stable acquisition rankings. Whether this translates to better optimization performance depends on the problem (see [benchmark results](#benchmark-results) below).

## Repository Structure
```bash
orthobo/
├── configs/
│   ├── config.yaml               # Experiment defaults
│   ├── benchmark/                # hartmann6.yaml, ackley10.yaml, levy16.yaml
│   └── sampler/                  # orthobo.yaml, naive.yaml, vanilla.yaml
├── src/
│   ├── samplers/
│   │   ├── base.py               # BaseSampler (wraps BoTorchSampler)
│   │   ├── marginal.py           # MarginalBoTorchSampler (OrthoBO + Naive)
│   │   └── vanilla.py            # VanillaBoTorchSampler
│   └── utils/
│       ├── hyperposterior.py     # Laplace approximation + theta sampling
│       ├── ortho_acquisition.py  # OrthogonalLogEI acquisition function
│       └── theta_extraction.py   # GP hyperparameter extraction utilities
├── scripts/
│   ├── run_benchmark.py          # Hydra entry point for single experiments
│   ├── run_experiments.sh        # Experiment sweep launcher (submits 45 jobs with hydra multirun)
│   └── aggregate_and_plot.py     # Results aggregation and plotting
├── figures/                      # Benchmark plots
├── README.md
└── pyproject.toml
```
<!--
├── REPRODUCING.md                # How to run the benchmarks
├── IMPLEMENTATION_NOTES.md       # Math and design decisions
-->

<!--
## Using OrthoBO via OptunaHub

```python
import optuna
import optunahub

OrthoBoSampler = optunahub.load_module("samplers/orthobo").OrthoBoSampler

def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5.0, 5.0)
    y = trial.suggest_float("y", -5.0, 5.0)
    return (x - 2) ** 2 + (y + 1) ** 2

sampler = OrthoBoSampler(n_startup_trials=10, mc_budget=64)
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=50)
```

The orthogonal correction can be disabled for ablation purposes:

```python
# Naive Marginal BO without variance reduction
sampler = OrthoBoSampler(use_orthogonal_correction=False)
```
-->

## Benchmark Results
Mean best-so-far regret ± 1 std across 5 random seeds, 200 trials, 10 startup trials, MC budget S=64. Three methods compared:
- Vanilla BO: Single GP qLogEI with MAP hyperparameters and Sobol MC sampling.
- Naive Marginal BO: MC marginalisation over the hyperposterior (S=64) without any correction.
- OrthoBO: MC marginalisation over the hyperposterior (S=64) with orthogonal score-function control variate.

The benchmark functions used here are standard test problems from the global optimization literature [1]. Their properties are well-documented, but how those properties interact with any specific BO implementation depends on many factors (e.g. kernel choice, initialization, and MC budget). We provide some plausible explanations below, though we are not experts so these conclusions are subject to scrutiny. If you have any comments please leave feel free to write a comment in the Issues tab.

### Hartmann-6 (6 dimensions)
![Hartmann6](figures/Hartmann6.png)

OrthoBO achieves the lowest regret, with Naive Marginal BO in the middle and Vanillac BO plateauing earliest. Hartmann-6 is a relatively smooth function with a small number of local minima [[1]](#references), which may explain why a GP surrogate can learn its global structure reliably even with few observations. It's possible that when the surrogate is well-calibrated, Vanilla BO's single maximum a posteriori (MAP) hyperparameter estimate may lead to overconfidence and premature convergence. While marginalising over the hyperposterior helps, Naive Marginal's MC noise
appears to prevent it from fully exploiting that better model. OrthoBO's variance reduction may be removing that remaining noise, allowing cleaner exploitation of an accurate landscape. This is the setting the paper [[2]](#references) identifie as most favourable for OrthoBO.
 
### Levy-16 (16 dimensions)
![Levy16](figures/Levy16.png)

All three methods converge to similar final regret after 200 trials, but OrthoBO and Naive Marginal reach that level earlier than Vanilla BO. Levy-16 is a multimodal function [1] scaled to 16 dimensions, which means the GP faces high
uncertainty throughout the search. The earlier convergence of both marginal methods over Vanilla BO is consistent with
the idea that marginalising over hyperparameter uncertainty forces more exploration when the surrogate fit is uncertain. The methods are close overall, possibly because data scarcity dominates at this scale regardless of acquisition strategy. That OrthoBO edges out Naive Marginal is consistent with the paper's  argument that
variance reduction becomes more important as the MC budget becomes more constrained relative to the dimensionality of the problem [[2]](#references).
 
### Ackley-10 (10 dimensions)
![Ackley10](figures/Ackley10.png)

Vanilla BO outperforms both marginal methods, with OrthoBO and Naive Marginal plateauing around iteration 75-100. Ackley is a highly multimodal function with a nearly flat outer basin of local minima surrounding a steep central funnel [[1, 3]](#references). With only 10 startup points in 10 dimensions, it seems plausible
that the GP initialises in the outer basin and fits its lengthscales to the local structure rather than the global funnel. It's possible that averaging EI across 64 models sampled from a miscalibrated hyperposterior may oversmooth the acquisition landscape, causing both OrthoBO and Naive Marginal to plateau in the outer basin. Vanilla BO's single MAP estimate may accidentally provide a more consistent target if it is overconfident and allows the optimizer to keep improving. This result is consistent with the paper's observation that OrthoBO's advantage is most pronounced when acquisition estimation noise is the primary failure mode [[2]](#references). On highly multimodal functions, surrogate model miscalibration may dominate instead.

We note that this implementation uses the same RBF kernel with ARD lengthscales as the original paper [[2]](#references), so kernel choice is unlikely to explain the discrepancy. The most likely explanation remains surrogate miscalibration in the outer basin under limited startup data, though we cannot rule out other implementation differences. 

## References
```bibtex
[1]
@article{jamil2013literature,
  author    = {Jamil, Misha and Yang, Xin-She},
  title     = {A Literature Survey of Benchmark Functions for Global Optimisation Problems},
  journal   = {International Journal of Mathematical Modelling and Numerical Optimisation},
  volume    = {4},
  number    = {2},
  year      = {2013},
  url       = {https://arxiv.org/abs/1308.4008}
}

[2]
@article{schroder2026orthobo,
  author    = {Schr{\"o}der, Marcel and Janetzky, Pascal and Klar, Markus and Feuerriegel, Stefan},
  title     = {ORTHOBO: Orthogonal Bayesian Hyperparameter Optimization},
  journal   = {arXiv preprint arXiv:2605.06454},
  year      = {2026},
  url       = {https://arxiv.org/abs/2605.06454}
}

[3]
@misc{surjanovicbingham,
  author    = {Surjanovic, Sonja and Bingham, Derek},
  title     = {Virtual Library of Simulation Experiments: Test Functions and Datasets},
  howpublished = {\url{https://www.sfu.ca/~ssurjano/ackley.html}},
  note      = {Accessed: May 2026}
}

```