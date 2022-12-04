# Robust Classification with Non-Perturbable Predictors

MIT 15.095 ML Opt Final Presentation: Equivalence between Robustness and Regularization with Non-Perturbable Predictors

## Requirement

Gurobi version >= 9.5.2, PyTorch >= 1.12.0

## Dataset

### Synthetic Dataset

At root, run the following to create the synthetic and real dataset

```bash
python src/data/synthetic.py  # create synthetic dataset
python src/data/real.py       # parse real dataset from uci
```

Note that each takes an argument ```seed``` for reproducibity: synthetic dataset are randomly generated and some real dataset contains probabilistic imputation in parsing.
