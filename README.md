# gpt

A from-scratch GPT, built up one layer at a time. Starts from a single neuron and ends — eventually — at a working transformer.

This is a learning project, so the code follows the journey rather than jumping straight to the final architecture. Each module is small, readable, and meant to be understood end-to-end.

## Status

| Area | What's there |
| --- | --- |
| `foundations/` | Neural-net primitives in NumPy: linear regression, gradient descent, activations, softmax, losses, a single neuron's forward pass, single- and multi-layer backprop. |
| `model/` | Reserved for attention, transformer blocks, and the GPT itself. Not started. |
| `data/` | Reserved for the tokenizer and data loader. Not started. |

Recent work: multi-layer backpropagation (step 3 of 5 in "Build a Neural Net").

## Setup

This is a [uv](https://docs.astral.sh/uv/) project. Install uv, then:

```bash
uv sync
```

That creates `.venv/` and installs everything pinned in `uv.lock`. Run anything with `uv run`:

```bash
uv run python -c "from foundations.neuron import Solution; print(Solution())"
```

Or activate the venv directly:

```bash
source .venv/bin/activate
python -c "from foundations.neuron import Solution; print(Solution())"
```

## Layout

```
foundations/                    NumPy-only building blocks
  linear_regression.py          Forward pass + MSE for a linear model
  linear_regression_training.py Gradient-descent training loop for the above
  gradient_descent.py           Plain gradient descent
  activations.py                ReLU, sigmoid, tanh
  softmax.py                    Numerically stable softmax
  loss.py                       MSE, cross-entropy
  neuron.py                     Single neuron forward pass
  backprop.py                   Backprop through a single layer
  multi_layer_backprop.py       Backprop through a stack of layers

model/                          (planned) attention, transformer, GPT
data/                           (planned) tokenizer, dataset, loader
```

## Dependencies

- Python ≥ 3.10
- `numpy` — foundations
- `torch`, `torchtyping` — for the model layer once it lands

Pinned versions live in `pyproject.toml` and `uv.lock`.
