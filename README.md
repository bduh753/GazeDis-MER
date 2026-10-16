# GazeDis-MER: Periocular Response Aggregation

This repository provides the core implementation of **Periocular Response Aggregation (PRA)** for cross-subject micro-expression recognition.

PRA compares frozen D-FACE action tokens from an original onset-apex pair with tokens from a paired input whose apex has neutralized periocular motion. The token-wise response adjusts the aggregation weights of the original tokens. The neutralized tokens guide weighting only and are not passed directly to the classifier.

## Method

For token index `i`, the shared projection produces the original representation `h_i` and neutralized representation `h_i_neu`:

```text
d_i = ||h_i - h_i_neu||_2
c_i = w^T h_i + b
s_i = c_i + alpha * d_i
a_i = MaskedSoftmax(s_i)
z   = sum_i a_i * h_i
```

Relative to Content-only Aggregation, PRA adds one learnable scalar, `alpha`, to scale the token response.

## Public release scope

This is a **core-method release**, not a complete reproduction package. It includes:

- periocular neutralization from aligned onset and apex images;
- token-wise response computation;
- Content-only and PRA token aggregation;
- lightweight classifier wrappers and unit tests.

It does not include:

- complete training or evaluation entry points;
- automated leave-one-subject-out experiment orchestration;
- CASME II, SAMM, or other dataset images;
- derived or neutralized face images;
- cached D-FACE tokens or third-party model weights;
- server configurations, checkpoints, logs, or experiment outputs.

The datasets and D-FACE resources must be obtained from their original providers under the applicable access conditions and licenses. The paper reports the label mapping, random seeds, hyperparameters, and LOSO protocol used in the experiments.

## Installation

Python 3.10 or later is recommended.

```bash
python -m pip install -r requirements.txt
```

## Core usage

```python
import torch

from pra import PRAClassifier

model = PRAClassifier(token_dim=256, hidden_dim=128, num_classes=3)
original_tokens = torch.randn(2, 32, 256)
neutralized_tokens = torch.randn(2, 32, 256)
valid_tokens = torch.ones(2, 32, dtype=torch.bool)

logits, details = model(original_tokens, neutralized_tokens, valid_tokens)
print(logits.shape)
print(details["attention"].shape)
```

The image neutralization function accepts RGB onset/apex arrays, their two-dimensional facial landmarks, and a `LandmarkRegionSpec`. This keeps landmark detection separate from the core method and allows callers to use the detector permitted by their environment.

## Tests

```bash
python -m pytest tests -q
python -m py_compile pra/*.py
```

## License

Original code in this repository is released under the MIT License. Dataset files, model weights, cached features, and third-party components are not covered by this license.

