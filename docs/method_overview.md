# PRA method overview

PRA operates after a frozen token encoder. The same frozen encoder and the same trainable projection are applied to an original onset-apex pair and to its periocular-neutralized counterpart. For every valid token, PRA computes the Euclidean response between the projected representations. A learnable scalar scales this response before it is added to the original-token content score. Masked softmax produces the aggregation weights, and only the projected original tokens are pooled for classification.

The included `ContentOnlyAggregation` class implements the matched baseline. It uses the same projection, content score, masked softmax, and original-token pooling path, but does not use the neutralized branch or the response term.

The public code intentionally stops at the core method boundary. Dataset acquisition, D-FACE feature extraction, LOSO orchestration, and experiment reporting remain outside this repository because they depend on licensed data, third-party resources, and the authors' full experimental infrastructure.

