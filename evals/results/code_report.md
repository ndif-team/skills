# nnsight resource evaluation

336 runs · 48 tasks · 7 conditions · 1 model(s) · 25,965,862 tokens ($36.02 API-equivalent)

## By resource condition

| condition      | pass | 95% CI | n | tok/solve | $-equiv/solve | med s | turns | reads |
|----------------|------|--------|---|-----------|---------------|-------|-------|-------|
| none           | 44% | 31%-58% | 48 | 1,499 | $0.012 | 3 | 1.0 | 0.0 |
| source         | 77% | 63%-87% | 48 | 170,305 | $0.193 | 16 | 6.2 | 5.2 |
| tutorials      | 92% | 80%-97% | 48 | 116,080 | $0.142 | 9 | 4.7 | 3.7 |
| docs           | 94% | 83%-98% | 48 | 88,286 | $0.120 | 10 | 4.1 | 3.1 |
| docs+tutorials | 85% | 73%-93% | 48 | 91,462 | $0.131 | 9 | 4.1 | 3.1 |
| skills         | 94% | 83%-98% | 48 | 77,101 | $0.131 | 8 | 3.4 | 0.4 |
| everything     | 88% | 75%-94% | 48 | 79,363 | $0.136 | 7 | 2.9 | 0.5 |

## By task kind

| kind  | pass | 95% CI | n | tok/solve | $-equiv/solve | med s | turns | reads |
|-------|------|--------|---|-----------|---------------|-------|-------|-------|
| debug | 90% | 83%-95% | 105 | 81,168 | $0.118 | 7 | 3.3 | 1.8 |
| code  | 78% | 72%-83% | 231 | 101,416 | $0.138 | 9 | 4.0 | 2.5 |

## By difficulty

| difficulty   | pass | 95% CI | n | tok/solve | $-equiv/solve | med s | turns | reads |
|--------------|------|--------|---|-----------|---------------|-------|-------|-------|
| basic        | 93% | 85%-97% | 84 | 57,205 | $0.085 | 5 | 2.7 | 1.3 |
| intermediate | 83% | 75%-89% | 112 | 85,414 | $0.124 | 8 | 3.7 | 2.2 |
| advanced     | 74% | 66%-81% | 140 | 130,389 | $0.172 | 10 | 4.6 | 3.0 |

## By model

| model  | pass | 95% CI | n | tok/solve | $-equiv/solve | med s | turns | reads |
|--------|------|--------|---|-----------|---------------|-------|-------|-------|
| sonnet | 82% | 77%-86% | 336 | 94,421 | $0.131 | 8 | 3.8 | 2.3 |

## Condition x kind

| cell                   | pass | 95% CI | n | tok/solve | $-equiv/solve | med s | turns | reads |
|------------------------|------|--------|---|-----------|---------------|-------|-------|-------|
| skills / debug         | 100% | 80%-100% | 15 | 65,358 | $0.123 | 8 | 3.3 | 0.4 |
| docs / debug           | 100% | 80%-100% | 15 | 73,064 | $0.102 | 10 | 3.5 | 2.5 |
| tutorials / debug      | 100% | 80%-100% | 15 | 99,587 | $0.123 | 7 | 4.1 | 3.1 |
| source / debug         | 93% | 70%-99% | 15 | 130,329 | $0.169 | 7 | 4.5 | 3.5 |
| skills / code          | 91% | 76%-97% | 33 | 82,973 | $0.135 | 7 | 3.4 | 0.4 |
| docs / code            | 91% | 76%-97% | 33 | 95,897 | $0.129 | 10 | 4.5 | 3.5 |
| tutorials / code       | 88% | 73%-95% | 33 | 124,610 | $0.151 | 9 | 5.0 | 4.0 |
| everything / code      | 88% | 73%-95% | 33 | 71,739 | $0.127 | 7 | 2.8 | 0.2 |
| docs+tutorials / debug | 87% | 62%-96% | 15 | 80,995 | $0.121 | 8 | 3.5 | 2.5 |
| everything / debug     | 87% | 62%-96% | 15 | 96,368 | $0.155 | 9 | 3.3 | 0.9 |
| docs+tutorials / code  | 85% | 69%-93% | 33 | 96,322 | $0.135 | 9 | 4.4 | 3.4 |
| source / code          | 70% | 53%-83% | 33 | 194,638 | $0.208 | 17 | 7.0 | 6.0 |
| none / debug           | 67% | 42%-85% | 15 | 1,046 | $0.008 | 3 | 1.0 | 0.0 |
| none / code            | 33% | 20%-50% | 33 | 1,910 | $0.016 | 3 | 1.0 | 0.0 |

## Failures

| failure | count |
|---------|-------|
| IndexError | 25 |
| AttributeError | 11 |
| VerifyFailed | 10 |
| NameError | 7 |
| Timeout | 3 |
| TypeError | 2 |
| Unknown | 2 |
| ValueError | 1 |

## What the agents opened

- **docs**: docs (77), CLAUDE.md (7), invoke-and-batching.md (6), generate.md (6), iter-all-next.md (5), transformers-model.md (4), save.md (4), logit-lens.md (3)
- **docs+tutorials**: docs (74), features (6), transformers-model.md (6), generate.md (6), invoke-and-batching.md (6), iter-all-next.md (4), access-and-modify.md (3), logit-lens.md (3)
- **everything**: skill:nnsight:nnsight (17), skill:nnsight:nnsight-debugging (9), docs (4), control-flow.md (3), nnsight (3), test_batching.py (3), skill:nnsight:model-editing-and-lora (2), skill:nnsight:logit-lens (2)
- **skills**: skill:nnsight:nnsight (26), skill:nnsight:nnsight-debugging (10), control-flow.md (4), batching.md (4), access-and-modify.md (3), skill:nnsight:model-editing-and-lora (2), caching-and-scan.md (2), skill:nnsight:logit-lens (2)
- **source**: nnsight (134), tracer.py (26), transformers.py (23), envoy.py (22), iterator.py (5), meta.py (4), backward.py (4), hint.py (3)
- **tutorials**: features (45), docs (30), tutorials (15), 4_multiple_token.ipynb (13), 1_getting.ipynb (8), 8_batching.ipynb (7), 6_modules.ipynb (7), nnsight-0.6.md (6)
