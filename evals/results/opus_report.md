# nnsight resource evaluation

560 runs · 80 tasks · 7 conditions · 1 model(s) · 36,383,734 tokens ($92.71 API-equivalent)

## By resource condition

| condition      | pass | 95% CI | n | tok/solve | $-equiv/solve | med s | turns | reads |
|----------------|------|--------|---|-----------|---------------|-------|-------|-------|
| none           | 66% | 55%-76% | 80 | 1,045 | $0.015 | 4 | 1.0 | 0.0 |
| source         | 96% | 90%-99% | 80 | 135,282 | $0.288 | 19 | 6.0 | 5.0 |
| tutorials      | 92% | 85%-97% | 80 | 77,753 | $0.182 | 13 | 4.5 | 3.5 |
| docs           | 96% | 90%-99% | 80 | 69,013 | $0.183 | 13 | 4.2 | 3.1 |
| docs+tutorials | 95% | 88%-98% | 80 | 71,956 | $0.189 | 12 | 4.5 | 3.5 |
| skills         | 92% | 85%-97% | 80 | 61,882 | $0.184 | 9 | 3.7 | 0.4 |
| everything     | 99% | 93%-100% | 80 | 60,708 | $0.179 | 10 | 3.4 | 1.1 |

## By task kind

| kind  | pass | 95% CI | n | tok/solve | $-equiv/solve | med s | turns | reads |
|-------|------|--------|---|-----------|---------------|-------|-------|-------|
| mcq   | 93% | 89%-96% | 224 | 48,424 | $0.129 | 8 | 3.0 | 1.8 |
| debug | 92% | 86%-96% | 105 | 70,320 | $0.189 | 15 | 4.1 | 2.3 |
| code  | 89% | 84%-92% | 231 | 95,075 | $0.232 | 13 | 4.6 | 3.0 |

## By difficulty

| difficulty   | pass | 95% CI | n | tok/solve | $-equiv/solve | med s | turns | reads |
|--------------|------|--------|---|-----------|---------------|-------|-------|-------|
| basic        | 100% | 97%-100% | 119 | 54,792 | $0.146 | 11 | 3.5 | 2.0 |
| intermediate | 88% | 83%-92% | 182 | 74,301 | $0.192 | 11 | 3.9 | 2.4 |
| advanced     | 89% | 84%-92% | 259 | 77,830 | $0.193 | 12 | 4.1 | 2.6 |

## By model

| model | pass | 95% CI | n | tok/solve | $-equiv/solve | med s | turns | reads |
|-------|------|--------|---|-----------|---------------|-------|-------|-------|
| opus  | 91% | 88%-93% | 560 | 71,341 | $0.182 | 11 | 3.9 | 2.4 |

## Condition x kind

| cell                   | pass | 95% CI | n | tok/solve | $-equiv/solve | med s | turns | reads |
|------------------------|------|--------|---|-----------|---------------|-------|-------|-------|
| skills / debug         | 100% | 80%-100% | 15 | 63,175 | $0.202 | 12 | 5.0 | 0.3 |
| docs / mcq             | 100% | 89%-100% | 32 | 48,695 | $0.123 | 10 | 3.4 | 2.3 |
| tutorials / debug      | 100% | 80%-100% | 15 | 80,130 | $0.192 | 20 | 4.6 | 3.6 |
| source / debug         | 100% | 80%-100% | 15 | 96,654 | $0.213 | 23 | 5.3 | 4.3 |
| source / mcq           | 100% | 89%-100% | 32 | 50,073 | $0.118 | 8 | 3.4 | 2.4 |
| docs+tutorials / mcq   | 100% | 89%-100% | 32 | 52,466 | $0.138 | 10 | 3.6 | 2.6 |
| everything / debug     | 100% | 80%-100% | 15 | 69,324 | $0.208 | 12 | 4.3 | 1.1 |
| everything / mcq       | 100% | 89%-100% | 32 | 51,225 | $0.161 | 10 | 2.8 | 1.7 |
| skills / code          | 97% | 85%-99% | 33 | 66,235 | $0.186 | 11 | 3.9 | 0.5 |
| docs / code            | 97% | 85%-99% | 33 | 80,754 | $0.221 | 13 | 4.8 | 3.8 |
| everything / code      | 97% | 85%-99% | 33 | 66,153 | $0.183 | 9 | 3.7 | 0.5 |
| tutorials / code       | 94% | 80%-98% | 33 | 80,115 | $0.192 | 14 | 4.8 | 3.8 |
| docs+tutorials / code  | 94% | 80%-98% | 33 | 90,125 | $0.231 | 14 | 5.3 | 4.3 |
| source / code          | 91% | 76%-97% | 33 | 245,486 | $0.507 | 32 | 8.9 | 7.9 |
| tutorials / mcq        | 88% | 72%-95% | 32 | 73,865 | $0.166 | 11 | 4.0 | 3.0 |
| docs / debug           | 87% | 62%-96% | 15 | 90,129 | $0.239 | 19 | 4.4 | 3.4 |
| docs+tutorials / debug | 87% | 62%-96% | 15 | 76,602 | $0.211 | 15 | 4.5 | 3.5 |
| skills / mcq           | 84% | 68%-93% | 32 | 56,005 | $0.173 | 4 | 2.9 | 0.4 |
| none / mcq             | 78% | 61%-89% | 32 | 529 | $0.005 | 3 | 1.0 | 0.0 |
| none / debug           | 73% | 48%-89% | 15 | 1,295 | $0.021 | 9 | 1.0 | 0.0 |
| none / code            | 52% | 35%-67% | 33 | 1,641 | $0.027 | 7 | 1.0 | 0.0 |

## Failures

| failure | count |
|---------|-------|
| WrongChoice | 16 |
| IndexError | 12 |
| AttributeError | 9 |
| NameError | 7 |
| VerifyFailed | 3 |
| TypeError | 2 |
| Unknown | 1 |

## What the agents opened

- **docs**: docs (81), CLAUDE.md (53), nnsight (26), save.md (8), access-and-modify.md (7), generate.md (6), barrier.md (5), iter-all-next.md (5)
- **docs+tutorials**: docs (60), features (36), nnsight (30), CLAUDE.md (29), wd (22), save.md (7), modification.md (6), invoke-and-batching.md (6)
- **everything**: skill:nnsight:nnsight (29), nnsight (28), docs (15), skill:nnsight:nnsight-debugging (12), control-flow.md (5), features (5), skills (4), caching-and-scan.md (3)
- **skills**: skill:nnsight:nnsight (51), skill:nnsight:nnsight-debugging (23), control-flow.md (6), batching.md (5), skill:nnsight:attention-analysis (3), nnsight (3), skills (3), skill:nnsight:model-editing-and-lora (3)
- **source**: nnsight (198), tracer.py (52), envoy.py (39), transformers.py (25), interleaver.py (8), eproperty.py (7), __init__.py (6), iterator.py (6)
- **tutorials**: features (88), docs (86), 8_batching.ipynb (11), 2_setting.ipynb (11), 1_getting.ipynb (9), tutorials (8), 4_multiple_token.ipynb (7), 15_remote_execution.ipynb (6)
