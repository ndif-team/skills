# nnsight resource evaluation

224 runs · 32 tasks · 7 conditions · 1 model(s) · 11,247,274 tokens ($16.09 API-equivalent)

## By resource condition

| condition      | pass | 95% CI | n | tok/solve | $-equiv/solve | med s | turns | reads |
|----------------|------|--------|---|-----------|---------------|-------|-------|-------|
| none           | 88% | 72%-95% | 32 | 456 | $0.002 | 2 | 1.0 | 0.0 |
| source         | 97% | 84%-99% | 32 | 77,552 | $0.088 | 7 | 3.2 | 2.2 |
| tutorials      | 94% | 80%-98% | 32 | 61,508 | $0.075 | 5 | 2.5 | 1.5 |
| docs           | 100% | 89%-100% | 32 | 57,492 | $0.076 | 6 | 2.4 | 1.4 |
| docs+tutorials | 100% | 89%-100% | 32 | 48,621 | $0.074 | 5 | 2.2 | 1.2 |
| skills         | 97% | 84%-99% | 32 | 63,862 | $0.107 | 5 | 2.8 | 0.4 |
| everything     | 97% | 84%-99% | 32 | 51,929 | $0.095 | 3 | 1.9 | 0.3 |

## By task kind

| kind | pass | 95% CI | n | tok/solve | $-equiv/solve | med s | turns | reads |
|------|------|--------|---|-----------|---------------|-------|-------|-------|
| mcq  | 96% | 93%-98% | 224 | 52,313 | $0.075 | 5 | 2.3 | 1.0 |

## By difficulty

| difficulty   | pass | 95% CI | n | tok/solve | $-equiv/solve | med s | turns | reads |
|--------------|------|--------|---|-----------|---------------|-------|-------|-------|
| basic        | 100% | 90%-100% | 35 | 38,933 | $0.063 | 4 | 1.8 | 0.6 |
| intermediate | 93% | 86%-97% | 91 | 48,767 | $0.072 | 5 | 2.1 | 0.8 |
| advanced     | 97% | 91%-99% | 98 | 60,415 | $0.082 | 5 | 2.6 | 1.3 |

## By model

| model  | pass | 95% CI | n | tok/solve | $-equiv/solve | med s | turns | reads |
|--------|------|--------|---|-----------|---------------|-------|-------|-------|
| sonnet | 96% | 93%-98% | 224 | 52,313 | $0.075 | 5 | 2.3 | 1.0 |

## Condition x kind

| cell                 | pass | 95% CI | n | tok/solve | $-equiv/solve | med s | turns | reads |
|----------------------|------|--------|---|-----------|---------------|-------|-------|-------|
| docs / mcq           | 100% | 89%-100% | 32 | 57,492 | $0.076 | 6 | 2.4 | 1.4 |
| docs+tutorials / mcq | 100% | 89%-100% | 32 | 48,621 | $0.074 | 5 | 2.2 | 1.2 |
| skills / mcq         | 97% | 84%-99% | 32 | 63,862 | $0.107 | 5 | 2.8 | 0.4 |
| source / mcq         | 97% | 84%-99% | 32 | 77,552 | $0.088 | 7 | 3.2 | 2.2 |
| everything / mcq     | 97% | 84%-99% | 32 | 51,929 | $0.095 | 3 | 1.9 | 0.3 |
| tutorials / mcq      | 94% | 80%-98% | 32 | 61,508 | $0.075 | 5 | 2.5 | 1.5 |
| none / mcq           | 88% | 72%-95% | 32 | 456 | $0.002 | 2 | 1.0 | 0.0 |

## Failures

| failure | count |
|---------|-------|
| WrongChoice | 9 |

## What the agents opened

- **docs**: docs (18), nnsight (7), glossary.md (3), envoy.py (2), source.md (1), source-tracing.md (1), eproperty.py (1), interleaver.py (1)
- **docs+tutorials**: docs (24), nnsight (5), vllm.md (2), source.md (1), source-internals.md (1), eproperty.py (1), non-blocking-jobs.md (1), iter-all-next.md (1)
- **everything**: skill:nnsight:nnsight (3), nnsight (3), skill:nnsight:nnsight-debugging (2), tracer.py (2), source-tracing.md (1), skill:nnsight:attention-analysis (1), eproperty.py (1), skill:nnsight:sae-and-dictionary-learning (1)
- **skills**: skill:nnsight:nnsight-debugging (9), skill:nnsight:nnsight (9), nnsight (4), batching.md (2), source-tracing.md (1), skill:nnsight:attention-analysis (1), eproperty (1), *.py (1)
- **source**: nnsight (33), envoy.py (10), interleaver.py (9), tracer.py (4), vllm.py (3), iterator.py (2), source.py (1), eproperty.py (1)
- **tutorials**: docs (20), features (9), 9_empty_invokers.ipynb (3), tutorials (2), 11_source.ipynb (2), 16_vllm_support.ipynb (2), llms.txt (2), 15_remote_execution.ipynb (1)
