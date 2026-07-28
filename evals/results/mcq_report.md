# nnsight resource evaluation

224 runs · 32 tasks · 7 conditions · 1 model(s) · 11,437,154 tokens ($16.15 API-equivalent)

## By resource condition

| condition      | pass | 95% CI | n | tok/solve | $-equiv/solve | med s | turns | reads |
|----------------|------|--------|---|-----------|---------------|-------|-------|-------|
| none           | 88% | 72%-95% | 32 | 454 | $0.002 | 2 | 1.0 | 0.0 |
| tutorials      | 88% | 72%-95% | 32 | 71,609 | $0.086 | 6 | 2.8 | 1.8 |
| source         | 97% | 84%-99% | 32 | 69,511 | $0.079 | 7 | 3.0 | 2.0 |
| skills         | 97% | 84%-99% | 32 | 77,935 | $0.115 | 5 | 3.2 | 0.8 |
| everything     | 97% | 84%-99% | 32 | 52,766 | $0.095 | 4 | 2.0 | 0.3 |
| docs           | 100% | 89%-100% | 32 | 50,883 | $0.072 | 5 | 2.2 | 1.2 |
| docs+tutorials | 100% | 89%-100% | 32 | 49,519 | $0.075 | 5 | 2.3 | 1.3 |

## By task kind

| kind | pass | 95% CI | n | tok/solve | $-equiv/solve | med s | turns | reads |
|------|------|--------|---|-----------|---------------|-------|-------|-------|
| mcq  | 95% | 91%-97% | 224 | 53,696 | $0.076 | 5 | 2.3 | 1.1 |

## By difficulty

| difficulty   | pass | 95% CI | n | tok/solve | $-equiv/solve | med s | turns | reads |
|--------------|------|--------|---|-----------|---------------|-------|-------|-------|
| basic        | 100% | 90%-100% | 35 | 38,933 | $0.063 | 4 | 1.8 | 0.6 |
| intermediate | 93% | 85%-97% | 84 | 48,803 | $0.073 | 4 | 2.1 | 0.8 |
| advanced     | 95% | 89%-98% | 105 | 62,679 | $0.083 | 5 | 2.7 | 1.4 |

## By model

| model  | pass | 95% CI | n | tok/solve | $-equiv/solve | med s | turns | reads |
|--------|------|--------|---|-----------|---------------|-------|-------|-------|
| sonnet | 95% | 91%-97% | 224 | 53,696 | $0.076 | 5 | 2.3 | 1.1 |

## Condition x kind

| cell                 | pass | 95% CI | n | tok/solve | $-equiv/solve | med s | turns | reads |
|----------------------|------|--------|---|-----------|---------------|-------|-------|-------|
| docs / mcq           | 100% | 89%-100% | 32 | 50,883 | $0.072 | 5 | 2.2 | 1.2 |
| docs+tutorials / mcq | 100% | 89%-100% | 32 | 49,519 | $0.075 | 5 | 2.3 | 1.3 |
| skills / mcq         | 97% | 84%-99% | 32 | 77,935 | $0.115 | 5 | 3.2 | 0.8 |
| source / mcq         | 97% | 84%-99% | 32 | 69,511 | $0.079 | 7 | 3.0 | 2.0 |
| everything / mcq     | 97% | 84%-99% | 32 | 52,766 | $0.095 | 4 | 2.0 | 0.3 |
| none / mcq           | 88% | 72%-95% | 32 | 454 | $0.002 | 2 | 1.0 | 0.0 |
| tutorials / mcq      | 88% | 72%-95% | 32 | 71,609 | $0.086 | 6 | 2.8 | 1.8 |

## Failures

| failure | count |
|---------|-------|
| WrongChoice | 11 |

## What the agents opened

- **docs**: docs (20), nnsight (4), source.md (1), source-tracing.md (1), scan.md (1), types-and-values.md (1), vllm.md (1), cross-invoke.md (1)
- **docs+tutorials**: docs (27), nnsight (3), vllm.md (2), source.md (1), source-internals.md (1), non-blocking-jobs.md (1), iter-all-next.md (1), iteration.md (1)
- **everything**: nnsight (4), skill:nnsight:nnsight (3), skill:nnsight:nnsight-debugging (2), tracer.py (2), source-tracing.md (1), skill:nnsight:attention-analysis (1), skill:nnsight:sae-and-dictionary-learning (1), skill:nnsight:vllm (1)
- **skills**: skill:nnsight:nnsight-debugging (9), skill:nnsight:nnsight (9), nnsight (7), execution-model.md (4), api-reference.md (4), batching.md (2), *.py (2), source-tracing.md (1)
- **source**: nnsight (32), interleaver.py (7), envoy.py (7), tracer.py (4), vllm.py (3), iterator.py (2), source.py (1), remote.py (1)
- **tutorials**: docs (21), features (9), 9_empty_invokers.ipynb (3), extending-nnsight.md (3), index.md (3), tutorials (2), 11_source.ipynb (2), 16_vllm_support.ipynb (2)
