CONDA_ENV ?= ndif2
NDIF_HOST ?= http://localhost:8001
PYTEST ?= conda run --no-capture-output -n $(CONDA_ENV) pytest

.PHONY: test test-local test-structure test-remote report

## Full local run: every code block, including the ones that need a GPU and NDIF.
test:
	NDIF_HOST=$(NDIF_HOST) $(PYTEST) -q

## Everything that does not need an NDIF deployment.
test-local:
	$(PYTEST) -q

## Packaging only — fast, no model loading.
test-structure:
	$(PYTEST) -q tests/test_structure.py

## One skill: make test-skill SKILL=nnsight
test-skill:
	NDIF_HOST=$(NDIF_HOST) $(PYTEST) -q tests/test_skills.py -k "$(SKILL)"

## Print the per-block verification table from the last run.
report:
	@python tests/report.py
