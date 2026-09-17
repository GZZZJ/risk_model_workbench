PYTHON ?= .venv/bin/python
RMW ?= .venv/bin/rmw

.PHONY: doctor test compile

doctor:
	$(RMW) doctor

test:
	$(PYTHON) -m pytest tests -q

compile:
	env PYTHONPYCACHEPREFIX=/private/tmp/jingying_agent_pycache $(PYTHON) -m compileall agent.py src tests projects/2026-05-fujie-gcard-v1/legacy_scripts
