.PHONY: reproduce verify figures clean

PYTHON ?= python3

reproduce: verify figures

verify:
	$(PYTHON) scripts/verify_bundle.py
	$(PYTHON) scripts/validate_results.py

figures:
	$(PYTHON) scripts/render_figures.py

clean:
	rm -rf build
