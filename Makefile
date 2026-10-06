# DemandCast developer targets. Every command goes through $(PYTHON), so the same Makefile works
# with a virtualenv interpreter (`make PYTHON=path/to/python check`) and with plain `python3`
# in CI. `make check` = lint + typecheck + test; `make smoke` mirrors the CI smoke run.
PYTHON ?= python3
DB ?= demandcast.db
DEMO_DB ?= demo.db
OUT ?= out
WORKERS ?= 2
HTML ?= dashboard/index.html

.PHONY: install test lint format typecheck check init run dashboard verify-dashboard smoke demo bench clean all

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q -rs

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

typecheck:
	$(PYTHON) -m mypy demandcast scripts/verify_dashboard.py benchmarks/bench.py

check: lint typecheck test

init:
	$(PYTHON) -m demandcast --db $(DB) init

run:
	$(PYTHON) -m demandcast --db $(DB) run --workers $(WORKERS)

dashboard:
	$(PYTHON) -m demandcast --db $(DB) dashboard --out $(HTML)

verify-dashboard:
	$(PYTHON) scripts/verify_dashboard.py --html $(HTML) --vercel vercel.json

# The CI check sequence: synthetic data with snapshots and future promotions -> backdated run ->
# evaluation against realised sales -> budgeted run with per-class service levels -> queries ->
# exports -> load round-trip -> dashboard -> response-header verification -> quick benchmark.
smoke:
	rm -f ci.db ci.db-wal ci.db-shm rt.db rt.db-wal rt.db-shm
	$(PYTHON) -m demandcast --db ci.db init --stores 3 --products 8 --days 300 --snapshot-every 7 --future-promo-days 14
	$(PYTHON) -m demandcast --db ci.db run --horizon 14 --folds 2 --workers $(WORKERS) --cutoff 2024-10-12
	$(PYTHON) -m demandcast --db ci.db evaluate --json
	$(PYTHON) -m demandcast --db ci.db run --horizon 14 --folds 2 --workers $(WORKERS) --order-budget 20000 --service-level-by-class A=0.98,B=0.95,C=0.90
	$(PYTHON) -m demandcast --db ci.db runs
	$(PYTHON) -m demandcast --db ci.db query forecast_accuracy_leaderboard
	$(PYTHON) -m demandcast --db ci.db query evaluation_history
	$(PYTHON) -m demandcast --db ci.db export orders --out $(OUT)/orders.csv
	$(PYTHON) -m demandcast --db ci.db export dataset --out $(OUT)/dataset
	$(PYTHON) -m demandcast --db rt.db load --dir $(OUT)/dataset
	$(PYTHON) -m demandcast --db rt.db run --horizon 14 --folds 2 --workers $(WORKERS)
	$(PYTHON) -m demandcast --db ci.db dashboard --out $(OUT)/index.html
	$(PYTHON) scripts/verify_dashboard.py --html $(OUT)/index.html --vercel vercel.json --json
	$(PYTHON) benchmarks/bench.py --quick

# End-to-end demo on a fresh database: weekly inventory snapshots and 28 days of scheduled
# promotions, a backdated run evaluated against the sales that followed, then the current run
# and the dashboard. 365 days from 2024-01-01 end on 2024-12-30, so a cutoff of 2024-12-02
# leaves exactly one 28-day horizon of realised sales for `evaluate`.
demo:
	rm -f $(DEMO_DB) $(DEMO_DB)-wal $(DEMO_DB)-shm
	$(PYTHON) -m demandcast --db $(DEMO_DB) init --stores 3 --products 12 --days 365 --snapshot-every 7 --future-promo-days 28
	$(PYTHON) -m demandcast --db $(DEMO_DB) run --horizon 28 --folds 3 --workers $(WORKERS) --cutoff 2024-12-02
	$(PYTHON) -m demandcast --db $(DEMO_DB) evaluate
	$(PYTHON) -m demandcast --db $(DEMO_DB) run --horizon 28 --folds 3 --workers $(WORKERS)
	$(PYTHON) -m demandcast --db $(DEMO_DB) runs
	$(PYTHON) -m demandcast --db $(DEMO_DB) dashboard --out $(OUT)/demo.html
	$(PYTHON) scripts/verify_dashboard.py --html $(OUT)/demo.html --vercel vercel.json

bench:
	$(PYTHON) benchmarks/bench.py --out benchmarks/results/bench.json

all: init run dashboard

clean:
	rm -f $(DB) $(DB)-wal $(DB)-shm ci.db ci.db-wal ci.db-shm rt.db rt.db-wal rt.db-shm
	rm -f $(DEMO_DB) $(DEMO_DB)-wal $(DEMO_DB)-shm
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info $(OUT) benchmarks/results
