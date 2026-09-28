DB ?= demandcast.db

.PHONY: install test lint init run dashboard clean all

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check . && ruff format --check .

init:
	demandcast --db $(DB) init

run:
	demandcast --db $(DB) run

dashboard:
	demandcast --db $(DB) dashboard --out dashboard/index.html

all: init run dashboard

clean:
	rm -f $(DB) $(DB)-wal $(DB)-shm
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
