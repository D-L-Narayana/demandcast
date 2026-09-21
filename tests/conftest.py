from datetime import date

import pytest

from demandcast import db
from demandcast.simulate import SimConfig, generate


@pytest.fixture(scope="session")
def small_db():
    """In-memory database with a compact synthetic dataset shared across the session."""
    conn = db.connect(":memory:")
    db.init_schema(conn)
    generate(conn, SimConfig(n_stores=3, n_products=8, start=date(2024, 1, 1), days=300, seed=7))
    return conn


@pytest.fixture
def fresh_db():
    conn = db.connect(":memory:")
    db.init_schema(conn)
    return conn
