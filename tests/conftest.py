"""
Pytest configuration and global test hooks.
Prevents unintended live API token consumption by skipping live API tests by default.
"""

import os
import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom command-line options."""
    parser.addoption(
        "--run-live-api",
        action="store_true",
        default=False,
        help="Run live API integration tests that make outbound network requests and consume tokens.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "live_api: mark test as requiring a live external API connection (skipped by default)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """
    Skip tests marked with 'live_api' unless --run-live-api flag is passed
    or the RUN_LIVE_API environment variable is set to true/1.
    """
    run_live = config.getoption("--run-live-api") or os.getenv("RUN_LIVE_API", "").lower() in ("true", "1")
    if run_live:
        return

    skip_live = pytest.mark.skip(
        reason="Live API tests are skipped by default to save tokens and prevent costs. "
               "Pass '--run-live-api' or set RUN_LIVE_API=true to run."
    )
    for item in items:
        if "live_api" in item.keywords:
            item.add_marker(skip_live)
