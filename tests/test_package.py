"""The package imports and declares a version. Keeps CI meaningful from day one."""

import agent_harness


def test_package_has_a_version() -> None:
    assert agent_harness.__version__
