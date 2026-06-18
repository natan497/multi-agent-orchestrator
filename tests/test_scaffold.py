"""Phase 0 smoke test: the package is importable and installed correctly.

Real coverage arrives with each feature phase; this just keeps the suite green
and proves the editable install + src layout resolve.
"""

import orchestrator


def test_package_imports_and_has_version():
    assert orchestrator.__version__ == "0.1.0"


def test_subpackages_importable():
    import providers  # noqa: F401
    import tools  # noqa: F401
