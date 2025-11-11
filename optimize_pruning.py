"""Compatibility wrapper for the pruning optimizer entry-point."""

from vtm.optimize_pruning import *  # noqa: F401,F403

if __name__ == "__main__":
    main()  # type: ignore[name-defined]
