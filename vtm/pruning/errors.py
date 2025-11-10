from __future__ import annotations


class PruningError(RuntimeError):
    """Base error for pruning failures."""


class PrunedTreeComputationError(PruningError):
    """Raised when a tree cannot be pruned successfully."""
