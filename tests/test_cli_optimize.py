import pytest

pytest.importorskip("typer")

from typer.testing import CliRunner

app = pytest.importorskip("vtm.cli").app


def test_optimize_pruning_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["optimize-pruning", "--help"])

    assert result.exit_code == 0
    assert "Optimize pruning configuration parameters" in result.stdout

