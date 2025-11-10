from __future__ import annotations

try:
    from typer import Exit as _TyperExit
    from typer.main import get_command as _get_typer_command
except ImportError:  # pragma: no cover - Typer ships with the project
    _TyperExit = None  # type: ignore[assignment]
    _get_typer_command = None  # type: ignore[assignment]

from vtm.cli import app as cli_app


def main(argv: list[str] | None = None) -> None:
    """Delegate to the Typer-based CLI summarize subcommand."""

    if _get_typer_command is None or _TyperExit is None:  # pragma: no cover - safety
        raise RuntimeError("Typer is required to invoke the unified CLI")

    if argv is None:
        import sys

        args = list(sys.argv[1:])
    else:
        args = list(argv)

    command = _get_typer_command(cli_app)
    try:
        command.main(
            args=["summarize", *args],
            prog_name="vtm",
            standalone_mode=False,
        )
    except _TyperExit as exc:  # pragma: no cover - propagate exit code
        raise SystemExit(exc.exit_code) from exc


if __name__ == "__main__":
    main()
