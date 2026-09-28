from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _iter_snippet_files(snippets_dir: Path) -> list[Path]:
    files = []
    for path in sorted(snippets_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        files.append(path)
    return files


def _render_include(
    *, rel_snippet_path: str, source: str, output: str, commit: str | None
) -> str:
    header = (
        f"This page fragment is generated from `{rel_snippet_path}`.\n"
        "It is intended to show the exact code and typical output for this version.\n"
    )
    if commit:
        header += f"\nCommit: `{commit}`\n"
    return (
        f"{header}\n"
        "## Program\n\n"
        "```python\n"
        f"{source.rstrip()}\n"
        "```\n\n"
        "## Output\n\n"
        "```text\n"
        f"{output.rstrip()}\n"
        "```\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-exec",
        action="store_true",
        help="Do not execute snippets; emit placeholder output blocks (used in PR docs checks).",
    )
    args = parser.parse_args(argv)

    root = _repo_root()
    snippets_dir = root / "docs" / "snippets"
    includes_dir = root / "docs" / "_includes" / "snippets"
    includes_dir.mkdir(parents=True, exist_ok=True)

    snippet_files = _iter_snippet_files(snippets_dir)
    if not snippet_files:
        raise SystemExit(f"No snippet programs found in {snippets_dir}")

    # Clear previously generated includes to avoid stale files.
    for old in includes_dir.glob("*.md"):
        old.unlink()

    commit = os.environ.get("GITHUB_SHA") or os.environ.get("CI_COMMIT_SHA")

    env = dict(os.environ)
    # Ensure repo root is available for local imports (snippets use relative data paths too).
    env.setdefault("PYTHONPATH", str(root))

    for snippet in snippet_files:
        rel_snippet_path = snippet.relative_to(root).as_posix()
        source = snippet.read_text(encoding="utf-8")
        name = snippet.stem

        if args.no_exec:
            output = (
                "Output is not generated in this build.\n"
                "For real output, run this snippet in an environment where `mifrost` and `pymimir` are installed.\n"
            )
        else:
            proc = subprocess.run(
                [sys.executable, "-m", "docs.snippets." + name],
                cwd=str(root),
                env=env,
                text=True,
                capture_output=True,
            )
            combined = proc.stdout
            if proc.stderr:
                combined += "\n[stderr]\n" + proc.stderr
            if proc.returncode != 0:
                raise SystemExit(
                    f"Snippet failed ({rel_snippet_path}) with exit code {proc.returncode}:\n{combined}"
                )
            output = combined

        out_path = includes_dir / f"{name}.md"
        out_path.write_text(
            _render_include(
                rel_snippet_path=rel_snippet_path,
                source=source,
                output=output,
                commit=commit,
            ),
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
