from __future__ import annotations

import pytest

from scripts.install_pymimir_release import (
    _platform_tag,
    _python_tag,
    _selected_release,
    main,
    release_wheel_url,
    wheel_filename,
)


def test_python_tags_cover_gil_and_free_threaded_builds() -> None:
    assert _python_tag(3, 13, free_threaded=False) == "cp313"
    assert _python_tag(3, 14, free_threaded=True) == "cp314t"


@pytest.mark.parametrize(
    ("system", "machine", "expected"),
    [
        ("Darwin", "arm64", "macosx_11_0_arm64"),
        ("Linux", "x86_64", "manylinux_2_27_x86_64.manylinux_2_28_x86_64"),
    ],
)
def test_release_platform_tags(system: str, machine: str, expected: str) -> None:
    assert _platform_tag(system, machine) == expected


def test_unsupported_release_platform_is_explicit() -> None:
    with pytest.raises(RuntimeError, match="No published pymimir release wheel"):
        _platform_tag("Windows", "AMD64")


def test_release_wheel_url_uses_owner_release_asset() -> None:
    assert release_wheel_url(
        "v0.15.0",
        repository="maichmueller/mimir",
        python_tag="cp312",
        system="Darwin",
        machine="arm64",
    ) == (
        "https://github.com/maichmueller/mimir/releases/download/v0.15.0/"
        "pymimir-0.15.0-cp312-cp312-macosx_11_0_arm64.whl"
    )


def test_wheel_filename_handles_free_threaded_release() -> None:
    assert wheel_filename(
        "0.15.0",
        python_tag="cp314t",
        system="Linux",
        machine="x86_64",
    ) == (
        "pymimir-0.15.0-cp314t-cp314t-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
    )


def test_selected_release_cannot_drop_below_abi_floor() -> None:
    with pytest.raises(ValueError, match="below the supported minimum"):
        _selected_release("0.14.3")


def test_dry_run_does_not_install(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "--dry-run",
                "--version",
                "0.15.0",
                "--repository",
                "maichmueller/mimir",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip().endswith(wheel_filename("0.15.0"))
