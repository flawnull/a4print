"""Printer test page and saved print settings."""

import pytest

from a4print import calibrate, cli


def test_answers_map_to_settings():
    s = calibrate.answers_to_settings(4, "c")
    assert s == {"shadows": round(12 / 255, 3), "midtones": 0.1}
    with pytest.raises(ValueError):
        calibrate.answers_to_settings(0, "A")
    with pytest.raises(ValueError):
        calibrate.answers_to_settings(3, "Z")


def test_saved_settings_become_cli_defaults():
    args = cli.build_parser().parse_args(["x.png"])
    assert (args.shadows, args.midtones) == (0.03, 0.05)       # standard values
    calibrate.save_settings(calibrate.answers_to_settings(6, "D"))
    args = cli.build_parser().parse_args(["x.png"])
    assert (args.shadows, args.midtones) == (round(20 / 255, 3), 0.2)
    args = cli.build_parser().parse_args(["x.png", "--shadows", "0"])
    assert args.shadows == 0                                    # flags still win


def test_broken_settings_file_is_ignored():
    calibrate.settings_path().parent.mkdir(parents=True, exist_ok=True)
    calibrate.settings_path().write_text("{not json")
    assert calibrate.load_settings() == {}


def test_test_page_content():
    page = calibrate.test_page()
    assert page.shape == (3508, 2480, 3)
    assert (page[..., 0] == page[..., 1]).all() and (page[..., 1] == page[..., 2]).all()  # all neutral gray
    # The gray patches get lighter from A to E, as the midtone lift grows.
    values = [int(calibrate.enhance.tone_curve(0, m)[calibrate.STRIPES_GRAY])
              for m in calibrate.MIDTONE_CHOICES.values()]
    assert values == sorted(values) and values[0] == calibrate.STRIPES_GRAY


def test_cli_flow(tmp_path, capsys):
    assert calibrate.main(["-o", str(tmp_path)]) == 0
    assert (tmp_path / "a4print_test_page.pdf").read_bytes().startswith(b"%PDF")
    assert calibrate.main(["--black", "2", "--gray", "B"]) == 0
    assert calibrate.load_settings() == {"shadows": round(6 / 255, 3), "midtones": 0.05}
    assert calibrate.main(["--reset"]) == 0
    assert calibrate.load_settings() == {}
