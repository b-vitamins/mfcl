from mfcl.utils.consolemonitor import ConsoleMonitor


def test_consolemonitor_live_and_summary(capsys):
    c = ConsoleMonitor()
    c.live(1, 1, 10, {"loss": 1.0, "lr": 0.1})
    out = capsys.readouterr().out
    assert "\n" not in out
    c.newline()
    c.summary(1, {"loss": 1.0, "lr": 0.1})
    out = capsys.readouterr().out
    assert "[epoch" in out and "loss=" in out
