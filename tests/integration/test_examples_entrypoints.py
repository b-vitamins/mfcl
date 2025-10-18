import importlib


def test_import_examples_entrypoints():
    modules = [
        "train",
        "eval_linear",
        "scripts.plot_metrics",
    ]
    for m in modules:
        importlib.import_module(m)
