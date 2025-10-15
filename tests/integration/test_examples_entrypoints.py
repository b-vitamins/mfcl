import importlib


def test_import_examples_entrypoints():
    modules = [
        # SimCLR
        "examples.selfsupervised.imagenet1k.resnet18_160.simclr.train",
        "examples.selfsupervised.imagenet1k.resnet18_160.simclr.evallinear",
        "examples.selfsupervised.imagenet1k.resnet18_160.simclr.visual",
        # MoCo
        "examples.selfsupervised.imagenet1k.resnet18_160.moco.train",
        "examples.selfsupervised.imagenet1k.resnet18_160.moco.evallinear",
        "examples.selfsupervised.imagenet1k.resnet18_160.moco.visual",
        # BYOL
        "examples.selfsupervised.imagenet1k.resnet18_160.byol.train",
        "examples.selfsupervised.imagenet1k.resnet18_160.byol.evallinear",
        "examples.selfsupervised.imagenet1k.resnet18_160.byol.visual",
        # SimSiam
        "examples.selfsupervised.imagenet1k.resnet18_160.simsiam.train",
        "examples.selfsupervised.imagenet1k.resnet18_160.simsiam.evallinear",
        "examples.selfsupervised.imagenet1k.resnet18_160.simsiam.visual",
        # SwAV
        "examples.selfsupervised.imagenet1k.resnet18_160.swav.train",
        "examples.selfsupervised.imagenet1k.resnet18_160.swav.evallinear",
        "examples.selfsupervised.imagenet1k.resnet18_160.swav.visual",
        # Barlow
        "examples.selfsupervised.imagenet1k.resnet18_160.barlow.train",
        "examples.selfsupervised.imagenet1k.resnet18_160.barlow.evallinear",
        "examples.selfsupervised.imagenet1k.resnet18_160.barlow.visual",
        # VICReg
        "examples.selfsupervised.imagenet1k.resnet18_160.vicreg.train",
        "examples.selfsupervised.imagenet1k.resnet18_160.vicreg.evallinear",
        "examples.selfsupervised.imagenet1k.resnet18_160.vicreg.visual",
    ]
    for m in modules:
        importlib.import_module(m)
