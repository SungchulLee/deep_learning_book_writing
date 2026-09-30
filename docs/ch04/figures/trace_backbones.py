"""등뼈 다섯의 **층마다** 텐서 모양을 세 가지 입력 크기로 잰다.

그림에 쓸 값을 손으로 적지 않기 위한 스크립트다. 결과는 backbone_shapes.json
으로 떨어지고, make_backbone_figures.py 가 그것을 읽어 그린다. 그러므로
그림에 적힌 모양은 모두 이 파일이 실제로 재어 본 값이다.

실행:
    python trace_backbones.py
"""

import json

import torch
import torch.nn as nn
import torchvision.models as M


def trace(build, stages, sizes):
    """stages: [(라벨, 상자 글, 모듈 경로)] — 그 모듈을 지나고 난 모양을 잰다.

    도중에 터지면 거기까지 잰 것만 남기고 어디서 터졌는지를 함께 적는다.
    빈칸은 그림에서 붉은 "--" 로 나온다.
    """
    out = {}
    for s in sizes:
        m = build()
        m.eval()
        rec, handles = {}, []
        for lab, _txt, path in stages:
            mod = m
            for part in path.split("."):
                mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
            handles.append(mod.register_forward_hook(
                lambda _m, _i, o, lab=lab: rec.__setitem__(
                    lab, "x".join(map(str, tuple(o.shape)[1:])))))
        err = None
        try:
            with torch.no_grad():
                m(torch.zeros(1, 3, s, s))
        except Exception as e:                       # noqa: BLE001
            err = type(e).__name__
        for h in handles:
            h.remove()
        out[str(s)] = {"shapes": rec, "error": err}
    return out


# (라벨, 상자에 적을 글, 모듈 경로)
RESNET = [("input", "Input", None)] + [
    ("conv1",   "7x7 conv, 64, stride 2", "conv1"),
    ("maxpool", "MaxPool 3x3, stride 2",  "maxpool"),
    ("layer1.0", "BasicBlock, 64",  "layer1.0"),
    ("layer1.1", "BasicBlock, 64",  "layer1.1"),
    ("layer2.0", "BasicBlock, 128  (stride 2)", "layer2.0"),
    ("layer2.1", "BasicBlock, 128", "layer2.1"),
    ("layer3.0", "BasicBlock, 256  (stride 2)", "layer3.0"),
    ("layer3.1", "BasicBlock, 256", "layer3.1"),
    ("layer4.0", "BasicBlock, 512  (stride 2)", "layer4.0"),
    ("layer4.1", "BasicBlock, 512", "layer4.1"),
    ("avgpool", "Global AvgPool", "avgpool"),
    ("fc",      "FC 1000",        "fc"),
]

INCEPTION = [("input", "Input", None)] + [
    ("Conv2d_1a", "3x3 conv, 32, stride 2", "Conv2d_1a_3x3"),
    ("Conv2d_2a", "3x3 conv, 32",           "Conv2d_2a_3x3"),
    ("Conv2d_2b", "3x3 conv, 64, pad 1",    "Conv2d_2b_3x3"),
    ("maxpool1",  "MaxPool 3x3, stride 2",  "maxpool1"),
    ("Conv2d_3b", "1x1 conv, 80",           "Conv2d_3b_1x1"),
    ("Conv2d_4a", "3x3 conv, 192",          "Conv2d_4a_3x3"),
    ("maxpool2",  "MaxPool 3x3, stride 2",  "maxpool2"),
    ("Mixed_5b",  "Inception block",  "Mixed_5b"),
    ("Mixed_5c",  "Inception block",  "Mixed_5c"),
    ("Mixed_5d",  "Inception block",  "Mixed_5d"),
    ("Mixed_6a",  "Inception block  (stride 2)", "Mixed_6a"),
    ("Mixed_6b",  "Inception block",  "Mixed_6b"),
    ("Mixed_6c",  "Inception block",  "Mixed_6c"),
    ("Mixed_6d",  "Inception block",  "Mixed_6d"),
    ("Mixed_6e",  "Inception block",  "Mixed_6e"),
    ("Mixed_7a",  "Inception block  (stride 2)", "Mixed_7a"),
    ("Mixed_7b",  "Inception block",  "Mixed_7b"),
    ("Mixed_7c",  "Inception block",  "Mixed_7c"),
    ("avgpool",   "Global AvgPool",   "avgpool"),
    ("fc",        "FC 1000",          "fc"),
]

MOBILENET = [("input", "Input", None)] + \
    [("features.0", "3x3 conv, 16, stride 2", "features.0")] + \
    [(f"features.{i}", f"InvertedResidual {i}", f"features.{i}") for i in range(1, 16)] + \
    [("features.16", "1x1 conv, 960", "features.16"),
     ("avgpool",     "Global AvgPool", "avgpool"),
     ("classifier",  "FC 1280 - FC 1000", "classifier")]

EFFICIENTNET = [("input", "Input", None)] + \
    [("features.0", "3x3 conv, 32, stride 2", "features.0")] + \
    [(f"features.{i}", lab, f"features.{i}") for i, lab in
     zip(range(1, 8), ["MBConv x1, 16", "MBConv x2, 24", "MBConv x2, 40",
                       "MBConv x3, 80", "MBConv x3, 112", "MBConv x4, 192",
                       "MBConv x1, 320"])] + \
    [("features.8", "1x1 conv, 1280", "features.8"),
     ("avgpool",    "Global AvgPool", "avgpool"),
     ("classifier", "FC 1000",        "classifier")]

VIT = [("input", "Input", None)] + \
    [("conv_proj", "16x16 conv, 768, stride 16", "conv_proj")] + \
    [(f"block {i}", f"Transformer block {i}", f"encoder.layers.encoder_layer_{i}")
     for i in range(12)] + \
    [("encoder.ln", "LayerNorm", "encoder.ln"),
     ("heads",      "FC 1000",   "heads")]


DENSENET = [("input", "Input", None)] + [
    ("conv0",        "7x7 conv, 64, stride 2", "features.conv0"),
    ("pool0",        "MaxPool 3x3, stride 2",  "features.pool0"),
    ("denseblock1",  "DenseBlock x6",          "features.denseblock1"),
    ("transition1",  "1x1 conv + AvgPool",     "features.transition1"),
    ("denseblock2",  "DenseBlock x12",         "features.denseblock2"),
    ("transition2",  "1x1 conv + AvgPool",     "features.transition2"),
    ("denseblock3",  "DenseBlock x24",         "features.denseblock3"),
    ("transition3",  "1x1 conv + AvgPool",     "features.transition3"),
    ("denseblock4",  "DenseBlock x16",         "features.denseblock4"),
    ("norm5",        "BatchNorm",              "features.norm5"),
    ("classifier",   "FC 1000",                "classifier"),
]

CONVNEXT = [("input", "Input", None)] + [
    ("features.0", "4x4 conv, 96, stride 4  (patchify stem)", "features.0"),
    ("features.1", "ConvNeXt block x3, 96",                   "features.1"),
    ("features.2", "downsample -> 192",                       "features.2"),
    ("features.3", "ConvNeXt block x3, 192",                  "features.3"),
    ("features.4", "downsample -> 384",                       "features.4"),
    ("features.5", "ConvNeXt block x9, 384",                  "features.5"),
    ("features.6", "downsample -> 768",                       "features.6"),
    ("features.7", "ConvNeXt block x3, 768",                  "features.7"),
    ("avgpool",    "Global AvgPool",                          "avgpool"),
    ("classifier", "LayerNorm - Flatten - FC 1000",           "classifier"),
]

SPECS = {
    "resnet18":     (M.resnet18, RESNET, [224, 32, 448]),
    "inception":    (lambda: M.inception_v3(init_weights=False), INCEPTION, [299, 32, 598]),
    "mobilenet":    (M.mobilenet_v3_large, MOBILENET, [224, 32, 448]),
    "efficientnet": (M.efficientnet_b0, EFFICIENTNET, [224, 32, 448]),
    "vit":          (M.vit_b_16, VIT, [224, 32, 448]),
    "densenet":     (M.densenet121, DENSENET, [224, 32, 448]),
    "convnext":     (M.convnext_tiny, CONVNEXT, [224, 32, 448]),
}

if __name__ == "__main__":
    out = {}
    for name, (build, stages, sizes) in SPECS.items():
        out[name] = {
            "sizes": sizes,
            "rows": [[lab, txt] for lab, txt, _ in stages],
            "trace": trace(build, [s for s in stages if s[2]], sizes),
        }
        got = out[name]["trace"]
        print(f"  {name:13s} " + "  ".join(
            f"{s}:{'터짐 ' + got[str(s)]['error'] if got[str(s)]['error'] else '통과'}"
            for s in sizes))
    with open("backbone_shapes.json", "w") as f:
        json.dump(out, f, indent=1)
    print("  -> backbone_shapes.json")
