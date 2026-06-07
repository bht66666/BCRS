from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from bcrs_yolov12_p2 import build_base_model, build_model


def run_one(name, model):
    model.eval()
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        y = model(x)
    print("model:", name)
    print("params:", sum(p.numel() for p in model.parameters()))
    print("output shape:", tuple(y.shape))
    print("-" * 40)


def main():
    run_one("YOLOv12Baseline", build_base_model(nc=1, scale="n"))
    run_one("BCRSYOLOv12P2", build_model(nc=1, scale="n"))


if __name__ == "__main__":
    main()
