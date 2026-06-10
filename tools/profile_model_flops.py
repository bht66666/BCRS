from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import thop
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile a YOLO model with the Ultralytics FLOPs convention.")
    parser.add_argument("config", help="Model YAML path.")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model = YOLO(args.config).model.to(args.device).eval()
    x = torch.zeros(1, 3, args.imgsz, args.imgsz, device=args.device)
    saved_outputs = []
    total_flops = 0.0

    with torch.no_grad():
        for module in model.model:
            if module.f == -1:
                layer_input = x
            elif isinstance(module.f, int):
                layer_input = saved_outputs[module.f]
            else:
                layer_input = [x if index == -1 else saved_outputs[index] for index in module.f]

            profile_input = layer_input.copy() if isinstance(layer_input, list) else layer_input
            macs, _ = thop.profile(deepcopy(module), inputs=[profile_input], verbose=False)
            total_flops += macs * 2
            output = module(layer_input)
            saved_outputs.append(output if module.i in model.save else None)
            x = output

    params = sum(parameter.numel() for parameter in model.parameters())
    print(f"Params: {params / 1e6:.6f} M")
    print(f"FLOPs: {total_flops / 1e9:.3f} G @ {args.imgsz}")


if __name__ == "__main__":
    main()
