# YOLOv12 BCRS P2 Standalone

This is a clean, direct PyTorch architecture package for:

- `Original YOLOv12`
- `YOLOv12 + BCRS attention + P2 head`

It is intended for architecture inspection, paper/code packaging, and quick forward-pass validation. It does not depend on Ultralytics YAML parsing.

Included:

- explicit model definitions for both baseline and BCRS-P2
- minimal building blocks required by this model
- four-branch BCRS attention:
  - local branch
  - context branch
  - global branch
  - boundary branch
- P2-P5 four-scale detection head
- demo script for model construction and forward pass

Not included:

- the full Ultralytics training framework
- unrelated attention branches
- dataset, trainer, parser, or paper files

## Structure

- `bcrs_yolov12_p2/modules.py`: minimal modules, BCRS attention, and detection head
- `bcrs_yolov12_p2/model.py`: explicit baseline and BCRS-P2 architectures
- `demo_build.py`: simple smoke test for both models
- `requirements.txt`: minimal runtime dependencies

## Quick Start

```bash
pip install -r requirements.txt
python demo_build.py
```

## Example

```python
from bcrs_yolov12_p2 import build_base_model, build_model

base_model = build_base_model(nc=1, scale="n")
bcrs_model = build_model(nc=1, scale="n")
```
