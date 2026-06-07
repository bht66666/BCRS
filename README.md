# YOLOv12-BCRS-P2 Wheat Detection

This repository contains the minimal training project used for the YOLOv12-BCRS-P2 wheat detection/counting experiments.

It keeps the Ultralytics training path intact and adds the BCRS attention modules and YOLOv12-P2 model configuration used by our training script.

## Included

- `train_bcrs_p2_100.py`: the 100-epoch training entry used for YOLOv12-BCRS-P2.
- `ultralytics/`: the source tree required by the local training framework.
- `ultralytics/cfg/models/v12/yolov12n-ablate-bcrs-p2.yaml`: YOLOv12 with BCRS attention in the backbone area-attention blocks and a P2-P5 detection head.
- `data/wheat.yaml`: dataset configuration template for one-class wheat detection.
- `standalone_architecture/`: a small independent PyTorch architecture package for quick forward-pass inspection.

## Not Included

- Dataset images and labels.
- Training outputs under `runs/`.
- Model weights such as `.pt`, `.pth`, `.onnx`, or `.engine`.
- Paper drafts, spreadsheets, visual exports, and other experiment artifacts.

## Environment

Install PyTorch for your CUDA version first, then install this project in editable mode:

```bash
pip install -e .
pip install -r requirements.txt
```

The original training environment used a CUDA GPU. CPU execution is possible for code checks but is not practical for full 1024-resolution training.

## Dataset Layout

Prepare a YOLO-format dataset and update `data/wheat.yaml` if your path is different:

```text
data/wheat/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

The label files should use normal YOLO detection format:

```text
class x_center y_center width height
```

For the wheat experiments, `nc=1` and class `0` is `wheat`.

## Train

Place the YOLOv12n pretrained weight at the repository root as `yolov12n.pt`, or edit `train_bcrs_p2_100.py` to point to your pretrained weight.

```bash
python train_bcrs_p2_100.py
```

Default training settings in the script:

- model: `ultralytics/cfg/models/v12/yolov12n-ablate-bcrs-p2.yaml`
- data: `data/wheat.yaml`
- epochs: `100`
- image size: `1024`
- batch size: `4`
- device: `0`
- output: `runs/detect/ablate_bcrs_p2_100`

## Quick Architecture Smoke Test

The standalone package can be checked without the Ultralytics parser:

```bash
cd standalone_architecture
pip install -r requirements.txt
python demo_build.py
```

This only verifies model construction and forward output shape. Use `train_bcrs_p2_100.py` for the actual training workflow.

## License

This project is based on Ultralytics YOLO and keeps the AGPL-3.0 license.
