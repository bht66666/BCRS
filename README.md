# YOLOv12-LGCB-Lite Wheat Detection

This repository provides the training code for the lightweight YOLOv12-LGCB model used for dense wheat detection and counting.

LGCB denotes the Local-Global Context Block. The model retains four detection outputs, named P1-P4 in this project, and uses a lightweight box regression head.

## Model

- Model configuration: `ultralytics/cfg/models/v12/yolov12n-lgcb-lite.yaml`
- Training entry: `train_lgcb_lite_100.py`
- Attention block: `LGCBAttn`
- Backbone wrapper: `A2C2fLGCB`
- Lightweight detection head: `LGCBLiteDetect`
- Detection outputs: P1, P2, P3, and P4
- Actual output strides: 4, 8, 16, and 32
- Input resolution used for profiling: `1024 x 1024`
- Parameters: `2.363 M`
- FLOPs: `16.402 G`

P1-P4 are project-specific display names. They correspond to the feature maps conventionally named P2-P5 in standard FPN terminology. The renaming does not change feature-map resolution or model computation.

## Lightweight Design

The model retains:

- Four-scale detection, including the highest-resolution stride-4 output.
- Both LGCB backbone stages.
- The original DFL representation and prediction format.
- The original classification branch.

The box regression branch replaces standard `3 x 3` convolutions with depthwise-separable convolutions. Under the same `1024 x 1024` profiling setting, this reduces complexity from `25.886 G` to `16.402 G`, a reduction of approximately `36.6%`.

## Environment

Install PyTorch for the required CUDA version first, then install the project:

```bash
pip install -e .
pip install -r requirements.txt
```

## Dataset

Prepare a one-class YOLO-format wheat dataset:

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

Update `data/wheat.yaml` if the dataset is stored elsewhere.

## Train

Place the YOLOv12n pretrained weights at the repository root as `yolov12n.pt`, or update the weight path in the training script.

```bash
python train_lgcb_lite_100.py
```

Default settings:

- Epochs: `100`
- Image size: `1024`
- Batch size: `4`
- Device: `0`
- Output directory: `runs/detect/lgcb_lite_p1_p4_100`

## Profile

Reproduce the parameter and FLOPs measurement:

```bash
python tools/profile_model_flops.py ultralytics/cfg/models/v12/yolov12n-lgcb-lite.yaml --imgsz 1024
```

## Compatibility

Legacy model names remain available internally so that historical configurations and checkpoints are not broken. The current lightweight release uses only the LGCB naming in its model configuration and training entry.

## License

This project is based on Ultralytics YOLO and retains the AGPL-3.0 license.
