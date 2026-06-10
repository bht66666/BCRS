# YOLOv12-LGCB Wheat Detection

This repository provides the training code for the YOLOv12-LGCB model used for dense wheat detection and counting.

LGCB denotes the Local-Global Context Block. The model uses four detection outputs, named P1-P4 in this project.

## Model

- Model configuration: `ultralytics/cfg/models/v12/yolov12n-lgcb.yaml`
- Training entry: `train_lgcb_100.py`
- Attention block: `LGCBAttn`
- Backbone wrapper: `A2C2fLGCB`
- Detection head: `LGCBDetect`
- Detection outputs: P1, P2, P3, and P4
- Actual output strides: 4, 8, 16, and 32
- Input resolution used for profiling: `1024 x 1024`
- Parameters: `2.363 M`
- FLOPs: `16.402 G`

P1-P4 are project-specific display names. They correspond to the feature maps conventionally named P2-P5 in standard FPN terminology. The naming does not change feature-map resolution or model computation.

## Architecture

The model includes:

- Four-scale detection, including the highest-resolution stride-4 output.
- Two LGCB backbone stages.
- Four LGCB branches with naming aligned to the manuscript:
  - Local: depthwise 3 x 3 convolution for local features.
  - Global: dilated depthwise convolution for wider receptive-field features.
  - Contrast: average pooling followed by element-wise subtraction for foreground-background differences.
  - Boundary: fixed Sobel and Laplace operators for boundary features.
- DFL-based bounding-box regression.
- Separate box regression and classification branches.
- Depthwise-separable convolutions in the box regression branch.

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
python train_lgcb_100.py
```

Default settings:

- Epochs: `100`
- Image size: `1024`
- Batch size: `4`
- Device: `0`
- Output directory: `runs/detect/lgcb_p1_p4_100`

## Profile

Reproduce the parameter and FLOPs measurement:

```bash
python tools/profile_model_flops.py ultralytics/cfg/models/v12/yolov12n-lgcb.yaml --imgsz 1024
```

## Compatibility

Legacy class names remain available internally so historical configurations and checkpoints are not broken. The current model configuration and training entry use the LGCB naming.

## License

This project is based on Ultralytics YOLO and retains the AGPL-3.0 license.
