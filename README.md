# YOLOv12-LGCB

This repository contains the LGCB module implementation for dense-object feature extraction.

## Architecture

LGCB contains four branches:

  - Local: depthwise 3 x 3 convolution for local features.
  - Global: dilated depthwise convolution for wider receptive-field features.
  - Contrast: average pooling followed by element-wise subtraction for foreground-background differences.
  - Boundary: fixed Sobel and Laplace operators for boundary features.

## Environment

Install PyTorch for the required CUDA version first, then install the project:

```bash
pip install -e .
pip install -r requirements.txt
```

## Compatibility

Legacy class names remain available internally for compatibility.

## License

This project is based on Ultralytics YOLO and retains the AGPL-3.0 license.
