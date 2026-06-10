from pathlib import Path
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

from ultralytics import YOLO


def main() -> None:
    model = YOLO(str(root / "ultralytics" / "cfg" / "models" / "v12" / "yolov12n-lgcb.yaml"))
    model.train(
        data=str(root / "data" / "wheat.yaml"),
        epochs=100,
        imgsz=1024,
        batch=4,
        device="0",
        workers=0,
        pretrained=str(root / "yolov12n.pt"),
        project=str(root / "runs" / "detect"),
        name="lgcb_p1_p4_100",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
