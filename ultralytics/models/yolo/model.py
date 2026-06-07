# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.model import Model
from ultralytics.models.yolo import detect
from ultralytics.nn.tasks import DetectionModel


class YOLO(Model):
    """Slim YOLO wrapper for YOLOv12-based detection only."""

    def __init__(self, model="yolov12n.pt", task=None, verbose=False):
        super().__init__(model=model, task=task or "detect", verbose=verbose)

    @property
    def task_map(self):
        """Map the detect task to the retained local implementations."""
        return {
            "detect": {
                "model": DetectionModel,
                "trainer": detect.DetectionTrainer,
                "validator": detect.DetectionValidator,
                "predictor": detect.DetectionPredictor,
            }
        }
