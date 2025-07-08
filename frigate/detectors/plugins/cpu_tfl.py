import logging

import numpy as np
from pydantic import Field
from typing_extensions import Literal

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig

try:
    from tflite_runtime.interpreter import Interpreter
except ModuleNotFoundError:
    from tensorflow.lite.python.interpreter import Interpreter


logger = logging.getLogger(__name__)

DETECTOR_KEY = "cpu"


class CpuDetectorConfig(BaseDetectorConfig):
    type: Literal[DETECTOR_KEY]
    num_threads: int = Field(default=3, title="Number of detection threads")


class CpuTfl(DetectionApi):
    type_key = DETECTOR_KEY

    def __init__(self, detector_config: CpuDetectorConfig):
        self.interpreter = Interpreter(
            model_path=detector_config.model.path,
            num_threads=detector_config.num_threads or 3,
        )

        self.interpreter.allocate_tensors()

        self.tensor_input_details = self.interpreter.get_input_details()
        self.tensor_output_details = self.interpreter.get_output_details()

    def detect_raw(self, tensor_input):
        self.interpreter.set_tensor(self.tensor_input_details[0]["index"], tensor_input)
        self.interpreter.invoke()

        raw_scores = self.interpreter.tensor(self.tensor_output_details[0]["index"])()[
            0
        ]
        raw_boxes = self.interpreter.tensor(self.tensor_output_details[1]["index"])()[0]
        # raw_count = int(
        #     self.interpreter.tensor(self.tensor_output_details[2]["index"])()[0]
        # )
        raw_classes = self.interpreter.tensor(self.tensor_output_details[3]["index"])()[
            0
        ]

        # 3 Get quantization params for each tensor
        scale_s, zp_s = self.tensor_output_details[0]["quantization"]
        scale_b, zp_b = self.tensor_output_details[1]["quantization"]
        scale_c, zp_c = self.tensor_output_details[2]["quantization"]
        scale_cl, zp_cl = self.tensor_output_details[3]["quantization"]

        # 4 Dequantize
        scores = (raw_scores.astype(np.float32) - zp_s) * scale_s
        boxes = (raw_boxes.astype(np.float32) - zp_b) * scale_b
        # count = int(round((raw_count.astype(np.float32)[0] - zp_c) * scale_c))
        classes = np.round((raw_classes.astype(np.float32) - zp_cl) * scale_cl).astype(
            int
        )

        # detections = np.zeros((20, 6), np.float32)
        detections = np.zeros((20, 6), dtype=np.float32)
        for i in range(5):
            if scores[i] < 0.4:
                break
            detections[i] = [
                classes[i],
                float(scores[i]),
                boxes[i][0],
                boxes[i][1],
                boxes[i][2],
                boxes[i][3],
            ]
        return detections
