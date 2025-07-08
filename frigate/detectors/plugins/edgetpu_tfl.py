import logging
import os

import numpy as np
from pydantic import Field
from typing_extensions import Literal

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ModuleNotFoundError:
    from tensorflow.lite.python.interpreter import Interpreter, load_delegate


logger = logging.getLogger(__name__)

DETECTOR_KEY = "edgetpu"


class EdgeTpuDetectorConfig(BaseDetectorConfig):
    type: Literal[DETECTOR_KEY]
    device: str = Field(default=None, title="Device Type")


class EdgeTpuTfl(DetectionApi):
    type_key = DETECTOR_KEY

    def __init__(self, detector_config: EdgeTpuDetectorConfig):
        device_config = {}
        if detector_config.device is not None:
            device_config = {"device": detector_config.device}

        edge_tpu_delegate = None

        try:
            device_type = (
                device_config["device"] if "device" in device_config else "auto"
            )
            logger.info(f"Attempting to load TPU as {device_type}")
            edge_tpu_delegate = load_delegate("libedgetpu.so.1.0", device_config)
            logger.info("TPU found")
            self.interpreter = Interpreter(
                model_path=detector_config.model.path,
                experimental_delegates=[edge_tpu_delegate],
            )
        except ValueError:
            _, ext = os.path.splitext(detector_config.model.path)

            if ext and ext != ".tflite":
                logger.error(
                    "Incorrect model used with EdgeTPU. Only .tflite models can be used with a Coral EdgeTPU."
                )
            else:
                logger.error(
                    "No EdgeTPU was detected. If you do not have a Coral device yet, you must configure CPU detectors."
                )

            raise

        self.interpreter.allocate_tensors()

        self.tensor_input_details = self.interpreter.get_input_details()
        self.tensor_output_details = self.interpreter.get_output_details()
        self.model_type = detector_config.model.model_type

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
