import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForDepthEstimation, AutoImageProcessor
from .calibration import DepthCalibrator


class DepthEngine:
    """
    Runs monocular depth inference (Depth Anything V2) and converts the relative
    output to a unitless 0-1 map for the viewer.

    A fine-tuned checkpoint may be supplied via ``model_path``. It is only used
    if it loads cleanly against the same architecture; any failure (corrupt file,
    key mismatch, shape mismatch) falls back to the pretrained model with a
    warning rather than crashing the server.
    """

    def __init__(
        self,
        model_path=None,
        model_id="depth-anything/Depth-Anything-V2-Small-hf",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
        self.calibrator = DepthCalibrator()
        self.using_finetuned = False

        self._try_load_finetuned(model_path)

        self.model.to(self.device)
        self.model.eval()

    def _try_load_finetuned(self, model_path):
        """Load custom weights if present and valid; otherwise keep the base model."""
        if not model_path or not os.path.exists(model_path):
            return
        try:
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.using_finetuned = True
            print(f"[DepthEngine] Using fine-tuned weights from {model_path}")
        except Exception as e:
            self.using_finetuned = False
            print(
                f"[DepthEngine] WARNING: could not load {model_path} "
                f"({e.__class__.__name__}: {e}). Falling back to pretrained {self.model_id}."
            )

    @torch.inference_mode()
    def predict(self, image_path):
        """
        Predict a relative depth map from an image.

        Returns a float32 numpy array in [0, 1] (unitless relative depth).
        """
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        outputs = self.model(**inputs)
        if hasattr(outputs, "predicted_depth"):
            depth = outputs.predicted_depth.squeeze()
        elif hasattr(outputs, "predicted_depth_maps"):
            depth = outputs.predicted_depth_maps[0].squeeze()
        else:
            depth = outputs[0].squeeze()

        depth = depth.cpu().numpy().astype(np.float32)

        # Processor may resize non-square inputs; resize back to the original image.
        depth = np.array(
            Image.fromarray(depth).resize(image.size, Image.BILINEAR),
            dtype=np.float32,
        )

        lo, hi = depth.min(), depth.max()
        if hi - lo < 1e-8:
            return np.full(image.size[::-1], 0.5, dtype=np.float32)
        return (depth - lo) / (hi - lo)

    def calibrate_depth(self, depth_map, calibration_params_path=None):
        """
        Converts relative depth map to absolute metric height.
        """
        if calibration_params_path:
            self.calibrator.load_params(calibration_params_path)
        return self.calibrator.calibrate(depth_map)