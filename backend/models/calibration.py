import numpy as np
from sklearn.linear_model import LinearRegression

class DepthCalibrator:
    """
    Handles the conversion of relative depth maps to absolute metric height (DSM).
    Formula: Absolute_Height = a * Relative_Depth + b
    """
    def __init__(self, a=1.0, b=0.0):
        self.a = a
        self.b = b

    def fit(self, relative_depth, absolute_depth):
        """
        Estimates calibration coefficients using linear regression.

        Args:
            relative_depth (np.ndarray): The output from Depth-Anything (flattened).
            absolute_depth (np.ndarray): Reference heights from SRTM/GCP (flattened).
        """
        # Flatten arrays for regression
        X = relative_depth.reshape(-1, 1)
        y = absolute_depth.reshape(-1)

        model = LinearRegression().fit(X, y)
        self.a = model.coef_[0]
        self.b = model.intercept_

        print(f"Calibration complete. Scale (a): {self.a:.4f}, Offset (b): {self.b:.4f}")
        return self.a, self.b

    def calibrate(self, relative_depth):
        """
        Transforms a relative depth map to absolute metric height.
        """
        return self.a * relative_depth + self.b

    def save_params(self, path):
        np.savez(path, a=self.a, b=self.b)

    def load_params(self, path):
        data = np.load(path)
        self.a = data['a']
        self.b = data['b']
