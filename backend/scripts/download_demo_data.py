import os
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import h5py

def download_demo_samples(num_samples=5):
    print(f"Downloading {num_samples} demo samples from GAMUS (HDF5)...")

    save_dir = "backend/data/raw/demo"
    os.makedirs(save_dir, exist_ok=True)

    # List of some sample files from the GAMUS repository
    # These are examples; in a real scenario, we would list the files first
    sample_files = [
        "classes/train/DC_01_25_CLS.h5",
        "classes/train/DC_01_26_CLS.h5",
        "classes/train/DC_01_27_CLS.h5",
        "classes/train/DC_01_28_CLS.h5",
        "classes/train/DC_01_29_CLS.h5"
    ]

    for i in range(min(num_samples, len(sample_files))):
        file_path = sample_files[i]
        try:
            print(f"Downloading {file_path}...")
            # Download the specific .h5 file
            local_h5_path = hf_hub_download(
                repo_id="earthflow/GAMUS",
                filename=file_path,
                repo_type="dataset"
            )

            with h5py.File(local_h5_path, 'r') as f:
                # Extract image and depth
                # Note: Keys may vary, using common ones'
                image = np.array(f['image'][:])
                depth = np.array(f['depth'][:])

                # Normalize image to 0-255 uint8
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)

                # Save RGB
                Image.fromarray(image).save(os.path.join(save_dir, f"sample_{i}_rgb.png"))

                # Save Depth as 16-bit PNG (normalized)
                depth_norm = ((depth - depth.min()) / (depth.max() - depth.min()) * 65535).astype(np.uint16)
                # Using cv2 for 16-bit PNG support
                import cv2
                cv2.imwrite(os.path.join(save_dir, f"sample_{i}_depth.png"), depth_norm)

                print(f"Successfully extracted sample {i}")

        except Exception as e:
            print(f"Failed to download/extract {file_path}: {e}")

    print(f"Demo samples saved to {save_dir}")

if __name__ == "__main__":
    download_demo_samples()
