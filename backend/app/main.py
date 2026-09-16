from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
from backend.models.depth_engine import DepthEngine

app = FastAPI(title="DepthWizard API")

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Depth Engine with fine-tuned weights if available
MODEL_PATH = "depth_wizard_model.pt" if os.path.exists("depth_wizard_model.pt") else None
engine = DepthEngine(model_path=MODEL_PATH)

UPLOAD_DIR = "backend/data/raw/uploads"
RESULT_DIR = "backend/data/processed/results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

@app.get("/")
async def root():
    return {"message": "DepthWizard API is running. Use /process to generate depth maps."}

@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    """
    Accepts an image, generates a relative depth map, and saves it.
    """
    try:
        # 1. Save uploaded file
        job_id = str(uuid.uuid4())
        file_ext = os.path.splitext(file.filename)[1]
        input_path = os.path.join(UPLOAD_DIR, f"{job_id}{file_ext}")

        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Generate Depth Map
        # The engine returns a normalized numpy array (0-1)
        depth_map = engine.predict(input_path)

        # 3. Save result as a grayscale PNG (16-bit for precision)
        # We'll save as a simple image for now, then implement 16-bit PNG
        import cv2
        import numpy as np

        # Convert 0-1 float to 0-255 uint8 for simple preview
        depth_uint8 = (depth_map * 255).astype(np.uint8)
        result_path = os.path.join(RESULT_DIR, f"{job_id}_depth.png")
        cv2.imwrite(result_path, depth_uint8)

        # Save high-precision 16-bit PNG for GIS export
        depth_uint16 = (depth_map * 65535).astype(np.uint16)
        path_16bit = os.path.join(RESULT_DIR, f"{job_id}_16bit.png")
        cv2.imwrite(path_16bit, depth_uint16)

        # Generate 3D Wavefront OBJ mesh
        obj_path = os.path.join(RESULT_DIR, f"{job_id}.obj")
        export_obj_mesh(depth_map, obj_path)

        return {
            "job_id": job_id,
            "status": "completed",
            "result_url": f"/results/{job_id}_depth.png",
            "original_url": f"/uploads/{job_id}{file_ext}",
            "export_png16_url": f"/export/png16/{job_id}",
            "export_obj_url": f"/export/obj/{job_id}",
            "units": "relative",
            "crs": None,
            "calibration": {
                "method": "none",
                "scale": None,
                "offset": None,
                "finetuned": engine.using_finetuned,
                "note": "Relative depth (0-1), not metres. Calibration requires "
                        "a georeferenced input or GCPs.",
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def export_obj_mesh(depth_map, output_path, scale=3.0, step=4, max_grid=512):
    """
    Converts 2D depth map array into Wavefront 3D OBJ file format for 3D modeling tools (Blender, QGIS, Unity).
    """
    h, w = depth_map.shape
    # Bound the exported grid so very large images don't produce huge OBJ files.
    step = max(step, -(-max(h, w) // max_grid))
    h_sub = h // step
    w_sub = w // step

    vertices = []
    uvs = []
    faces = []

    for i in range(h_sub):
        y = (i / max(1, h_sub - 1)) * 10.0 - 5.0
        v = 1.0 - (i / max(1, h_sub - 1))
        for j in range(w_sub):
            x = (j / max(1, w_sub - 1)) * 10.0 - 5.0
            u = j / max(1, w_sub - 1)

            orig_y = min(i * step, h - 1)
            orig_x = min(j * step, w - 1)
            z = float(depth_map[orig_y, orig_x]) * scale

            vertices.append(f"v {x:.4f} {z:.4f} {-y:.4f}")
            uvs.append(f"vt {u:.4f} {v:.4f}")

    for i in range(h_sub - 1):
        for j in range(w_sub - 1):
            v1 = i * w_sub + j + 1
            v2 = i * w_sub + (j + 1) + 1
            v3 = (i + 1) * w_sub + j + 1
            v4 = (i + 1) * w_sub + (j + 1) + 1

            faces.append(f"f {v1}/{v1} {v3}/{v3} {v2}/{v2}")
            faces.append(f"f {v2}/{v2} {v3}/{v3} {v4}/{v4}")

    with open(output_path, "w") as f:
        f.write("# DepthWizard ISRO 3D OBJ Export\n")
        f.write("\n".join(vertices) + "\n")
        f.write("\n".join(uvs) + "\n")
        f.write("\n".join(faces) + "\n")

@app.get("/export/png16/{job_id}")
async def export_png16(job_id: str):
    path = os.path.join(RESULT_DIR, f"{job_id}_16bit.png")
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png", filename=f"depthwizard_{job_id}_16bit.png")
    raise HTTPException(status_code=404, detail="16-bit heightmap file not found")

@app.get("/export/obj/{job_id}")
async def export_obj(job_id: str):
    path = os.path.join(RESULT_DIR, f"{job_id}.obj")
    if os.path.exists(path):
        return FileResponse(path, media_type="application/object", filename=f"depthwizard_3d_mesh_{job_id}.obj")
    raise HTTPException(status_code=404, detail="3D OBJ mesh file not found")

@app.get("/results/{file_name}")
async def get_result(file_name: str):
    path = os.path.join(RESULT_DIR, file_name)
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/uploads/{file_name}")
async def get_upload(file_name: str):
    path = os.path.join(UPLOAD_DIR, file_name)
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="File not found")
