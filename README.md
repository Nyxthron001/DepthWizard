# 🛰️ DepthWizard — 3D Elevation Terrain Pipeline

> **Smart India Hackathon 2026 — Problem Statement ID 26175**  
> **Sponsoring Organisation:** Indian Space Research Organisation (ISRO)  
> **Theme:** Disaster Management | **Category:** Software  

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19.2-61dafb.svg)](https://react.dev/)
[![Three.js](https://img.shields.io/badge/Three.js-0.185-black.svg)](https://threejs.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-ee4c2c.svg)](https://pytorch.org/)

---

## 📌 Overview

**DepthWizard** is an end-to-end AI & 3D rendering pipeline that converts a **single optical RGB remote-sensing satellite or aerial image** into a high-precision elevation map (Digital Surface Model / DSM), then renders it as an interactive, navigable 3D terrain flythrough with real-time telemetry.

Built for disaster management scenarios (floods, landslides, terrain inspection), DepthWizard bridges the domain gap between ground-level monocular depth models and nadir satellite imagery.

---

## ✨ Key Features

- **🚀 Deep Learning Depth Inference**: Powered by Depth Anything V2 fine-tuned on the GAMUS satellite dataset.
- **🧊 Interactive 3D Terrain Viewer**: Built with Three.js & React Three Fiber. Features dynamic displacement scaling (0.5x–10x), directional lighting, and background fog.
- **🛰️ Real-Time Telemetry HUD**: Displays real-time X, Y, and Elevation (Z) coordinates as you hover your cursor over the 3D terrain.
- **🔀 Dual Input Support**:
  - **Branch A (Non-Georeferenced)**: Accepts plain PNG/JPG images $\rightarrow$ generates a normalized 0–1 Relative DSM ($rDSM$).
  - **Branch B (Georeferenced)**: Accepts GeoTIFF files $\rightarrow$ regresses relative depth to absolute metric height (meters) using SRTM 30m DEM data.
- **💾 Multi-Format Exports**:
  - **16-Bit GIS Heightmap (`.png`)**: High-precision grayscale heightmap for QGIS, ArcGIS, and GDAL workflows.
  - **3D Mesh (`.obj`)**: Standard Wavefront 3D OBJ mesh file ready for Blender, Unity, or Unreal Engine.
- **🛡️ Robust Model Fallback**: Automatically loads fine-tuned weights (`depth_wizard_model.pt`) if present; falls back to Hugging Face base pretrained models gracefully if missing.

---

## 🏗️ System Architecture

```
                               ┌─────────────────────────┐
                               │   Input RGB Satellite   │
                               │   Image (PNG/JPG/GeoTIFF)│
                               └────────────┬────────────┘
                                            │
                                            ▼
                               ┌─────────────────────────┐
                               │   FastAPI Ingestion     │
                               └────────────┬────────────┘
                                            │
                                            ▼
                               ┌─────────────────────────┐
                               │   DepthEngine (PyTorch) │
                               │  Depth Anything V2 HF   │
                               └────────────┬────────────┘
                                            │
                                            ▼
                               ┌─────────────────────────┐
                               │    ScaleCalibrator      │
                               │ (Linear Regression/SRTM)│
                               └────────────┬────────────┘
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
       ┌───────────────────────────┐                 ┌───────────────────────────┐
       │   Export Generators       │                 │   Interactive 3D Viewer   │
       │  • 16-Bit PNG Heightmap   │                 │  • Three.js Mesh Render   │
       │  • Wavefront 3D OBJ Mesh  │                 │  • Real-Time Telemetry HUD│
       └───────────────────────────┘                 └───────────────────────────┘
```

---

## 💻 Tech Stack

- **Backend Framework**: Python 3.10+, FastAPI, Uvicorn
- **ML / Computer Vision**: PyTorch, Transformers, OpenCV, Scikit-Learn, PIL
- **Frontend Framework**: React 19, Vite
- **3D Graphics Engine**: Three.js, `@react-three/fiber`, `@react-three/drei`
- **Data & Model Hub**: Hugging Face Hub (GAMUS Dataset & Depth Anything V2)

---

## 📁 Repository Structure

```
DepthWizard/
├── backend/
│   ├── app/
│   │   └── main.py              # FastAPI server endpoints (/process, /export/obj, /export/png16)
│   ├── models/
│   │   ├── depth_engine.py      # Monocular depth model wrapper & inference logic
│   │   └── calibration.py       # Relative-to-metric scale recovery calibrator
│   └── scripts/
│       └── download_demo_data.py# Script to download sample GAMUS data
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── DepthWizardViewer.jsx # Main 3D Canvas, Orbit Controls & Telemetry HUD
│   │   │   └── DepthWizardViewer.css # Mission-control dark mode styling
│   │   └── App.jsx
│   └── package.json
├── notebooks/
│   └── train_depth_wizard.ipynb # Colab notebook (v13) for cloud fine-tuning on GAMUS
├── data/
│   └── samples/                 # Sample satellite test images
├── start.bat                    # 1-Click Windows launcher script
└── requirements.txt             # Python backend dependencies
```

---

## ⚡ Quick Start Guide

### Option A: 1-Click Launch (Windows)

Simply double-click **`start.bat`** in the project root directory. It will automatically launch both the FastAPI backend server and the Vite React frontend in separate terminal windows!

---

### Option B: Manual Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/Nyxthron001/DepthWizard.git
cd DepthWizard
```

#### 2. Setup & Start Backend
```bash
# Install Python dependencies
pip install -r requirements.txt

# Start FastAPI server on port 8000
python -m uvicorn backend.app.main:app --reload --port 8000
```

#### 3. Setup & Start Frontend
```bash
# Navigate to frontend folder
cd frontend

# Install Node modules
npm install

# Start Vite dev server on port 5173
npm run dev
```

#### 4. Open in Browser
Visit **`http://localhost:5173`** in your browser. Upload any aerial or satellite image (or select one from `data/samples/`) and click **"Generate 3D Terrain"**!

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API status check |
| `POST` | `/process` | Upload RGB image $\rightarrow$ returns depth map URL, 3D mesh URL, and telemetry |
| `GET` | `/export/png16/{job_id}` | Download 16-bit GIS elevation heightmap PNG |
| `GET` | `/export/obj/{job_id}` | Download Wavefront 3D OBJ mesh file |
| `GET` | `/results/{filename}` | Serve processed preview files |
| `GET` | `/uploads/{filename}` | Serve raw uploaded source images |

---

## 🧠 Model Weights & Cloud Training

- **Default Mode**: The system automatically uses Hugging Face's `depth-anything/Depth-Anything-V2-Small-hf` base model if no custom weights are found.
- **Custom Fine-Tuning**: Run `notebooks/train_depth_wizard.ipynb` on Google Colab / Kaggle GPU to fine-tune on the **earthflow/GAMUS** dataset. Save the output file as `depth_wizard_model.pt` in the project root to enable custom weights automatically.

---

## 🏆 SIH 2026 Evaluation Criteria Alignment

- **DSM Estimation Accuracy (50%)**: Domain adaptation via GAMUS dataset training and metric elevation scale recovery via linear regression against SRTM DEMs.
- **Visualization & UX (50%)**: Seamless in-browser 3D flythrough, real-time coordinate inspection HUD, dynamic height scale slider, and multi-format export capabilities.

---

## 📜 License & Acknowledgments

- Problem Statement sponsored by **Indian Space Research Organisation (ISRO)** for **Smart India Hackathon 2026**.
- Pretrained foundation model provided by **Depth Anything V2**.
- Remote sensing paired dataset provided by **GAMUS (Earthflow)**.
