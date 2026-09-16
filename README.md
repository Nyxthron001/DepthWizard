# DepthWizard — Single-View Height Estimation & 3D Flythrough

**Smart India Hackathon 2026 — Problem Statement ID 26175 (ISRO / Disaster Management)**

DepthWizard is an AI-driven elevation reconstruction and visualization pipeline. It estimates high-resolution Digital Surface Models (DSMs) directly from a **single optical RGB image** (satellite or aerial/drone) and turns them into an interactive, textured 3D terrain for first-person flythrough and rapid disaster assessment.

---

## The Problem

High-resolution Digital Elevation Models (DEMs/DSMs) are essential for disaster response—including flood inundation modeling, landslide hazard zoning, and post-event structural damage assessment.

However, traditional photogrammetry and remote sensing techniques carry heavy operational tradeoffs:
- **Stereo Photogrammetry**: Requires overlapping multi-angle passes, demanding coordinated satellite tasking or multi-camera aerial rigs.
- **LiDAR**: Exceptional accuracy, but exorbitantly expensive and slow to mobilize over broad disaster zones.
- **InSAR (Interferometric Synthetic Aperture Radar)**: Complex baseline geometry, phase unwrapping artifacts, and latency in processing.

**DepthWizard explores single-view height estimation** as a rapid-deployment alternative: extracting reliable topographic and structural elevation from a single monocular optical frame.

---

## System Architecture

```
                    ┌───────────────────────────────────┐
                    │   Input Image (Satellite/Aerial)   │
                    └─────────────────┬─────────────────┘
                                      │
               ┌──────────────────────┴──────────────────────┐
               ▼                                             ▼
     [Mode A: Standard RGB]                       [Mode B: GeoTIFF]
     (PNG / JPEG format)                          (Embedded CRS & Geotransform)
               │                                             │
               ▼                                             ▼
    Tiling & Normalization                       Query SRTM 30m / Copernicus DEM
               │                                             │
               └──────────────────────┬──────────────────────┘
                                      ▼
             ┌─────────────────────────────────────────────────┐
             │         Depth Anything V2 (Fine-tuned)          │
             │           Monocular Depth Prediction            │
             └────────────────────────┬────────────────────────┘
                                      │ (Relative Depth Map)
               ┌──────────────────────┴──────────────────────┐
               ▼                                             ▼
     [Relative Elevation]                         [Metric Calibration]
   Normalized disparity [0, 1]                  Least-Squares Affine Alignment
                                                Z_metric = s * d_rel + t
                                                             │
                                                             ▼
                                                Absolute DSM (32-bit Float)
                                                             │
               ┌─────────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          FastAPI Backend                                │
│       - GeoTIFF / 16-Bit PNG Elevation Exporter                         │
│       - Wavefront 3D OBJ Mesh Exporter                                  │
│       - Heightmap & Texture Streaming (/process)                        │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               React Three Fiber (Three.js) 3D Viewer                   │
│  - Dynamic Vertex Displacement Heightmap                                │
│  - Orthorectified RGB Texture Draping                                   │
│  - Real-Time Telemetry HUD (X, Y, Elevation Readouts)                   │
│  - Dynamic Height Scale Slider & Reset Controls                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## End-to-End Phased Approach

The pipeline is structured into six core phases, taking raw optical inputs through machine learning inference to interactive web-based 3D visualization.

### Phase 1: Input Ingestion & Preprocessing
The system supports two distinct ingestion workflows depending on metadata availability:
- **Mode A: Non-Georeferenced Standard Imagery (PNG / JPG / TIFF)**
  - Suited for commercial drone surveys, reconnaissance photos, or ad-hoc aerial captures.
  - Image normalization, contrast enhancement, and automated tile-based tiling for large aerial scenes.
- **Mode B: Georeferenced Satellite Imagery (GeoTIFF)**
  - Extracts spatial bounding boxes, Coordinate Reference System (CRS, e.g., EPSG:4326, UTM projections), and Ground Sampling Distance (GSD) using `rasterio` and `pyproj`.
  - Automatically queries and caches overlapping coarse reference DEMs (e.g., SRTM 30m or Copernicus DEM GLO-30) for metric scaling.

### Phase 2: Monocular Depth Inference (Relative Elevation)
- Employs **Depth Anything V2** (Vision Transformer backbone: ViT-B / ViT-L) fine-tuned on the **GAMUS** remote sensing dataset.
- Standard depth models suffer from domain shift when applied to satellite imagery (they assume horizontal/egocentric viewpoints). Fine-tuning adapts the attention maps to:
  - Nadir and high-oblique satellite view geometry.
  - Sharp building facades, structural footprints, and roof boundaries.
  - Variable terrain features (cliffs, ravines, rolling hills, and riverbanks).
- Produces a dense, continuous relative disparity/depth map ($d_{rel}$).

### Phase 3: Metric Calibration & Absolute Elevation Modeling
- **Non-georeferenced images**: The relative depth is scaled to an intuitive normalized elevation range $[0, 1]$ or estimated based on user-configured height bounds.
- **Georeferenced GeoTIFFs**:
  - The coarse reference DEM (e.g., SRTM 30m) is resampled to match the spatial extents of the input image.
  - Coarse elevation values ($Z_{ref}$) serve as anchor points.
  - A robust least-squares affine fitting ($Z_{metric} = s \cdot d_{rel} + t$) determines the optimal global scale factor $s$ and vertical datum shift $t$.
  - High-frequency micro-topography (buildings, trees, retaining walls) predicted by the neural backbone is merged onto the low-frequency regional terrain baseline from the reference DEM.

### Phase 4: Backend API & Service Layer
- Built with **FastAPI** to provide asynchronous, low-latency processing pipelines:
  - `/process`: Accepts uploaded imagery, executes the inference & calibration pipeline, and returns generated artifacts.
  - `/export/png16/{job_id}`: Exports high-precision **16-bit grayscale heightmaps** for GIS suites (QGIS, ArcGIS).
  - `/export/obj/{job_id}`: Generates standard **Wavefront 3D OBJ mesh** files for Blender, Unity, or Unreal Engine.

### Phase 5: 3D Terrain Generation & Texture Draping
- Built on **Three.js** and **React Three Fiber**:
  - **Heightfield Displacement**: A subdivided plane geometry dynamically displaces vertices along the Z-axis in a custom vertex shader, directly driven by the processed elevation map.
  - **Texture Draping**: The original optical RGB image is draped as an albedo map over the displaced terrain mesh with matched UV coordinates.
  - **Dynamic Hillshading & Relief**: Calculates surface normals on the fly to simulate directional sun lighting, accentuating slopes, crests, and topological variations.

### Phase 6: First-Person 3D Flythrough & Disaster Analysis Tools
- **Navigable 3D Orbit & Flythrough Camera**:
  - Built with `@react-three/drei` controls allowing first-person flight and 360-degree inspection of target sites.
- **Disaster Assessment Tooling**:
  - **Real-Time Telemetry HUD**: Displays precise X, Y, and Z elevation data on cursor hover.
  - **Dynamic Elevation Scaling**: Interactive slider (0.5x–10x) for accentuating terrain topography in real time.

---

## Tech Stack

| Layer | Technologies |
|---|---|
| **Deep Learning & Inference** | PyTorch, Depth Anything V2 (ViT), Hugging Face `transformers`, `timm` |
| **Geospatial & Image Processing** | Rasterio, GDAL / PyOGRIO, Shapely, GeoPandas, PyProj, OpenCV, NumPy |
| **Backend Service** | FastAPI, Uvicorn, Pydantic |
| **Frontend & 3D Rendering** | React 19, Vite, Three.js, React Three Fiber (`@react-three/fiber`), `@react-three/drei` |

---

## Project Structure

```
DepthWizard/
├── backend/
│   ├── app/                 # FastAPI routes, schemas, and prediction pipelines
│   ├── data/                # Sample imagery and DEM caching
│   ├── models/              # Fine-tuned model checkpoints and weights
│   ├── scripts/             # Data download & preprocessing utilities
│   ├── tests/               # Unit and integration test suite
│   └── requirements.txt     # Python backend dependencies
├── frontend/
│   ├── src/
│   │   ├── components/      # 3D Canvas, Orbit Controls, and Telemetry HUD
│   │   ├── App.jsx          # Main application interface
│   │   └── index.css        # Styling and responsive design
│   ├── package.json         # React & Three.js dependencies
│   └── vite.config.js       # Vite bundler configuration
├── notebooks/               # Colab fine-tuning training notebook (v13)
├── start.bat                # 1-Click Windows server launcher
└── README.md
```

---

## Getting Started

### Prerequisites
- Node.js (v18+) and npm
- Python 3.10+ (CUDA-compatible GPU recommended for deep learning inference)

### 🚀 1-Click Quick Start (Windows)
Double click `start.bat` in the project root to automatically launch both the FastAPI backend and Vite frontend!

---

### 🛠️ Manual Setup

#### 1. Backend Setup

```bash
# Navigate to project root
cd DepthWizard

# Install Python dependencies
pip install -r requirements.txt

# Start backend server
python -m uvicorn backend.app.main:app --reload --port 8000
```

#### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install Node dependencies
npm install

# Start development server
npm run dev
```

Open `http://localhost:5173` in your browser to view the application.

---

## Roadmap

- [x] Problem statement scoping and architecture definition
- [x] Initial project scaffolding and clean environment setup
- [x] Interactive frontend landing page with drag-and-drop ingestion
- [x] Dataset preparation and fine-tuning Depth Anything V2 on GAMUS notebook
- [ ] Metric calibration module integration (`calibration.py`)
- [ ] FastAPI endpoint `/process` linking ML pipeline with frontend
- [ ] Three.js heightmap mesh rendering and texture draping
- [ ] 16-bit GIS Heightmap & Wavefront 3D OBJ mesh exporters
- [ ] Real-time Telemetry HUD & dynamic height scaling slider
- [ ] First-person drone camera flythrough controls
