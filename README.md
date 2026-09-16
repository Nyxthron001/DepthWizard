# DepthWizard

DepthWizard is an end-to-end pipeline designed for the Smart India Hackathon 2026 (Problem Statement ID 26175), sponsored by ISRO. 

## Project Goal
The project converts a single optical RGB remote-sensing image into a high-precision Digital Surface Model (DSM) and renders it as a navigable, textured 3D flythrough.

## Architecture
The system follows a modular architecture consisting of five core services:
1. **Ingest**: Detects image type (PNG/JPG vs GeoTIFF) and reads metadata.
2. **Relative Depth Engine**: Uses a foundation model (like Depth Anything V2) adapted for nadir remote-sensing imagery.
3. **Scale Calibration**: Regresses relative depth to absolute metric height using SRTM 30m DEMs or Ground Control Points (GCPs).
4. **DSM Export**: Outputs results as GeoTIFF/COG (for georeferenced data) or PNG16 heightmaps.
5. **3D Visualization Service**: Renders a textured 3D terrain mesh for an interactive first-person flythrough experience.

## Evaluation Focus
- **DSM Accuracy (50%)**: Measured via RMSE, MAE, and correlation across urban, sparse, hilly, and forested scenes.
- **Visualization & UX (50%)**: Focuses on projection accuracy, visual fidelity, navigability, and stability of the standalone deployment.
