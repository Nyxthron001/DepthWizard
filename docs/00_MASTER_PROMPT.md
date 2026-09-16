# DepthWizard — Master Solution Prompt

> **What this file is.** A single, self-contained brief you can paste into any capable coding
> agent (Claude Code, Cursor, Copilot Workspace, Codex) to make it build DepthWizard correctly.
> It is written as instructions *to an engineer*, not as a description of a problem. Paste it
> whole. Do not trim the constraints section — that is where most agents go wrong.
>
> SIH 2026 Problem Statement **26175**, Indian Space Research Organisation (ISRO / SAC),
> theme *Disaster Management*, category *Software*.

---

## 0. Role and standing orders

You are the lead engineer on **DepthWizard**: a pipeline that turns a *single* optical
RGB remote-sensing image into a Digital Surface Model, then lets a user fly through that
surface in 3D in a browser.

Standing orders, in priority order:

1. **Never invent numbers.** If you output an elevation in metres, you must be able to
   name the exact line of code that put it in metres. Relative depth silently relabelled
   as "height in metres" is the single most common failure of this problem statement and
   it will be caught immediately by an RMSE check.
2. **Two input classes, two contracts.** Non-georeferenced input (PNG/JPG) → *relative*
   DSM (rDSM), unitless, documented as unitless. Georeferenced input (GeoTIFF with CRS
   and geotransform) → *absolute* DSM in metres, with CRS preserved. Never blur these.
3. **Every raster you emit is a valid geospatial file.** Float32 GeoTIFF, explicit nodata,
   CRS and transform copied from the source when the source has them.
4. **Ship a running demo at every checkpoint.** A broken half-feature is worth less than a
   working simpler one. Evaluation is 50% accuracy, 50% visualisation and UX — a beautiful
   flythrough with honest numbers beats a clever model with no viewer.
5. **State uncertainty.** Where the pipeline is guessing (e.g. scale recovered from a 30 m
   DEM over a 0.5 m image), the UI and the docs must say so.

---

## 1. The physics you must get right before writing code

This section exists because the naïve reading of the problem statement produces a wrong
system. Read it twice.

### 1.1 Depth is not height

A monocular depth model trained on natural imagery predicts **depth**: distance from the
camera to the surface, or more usually *inverse* depth (disparity), which is large when
things are close. In a nadir (straight-down) satellite image the camera is above the scene,
so a tall building is **closer** to the camera than the street beside it.

Therefore:

```
disparity  (what Depth Anything / MiDaS actually output, "relative depth")
  ↑ large  = near camera = high on the ground
  ↓ small  = far  from camera = low on the ground

height  ∝  +disparity        (nadir imagery)
height  ∝  -metric_depth     (if the model outputs true metric depth in metres)
```

Get the sign wrong and your terrain is inverted — buildings become pits. Assert on this:
after calibration, the median height over detected road/water pixels must be lower than the
95th percentile over building pixels. Fail loudly if not.

### 1.2 DSM vs DTM vs nDSM — pick one and label it

| Product | Meaning | Typical use |
|---|---|---|
| **DSM** | Absolute elevation of the *top* surface: bare earth **plus** buildings, trees. Metres above a vertical datum (e.g. EGM96 geoid or WGS84 ellipsoid). | What ISRO 26175 asks for. |
| **DTM / DEM** | Bare-earth terrain only, buildings and vegetation removed. | SRTM, Copernicus GLO-30 are effectively DSM-ish at 30 m but are commonly used as terrain. |
| **nDSM (AGL)** | `DSM − DTM`. Height *above ground level*. Ground = 0 m. | What most academic "monocular height estimation" papers predict, including DFC2019 Track 1. |

**Consequence that will bite you:** most public ground truth (DFC2019 `*_AGL.tif`, GAMUS,
IMG2nDSM) is **nDSM**, not DSM. If you train or validate against nDSM but claim DSM, your
RMSE is meaningless. The clean architecture, which you should adopt:

```
predict nDSM (height above ground, ground = 0)
      +  terrain surface DTM  (from Copernicus GLO-30 / SRTM, upsampled)
      =  absolute DSM  (metres above datum)
```

This decomposition is also what makes scale calibration tractable, because the low-res DEM
supplies the *terrain* component and the depth model supplies the *object* component. They
are separable problems and you should keep them separate in code.

### 1.3 The scale problem, stated exactly

Affine-invariant depth models are trained with a loss that is invariant to scale and shift.
Their output `d̂` therefore relates to the truth only up to two unknowns:

```
d_true(p)  ≈  s · d̂(p)  +  t          for all pixels p,  with unknown s ∈ ℝ, t ∈ ℝ
```

This is the MiDaS / Depth-Anything evaluation convention. Recovering `(s, t)` is a
two-parameter least-squares fit, solvable in closed form from as few as two reliable
reference pixels — but you want hundreds for stability. Given a set of reference pixels
`P` with known truth `h(p)`, minimise `Σ_{p∈P} ( s·d̂(p) + t − h(p) )²`:

```
Let  A = [[ Σ d̂² ,  Σ d̂ ],
          [ Σ d̂  ,  |P| ]]              b = [ Σ d̂·h ,  Σ h ]ᵀ
Then [s, t]ᵀ = A⁻¹ b
```

Implement this in `backend/app/calibrate.py` as `fit_scale_shift()`, in float64, with:

* **Robustness.** Plain least squares is destroyed by outliers, and a coarse DEM over a
  fine image is *full* of outliers. Use RANSAC, or iteratively reweighted least squares with
  a Huber or Tukey loss, or simply iterate: fit → drop residuals beyond 2.5σ → refit, 3 times.
* **A degeneracy guard.** If `Σ(d̂ − mean(d̂))²` is near zero the image is flat and `s` is
  unidentifiable. Return a documented failure, not a silent `inf`.
* **A recorded quality metric.** Store `s`, `t`, inlier count, inlier RMSE, and R² in the
  output metadata. The UI shows these. This is your honesty mechanism.

### 1.4 Four ways to get reference pixels, in decreasing order of accuracy

| # | Source of truth | When available | Expected quality |
|---|---|---|---|
| 1 | **Ground Control Points** — user clicks N points and types known elevations | User has survey data or reads a map | Best. 3–10 GCPs beat everything below. |
| 2 | **Reference LiDAR / high-res DSM tile** for the same footprint | Validation only, not deployment | Gold standard; use for reporting RMSE. |
| 3 | **Coarse DEM (Copernicus GLO-30 / SRTM 30 m)** resampled to image grid | Any georeferenced input, worldwide, free | Good for the *terrain* trend, useless for buildings. See §1.5. |
| 4 | **Semantic priors** — roads/water are ground level; buildings quantised to ~3.2 m storeys | Always, no external data | Weak but free; excellent as a *regulariser*, not as the only source. |

Implement all four. Default chain for a georeferenced input: try 1 if the user supplied GCPs,
else 3 combined with 4. Always report which path was taken.

### 1.5 Why a 30 m DEM cannot calibrate building heights (and what to do)

A Copernicus GLO-30 pixel is 30 m × 30 m. A 0.5 m satellite image pixel is 3600× smaller in
area. One DEM pixel covers an entire building plus its surroundings, and the DEM value is
some blend of roof and ground. So:

* **Do use the coarse DEM** to fit the *low-frequency* terrain component — the hill, the
  valley, the general slope. It is genuinely good at that.
* **Do not** feed building-covered pixels into the scale fit. They are the outliers that
  will wreck `s`.

The practical recipe, implement exactly this:

```
1. Reproject coarse DEM onto the image grid (bilinear) → D_coarse  [metres]
2. Low-pass the predicted relative surface:  R_low = gaussian(R, σ ≈ 30m/GSD)
3. Build a ground mask M:
     - exclude pixels where |R − R_low| is large   (objects sticking up)
     - exclude vegetation (NDVI if you have NIR; else ExG excess-green index on RGB)
     - keep the lowest ~40% of R within each DEM cell   (roughly, bare ground)
4. Fit (s, t) on M only, robustly, against D_coarse
5. DTM_hi  = s·R_low + t                        (terrain, metres, absolute)
6. nDSM    = s·(R − R_low)                      (objects, metres above ground)
7. DSM     = DTM_hi + max(nDSM, 0)
```

Step 7's `max(·, 0)` encodes "objects do not go below ground". Keep it; it removes a whole
class of visible artefacts in the 3D view.

### 1.6 Domain gap: what actually goes wrong on satellite imagery

Foundation depth models are trained on egocentric photographs with strong perspective,
a horizon, and a ground plane receding to a vanishing point. A nadir satellite tile has
none of that. Symptoms you will observe and must handle:

* **Vignetting / global bowl artefacts** — the model hallucinates a soft depth gradient
  across the tile because it expects perspective. Fix: high-pass the prediction before
  extracting the object component, and let the coarse DEM own all low frequencies.
* **Shadow-as-depth** — dark shadows read as far away. Fix: shadow-aware masking, or
  simply do not trust pixels below a luminance threshold when fitting.
* **Scale confusion** — the model has no idea whether the tile is 200 m or 20 km across.
  Fix: this is exactly what `s` absorbs; but also feed the model a consistent input GSD by
  resampling every tile to a canonical GSD before inference (e.g. 0.5 m) so `s` is stable
  across tiles and you can cache a per-sensor prior for it.
* **Tiling seams** — running the model per 518×518 tile gives visible discontinuities. Fix:
  overlapping tiles (≥25% overlap) with cosine/Hann feathered blending, and per-tile
  scale-shift harmonisation against the global fit. Non-negotiable for large scenes.

---

## 2. Required architecture

```
                    ┌──────────────────────────────────────────────┐
  PNG / JPG ──────► │  ingest.py                                   │
  GeoTIFF   ──────► │  • sniff CRS + transform (rasterio)          │
                    │  • branch: GEO or PLAIN                      │
                    │  • resample to canonical GSD                 │
                    │  • tile with overlap                         │
                    └───────────────────┬──────────────────────────┘
                                        ▼
                    ┌──────────────────────────────────────────────┐
                    │  depth.py   — backbone inference             │
                    │  DA3 Mono / Depth-Anything-V2 → relative R   │
                    │  fp16, tiled, feathered blend                │
                    └───────────────────┬──────────────────────────┘
                                        ▼
                    ┌──────────────────────────────────────────────┐
        PLAIN ─────►│  rDSM: normalise R to [0,1], mark UNITLESS   │
                    └───────────────────┬──────────────────────────┘
                                        │
                    ┌───────────────────▼──────────────────────────┐
        GEO   ─────►│  calibrate.py                                │
                    │  dem_fetch (GLO-30) │ gcp │ semantic priors   │
                    │  robust fit (s,t) → DTM_hi, nDSM, DSM        │
                    │  emit calibration report                     │
                    └───────────────────┬──────────────────────────┘
                                        ▼
                    ┌──────────────────────────────────────────────┐
                    │  dsm_io.py — Float32 COG GeoTIFF + nodata    │
                    │  metrics.py — RMSE / MAE / Pearson r / δ     │
                    │  mesh.py    — decimated grid → glTF/binary   │
                    └───────────────────┬──────────────────────────┘
                                        ▼
                    ┌──────────────────────────────────────────────┐
                    │  FastAPI  /upload /estimate /dsm /mesh /eval │
                    └───────────────────┬──────────────────────────┘
                                        ▼
                    ┌──────────────────────────────────────────────┐
                    │  Three.js viewer — displaced grid + ortho    │
                    │  texture, FPS flycam, click-to-measure,      │
                    │  slope, profile, hypsometric tint, legend    │
                    └──────────────────────────────────────────────┘
```

Hard interface rules: the backend never returns a bare array without units; every response
carries `{"units": "metres" | "relative", "crs": <EPSG or null>, "calibration": {...}}`.

---

## 3. Model selection — decided, with reasons

Use **two** models behind one interface, selectable at runtime. Do not hard-code one.

**Primary — Depth Anything 3, monocular variant.** Released 14 Nov 2025 (arXiv 2511.10647,
ICLR 2026). `DA3MONO-LARGE` is a ViT-L/14 specialised for single-image depth, Apache-2.0,
and it is the current state of the art for monocular relative depth. Install via
`pip install depth-anything-3`. `depth-anything/DA3NESTED-GIANT-LARGE` adds metric-scale
capability but is far heavier — treat it as optional and gate it behind a VRAM check.

**Fallback — Depth Anything V2 Large via 🤗 transformers.** `depth-anything/Depth-Anything-V2-Large-hf`
(~335 M params). Works with the plain `pipeline("depth-estimation", ...)` API, runs in well
under 8 GB at fp16, and has metric-fine-tuned siblings
(`Depth-Anything-V2-Metric-Indoor-Large-hf`, `-Metric-Hypersim-Large`, Virtual-KITTI-2
outdoor variants). Ship this as the default on your 8–12 GB laptop GPU because it is the
one you can guarantee will load. `Depth-Anything-V2-Small-hf` is your CPU-only escape hatch.

VRAM guidance for an 8–12 GB laptop card, at fp16, tile 518×518:

| Model | Params | Fits 8 GB | Notes |
|---|---|---|---|
| DA-V2-Small-hf | ~25 M | trivially | CPU-viable, use for the live demo fallback |
| DA-V2-Base-hf | ~97 M | yes | good speed/quality trade |
| DA-V2-Large-hf | ~335 M | yes | **recommended default** |
| DA3MONO-LARGE | ViT-L/14 | yes | **recommended primary**, best quality |
| DA3-GIANT / NESTED | ViT-g | no on 8 GB, tight on 12 GB | gate behind a check; offer CPU offload |

Always: `torch.inference_mode()`, `autocast` fp16 on CUDA, `torch.cuda.empty_cache()` between
tiles, and an explicit `device` argument that falls back to CPU rather than crashing.

**Optional accuracy upgrade if you have training time (the 1–2 month track).** Fine-tune a
regression head on DFC2019 Track 1 nDSM using a frozen DINOv2/DA encoder + DPT decoder, or
reproduce the published remote-sensing-specific designs: **RDAH-Net** (*Remote Sens.* 2026,
18, 1024) explicitly uses relative depth as a geometry prior and learns terrain-dependent
scale — it is the closest published work to this exact problem statement. **TSE-Net**
(arXiv 2511.13552) adds semi-supervised pseudo-labelling. **HTC-DC Net** reframes height as
hierarchical classification, which fixes the long-tail-of-tall-buildings problem.




