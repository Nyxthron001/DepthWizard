import React, { useState, useRef, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Center } from '@react-three/drei';
import * as THREE from 'three';
import './DepthWizardViewer.css';

const Terrain = ({ depthMapUrl, textureUrl, displacementScale = 3, onHover }) => {
  const meshRef = useRef();

  useEffect(() => {
    if (!depthMapUrl || !textureUrl) return;

    const loader = new THREE.TextureLoader();
    let loadedTexture = null;
    let loadedDepth = null;

    const checkAndUpdate = () => {
      const mesh = meshRef.current;
      if (!mesh || !loadedTexture || !loadedDepth) return;
      const mat = mesh.material;
      loadedTexture.colorSpace = THREE.SRGBColorSpace;
      mat.map = loadedTexture;
      mat.color.set(0xffffff); // dark idle tint drops away once the texture drapes
      mat.displacementMap = loadedDepth;
      mat.displacementScale = displacementScale;
      mat.displacementBias = -displacementScale / 2; // center mesh around origin
      mat.needsUpdate = true;
    };

    loader.load(textureUrl, (tex) => {
      loadedTexture = tex;
      checkAndUpdate();
    });
    loader.load(depthMapUrl, (depth) => {
      loadedDepth = depth;
      checkAndUpdate();
    });
  }, [depthMapUrl, textureUrl, displacementScale]);

  return (
    <mesh
      ref={meshRef}
      rotation={[-Math.PI / 2, 0, 0]}
      onPointerMove={(e) => onHover && onHover(e)}
    >
      <planeGeometry args={[12, 12, 512, 512]} />
      <meshStandardMaterial color="#26282c" roughness={0.6} metalness={0.1} wireframe={false} />
    </mesh>
  );
};

const DepthWizardViewer = () => {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState('idle'); // idle | processing | ready | error
  const [errorMsg, setErrorMsg] = useState('');
  const [displacementScale, setDisplacementScale] = useState(3);
  const [hoverInfo, setHoverInfo] = useState({ x: 0, y: 0, z: 0 });
  const fileInputRef = useRef();
  const controlsRef = useRef();

  const onFileChange = (e) => {
    const next = e.target.files?.[0] ?? null;
    setFile(next);
    setResult(null);
    setErrorMsg('');
    setStatus('idle');
    if (e.target) e.target.value = ''; // allow re-selecting the same file
  };

  const clearFile = () => {
    setFile(null);
    setResult(null);
    setErrorMsg('');
    setStatus('idle');
  };

  const handleUpload = async () => {
    if (!file) return;
    setStatus('processing');
    setErrorMsg('');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/process', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || `Server said: ${response.status}`);
      }
      setResult({
        depthUrl: `http://localhost:8000${data.result_url}`,
        textureUrl: `http://localhost:8000${data.original_url}`,
        exportPng16Url: data.export_png16_url ? `http://localhost:8000${data.export_png16_url}` : null,
        exportObjUrl: data.export_obj_url ? `http://localhost:8000${data.export_obj_url}` : null,
      });
      setStatus('ready');
    } catch (err) {
      console.error('Error processing image:', err);
      setErrorMsg(err.message || 'Something went wrong while processing your image.');
      setStatus('error');
    }
  };

  const resetCamera = () => controlsRef.current?.reset();

  const handleTerrainHover = (event) => {
    const { x, y, z } = event.point;
    setHoverInfo({ x, y, z });
  };

  const busy = status === 'processing';

  return (
    <div className="dw-app">
      <header className="dw-header">
        <div className="dw-brand">
          <h1 className="dw-brand-name">DepthWizard</h1>
          <p className="dw-brand-sub">ISRO · SIH 2026 · single image to 3D terrain</p>
        </div>

        <div className="dw-toolbar" role="toolbar" aria-label="Terrain controls">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={onFileChange}
          />

          <button className="dw-btn" onClick={() => fileInputRef.current?.click()}>
            {file ? 'Change image' : 'Choose image'}
          </button>

          {file && !busy && (
            <span className="dw-file">
              <span className="dw-file-name" title={file.name}>{file.name}</span>
              <button
                className="dw-file-remove"
                onClick={clearFile}
                aria-label="Remove selected image"
              >
                ×
              </button>
            </span>
          )}

          <button
            className="dw-btn dw-btn-primary"
            onClick={handleUpload}
            disabled={!file || busy}
          >
            {busy ? 'Estimating…' : 'Generate terrain'}
          </button>

          <label className="dw-scale">
            <span className="dw-scale-label">Height scale</span>
            <input
              type="range"
              min="0.5"
              max="10"
              step="0.5"
              value={displacementScale}
              onChange={(e) => setDisplacementScale(parseFloat(e.target.value))}
            />
            <span className="dw-scale-value">{displacementScale}×</span>
          </label>

          <button className="dw-btn" onClick={resetCamera}>Reset view</button>

          {result?.exportPng16Url && result?.exportObjUrl && (
            <>
              <a className="dw-btn dw-btn-download" href={result.exportPng16Url} download>
                16-bit height map
              </a>
              <a className="dw-btn dw-btn-download" href={result.exportObjUrl} download>
                Mesh (.obj)
              </a>
            </>
          )}
        </div>
      </header>

      <main className="dw-scene">
        <Canvas shadows gl={{ alpha: true, antialias: true }}>
          <PerspectiveCamera makeDefault position={[0, 12, 12]} />
          <OrbitControls ref={controlsRef} makeDefault />

          <ambientLight intensity={0.7} />
          <directionalLight position={[10, 20, 10]} intensity={1.3} castShadow />
          <pointLight position={[-10, 10, -10]} intensity={0.35} color="#f2e6d0" />

          <Center>
            <Terrain
              depthMapUrl={result?.depthUrl}
              textureUrl={result?.textureUrl}
              displacementScale={displacementScale}
              onHover={handleTerrainHover}
            />
          </Center>

          <gridHelper args={[20, 20]} material={new THREE.LineBasicMaterial({ color: '#26282c' })} />
          <fog attach="fog" args={['#0d0e10', 10, 50]} />
        </Canvas>

        {status === 'idle' && (
          <section className="dw-empty" aria-label="How it works">
            <div className="dw-empty-inner">
              <h2>Estimate terrain height from a single image</h2>
              <p>
                Pick an optical satellite image. DepthWizard estimates relative
                surface height, then drapes the original image over it as a 3D
                scene you can explore.
              </p>
              <ol className="dw-steps">
                <li>Choose a satellite image</li>
                <li>Estimate surface heights</li>
                <li>Explore the terrain in 3D</li>
              </ol>
            </div>
          </section>
        )}

        {busy && (
          <p className="dw-status" role="status">
            <span className="dw-spinner" aria-hidden="true" />
            Estimating surface heights…
          </p>
        )}
        {status === 'error' && (
          <p className="dw-status dw-status-error" role="alert">{errorMsg}</p>
        )}

        {status === 'ready' && (
          <aside className="dw-hud" aria-label="Terrain telemetry">
            <h2 className="dw-hud-title">Terrain telemetry</h2>
            <dl className="dw-hud-rows">
              <div className="dw-hud-row">
                <dt>x</dt>
                <dd>{hoverInfo.x.toFixed(2)}</dd>
              </div>
              <div className="dw-hud-row">
                <dt>y</dt>
                <dd>{hoverInfo.y.toFixed(2)}</dd>
              </div>
              <div className="dw-hud-row">
                <dt>Elevation (relative)</dt>
                <dd>{hoverInfo.z.toFixed(4)}</dd>
              </div>
            </dl>
            <p className="dw-hud-note">Relative depth — uncalibrated, not metres</p>
          </aside>
        )}

        <p className="dw-hint">Drag to orbit · scroll to zoom · right-drag to pan</p>
      </main>
    </div>
  );
};

export default DepthWizardViewer;
