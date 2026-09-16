@echo off
echo ========================================================
echo Starting DepthWizard ISRO 3D Elevation Pipeline...
echo ========================================================
start "DepthWizard Backend (FastAPI)" cmd /k "cd /d %~dp0 && python -m uvicorn backend.app.main:app --reload --port 8000"
start "DepthWizard Frontend (Vite React)" cmd /k "cd /d %~dp0frontend && npm run dev"
echo Servers are launching in separate windows!
echo Open http://localhost:5173 in your browser once ready.
pause

