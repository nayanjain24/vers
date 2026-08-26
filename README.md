# VERS v5.0 — Visual Emergency Response System & Multimodal Sign AI

[![CI/CD Pipeline](https://github.com/nayanjain24/vers/actions/workflows/ci.yml/badge.svg)](https://github.com/nayanjain24/vers/actions/workflows/ci.yml)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/nayanjain24/vers)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/frontend-React%20%2B%20Vite-61dafb.svg)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI%20%2B%20WebSockets-009688.svg)](https://fastapi.tiangolo.com/)

**VERS** is an AI-powered visual emergency detection and sign language communication platform. It analyzes real-time video streams to identify emergency distress signals, conversational sign language gestures (ASL & ISL), and facial distress metrics, instantly dispatching voice (TTS), SMS, and incident logs to emergency responders.

---

## 🌟 Key Features

- **Multimodal AI Pipeline**: Combines MediaPipe 3D Hand Tracking, Face Mesh landmark extraction, and a 31-class LSTM neural network with temporal attention.
- **30+ Sign Language Vocabulary**:
  - **Conversational Signs**: `HELLO`, `THANK_YOU`, `PLEASE`, `YES`, `NO`, `WATER`, `FOOD`, `FRIEND`, `FAMILY`, `NAME`, `GOOD`, `BAD`, `SORRY`, `PHONE`, `UNDERSTAND`, `WANT`, `MORE`, `WHERE`.
  - **Emergency Signs**: `HELP / SOS`, `MEDICAL`, `FIRE`, `POLICE`, `AMBULANCE`, `ACCIDENT`, `DANGER`, `PAIN`, `FALL`, `STOP`, `SAFE`.
- **Command Center Dashboard**: Glassmorphism React interface featuring live webcam streaming over WebSockets, real-time telemetry meters, an interactive **Sign Language Dictionary & Guide**, and 1-click incident simulators.
- **Direct Browser Camera Mode**: Hardware-agnostic browser webcam streaming (`navigator.mediaDevices.getUserMedia`) with real-time bounding boxes and landmark overlays.
- **Automated Dispatch & Multi-Channel Alerting**: Instant Voice TTS siren announcements, GPS coordinate simulation, and rate-limited webhook dispatches.

---

## 🚀 Running on GitHub (One-Click Cloud Execution)

You can run VERS directly inside your browser without installing anything on your machine:

1. Click the **[Open in GitHub Codespaces](https://codespaces.new/nayanjain24/vers)** button.
2. GitHub Codespaces will automatically build the environment, install dependencies, and launch both the backend API and frontend dashboard.
3. The dashboard will automatically open at **`http://localhost:5173`**.

---

## 💻 Running Locally

### Prerequisites
- **Python**: 3.9, 3.10, or 3.11
- **Node.js**: v18+ and `npm`

### 1-Click Startup:
```bash
git clone https://github.com/nayanjain24/vers.git
cd vers
./run.sh
```

### Manual Startup:
```bash
# 1. Set up Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Install Frontend dependencies
cd frontend && npm install && cd ..

# 3. Launch Orchestrator (Backend on :8000 + Frontend on :5173)
python src/orchestrate.py --mode dashboard
```

---

## 🌐 Application Endpoints

| Service | URL | Description |
| :--- | :--- | :--- |
| **Command Center Dashboard** | [http://localhost:5173](http://localhost:5173) | Interactive React + Vite UI |
| **Backend REST API** | [http://localhost:8000](http://localhost:8000) | FastAPI core services |
| **Interactive API Docs** | [http://localhost:8000/docs](http://localhost:8000/docs) | Swagger UI for endpoint testing |
| **Live Stream WebSocket** | `ws://localhost:8000/api/v1/stream` | Bidirectional frame & telemetry stream |

---

## 📖 Sign Language Glossary

For the complete hand geometry diagrams and descriptions, see [SIGN_LANGUAGE_GLOSSARY.md](SIGN_LANGUAGE_GLOSSARY.md) or open the built-in dictionary in the Command Center.

| Sign | Category | Hand Movement / Geometry | Threat Level |
| :--- | :--- | :--- | :--- |
| **HELLO** | Conversational | Open hand waving side to side near temple | `Low` |
| **THANK_YOU** | Conversational | Flat hand moving from chin forward | `Low` |
| **PLEASE** | Conversational | Flat palm circular motion on chest | `Low` |
| **WATER** | Needs | W-handshape (3 fingers) tapped at chin | `Low` |
| **PHONE** | Needs | Y-handshape (thumb + pinky) at ear | `Low` |
| **HELP / SOS** | Emergency | 5 fingers extended wide / fist on flat palm | `Critical` |
| **MEDICAL** | Emergency | 4 fingers extended / crossed index fingers | `Critical` |
| **FIRE** | Emergency | Index finger pointing up with waving flickers | `High` |
| **POLICE** | Emergency | Index + Middle in V-shape | `High` |
| **SAFE** | Emergency | Thumbs up / situation under control | `Low` |

---

## 🧪 Running Automated Tests

Run the complete 38-case test suite:
```bash
source .venv/bin/activate
pytest tests/ -v
```

---

## 📄 License
This project is licensed under the MIT License.
