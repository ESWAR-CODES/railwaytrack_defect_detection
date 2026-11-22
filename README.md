# AI-Based Multi-Defect Railway Monitoring & Severity Assessment Using YOLO-Based Drone Vision

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics-YOLOv8%2Fv11-00FFFF.svg)](https://docs.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10%2B-5C3EE8.svg)](https://opencv.org/)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063.svg)](https://docs.pydantic.dev/)
[![React](https://img.shields.io/badge/React-18%2B-61DAFB.svg)](https://react.dev/)

> **Mohan Babu University (MBU)**  
> **Department of Computer Science & Engineering (Artificial Intelligence & Machine Learning)**  
> **Capstone Project — Team B6-B3**

---

## 📌 Executive Summary & Problem Context

Railway track health is fundamental to transport safety and freight continuity. Traditional visual inspection relying on human track walkers is labor-intensive, hazardous, inherently qualitative, and incapable of matching the frequency required for dense railway corridors.

This project delivers an **autonomous, end-to-end aerial drone vision pipeline** capable of detecting multi-class railway defects, assessing defect severity through deterministic computer vision heuristics, and computing an aggregate **Track Risk Index (TRI)**. The system is designed with a strictly decoupled architecture: high-throughput Python backend for AI inference and telemetry, paired with an industrial-grade **Anveshana-themed** React dashboard for railway operators.

---

## 🏗️ System Architecture & Technology Stack

The platform is cleanly separated into two distinct tiers:

```
Railway-defect-detection/
├── backend/                  # Python 3.10+ AI Inference, Engine, & API Server
│   ├── api/                  # FastAPI routers, schema models, & endpoints
│   ├── configs/              # Canonical class mappings, YAML configurations
│   ├── data/                 # Raw inputs, processed splits, test samples
│   ├── engine/               # Geometry, severity scoring & config loaders
│   ├── models/               # Trained YOLO checkpoints (.pt)
│   ├── scripts/              # Preprocessing, training, and evaluation scripts
│   ├── tests/                # Pytest verification suites
│   └── requirements.txt      # Pinned backend dependencies
├── frontend/                 # React 18 + Vite + TypeScript (Phase 6)
│   └── ...                   # Anveshana-themed real-time monitoring UI
└── README.md
```

### Core Technologies
- **Object Detection**: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (YOLOv8 / YOLOv11) optimized for high-resolution aerial drone captures.
- **Image Processing**: [OpenCV Headless](https://opencv.org/) & [Albumentations](https://albumentations.ai/) for geometric defect profiling and perspective augmentation.
- **Backend API**: [FastAPI](https://fastapi.tiangolo.com/) with [Uvicorn](https://www.uvicorn.org/) for async batch/video inference streaming.
- **Validation & Serialization**: [Pydantic v2](https://docs.pydantic.dev/) & [PyYAML](https://pyyaml.org/) for type safety and deterministic configuration parsing.
- **Frontend UI (Phase 6)**: React 18, Vite, TypeScript, TailwindCSS with dark industrial telemetry design.

---

## 🏷️ Canonical Defect Taxonomy & Alias Resolution

To eliminate label drift across heterogeneous Roboflow datasets (e.g. Dataset A and Dataset B), the system enforces an exact **5-class canonical taxonomy**:

| Class ID | Canonical Name | Description | Common Roboflow Aliases Harmonized |
| :---: | :--- | :--- | :--- |
| **0** | `rail_crack` | Longitudinal cracks, transverse fractures, rail breaks, head checks, squats. | `crack`, `broken_rail`, `broken_rail_low`, `broken_rail_medium`, `broken_rail_high`, `squats`, `head_checks`, `glue_crack`, `railway_join_bar_defects` |
| **1** | `missing_fastener` | Missing, broken, or loosened clips/bolts securing rails to sleepers. | `fastener`, `broken_fastener`, `missing_clip`, `damaged_fastener`, `missing_bolt`, `loose_fastener` |
| **2** | `rail_corrosion` | Surface oxidation, flaking, and severe rust degradation on the rail head/flange. | `corrosion`, `corrosion_low`, `corrosion_medium`, `corrosion_high`, `rust`, `rusting`, `surface_corrosion` |
| **3** | `obstacle` | Foreign objects, ballast stones, rocks, or debris fouling the track clearance gauge. | `object`, `foreign_object`, `rock`, `stone`, `debris`, `boulder` |
| **4** | `sleeper_misalignment`| Skewed, displaced, cracked, or misaligned railway cross-ties/sleepers. | `sleeper`, `misaligned_sleeper`, `broken_sleeper`, `tie_defect`, `tie_misalignment`, `sleeper_defect` |

---

## 📐 Deterministic Severity Assessment & Track Risk Index (TRI)

Instead of relying on black-box confidence scores, severity is computed deterministically using physical dimensions and defect prevalence:

```
                                  [Drone Video Frame]
                                           │
                                           ▼
                             [YOLO Multi-Defect Detection]
                                           │
                      ┌────────────────────┴────────────────────┐
                      ▼                                         ▼
            [Geometric Defect Profile]               [Prevalence Defect Count]
           (rail_crack, rail_corrosion)         (missing_fastener, sleeper_misalignment)
                      │                                         │
                      ├─ diagonal_ratio                         ├─ frame_count
                      └─ area_ratio                             └─ cluster density
                      │                                         │
                      └────────────────────┬────────────────────┘
                                           ▼
                            [Deterministic Severity Engine]
                              ├── LOW    (Weight: 10,  #D97706)
                              ├── MEDIUM (Weight: 25,  #EA580C)
                              └── HIGH   (Weight: 50,  #DC2626)
                                           │
                                           ▼
                             [Track Risk Index (TRI) Score]
```

### Deterministic Thresholds
1. **`rail_crack`** (Diagonal Bounding Box / Rail Dimension Ratio):
   - **LOW**: $\text{ratio} < 0.03$
   - **MEDIUM**: $0.03 \le \text{ratio} \le 0.08$
   - **HIGH**: $\text{ratio} > 0.08$
2. **`rail_corrosion`** (Defect Area Ratio):
   - **LOW**: $\text{ratio} < 0.05$
   - **MEDIUM**: $0.05 \le \text{ratio} \le 0.15$
   - **HIGH**: $\text{ratio} > 0.15$
3. **`missing_fastener`** (Cumulative Cluster Defect Count):
   - **LOW**: $\text{count} = 1$
   - **MEDIUM**: $\text{count} = 2$
   - **HIGH**: $\text{count} \ge 3$
4. **`sleeper_misalignment`** (Misaligned Sleeper Count):
   - **LOW**: $\text{count} = 1$
   - **MEDIUM**: $\text{count} = 2$
   - **HIGH**: $\text{count} \ge 3$
5. **`obstacle`**:
   - **MANDATORY HIGH**: Any obstruction immediately presents derailment risk.

### Risk Weights & Severity Color Palette
- **LOW**: Weight = `10` | Hex = `#D97706` (Amber)
- **MEDIUM**: Weight = `25` | Hex = `#EA580C` (Orange)
- **HIGH**: Weight = `50` | Hex = `#DC2626` (Crimson)

$$\text{Track Risk Index (TRI)} = \sum_{i=1}^{N} \text{Weight}(\text{severity}_i)$$

---

## 🗂️ Project Directory Structure

```
Railway-defect-detection/
├── .gitignore
├── README.md
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── routes.py
│   │   ├── schemas.py
│   │   └── store.py
│   ├── configs/
│   │   ├── classes.yaml
│   │   └── railway_dataset.yaml
│   ├── data/
│   │   ├── processed/
│   │   │   ├── dataset_report.json
│   │   │   ├── images/
│   │   │   └── labels/
│   │   ├── raw/
│   │   │   ├── dataset_a/
│   │   │   └── dataset_b/
│   │   └── samples/
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── config_loader.py
│   │   ├── dataset_pipeline.py
│   │   ├── inference_service.py
│   │   ├── severity_engine.py
│   │   ├── vision_annotator.py
│   │   └── yolo_trainer.py
│   ├── models/
│   │   ├── best.pt
│   │   ├── evaluation_metrics.json
│   │   └── last.pt
│   ├── requirements.txt
│   ├── scripts/
│   │   ├── prepare_dataset.py
│   │   └── train_yolo.py
│   └── tests/
│       ├── __init__.py
│       ├── test_api.py
│       ├── test_config.py
│       ├── test_dataset_pipeline.py
│       └── test_severity_and_vision.py
├── frontend/
│   ├── .env.example
│   ├── index.html
│   ├── package.json
│   ├── postcss.config.js
│   ├── tailwind.config.js
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx
│       ├── index.css
│       ├── main.tsx
│       ├── vite-env.d.ts
│       ├── components/
│       │   ├── AuditHistoryView.tsx
│       │   ├── Footer.tsx
│       │   ├── HeroOverview.tsx
│       │   ├── InspectionWorkbench.tsx
│       │   ├── ModelDatasetView.tsx
│       │   └── Navbar.tsx
│       ├── services/
│       │   └── api.ts
│       └── types/
│           └── api.ts
```

---

## 🗺️ Project Execution Roadmap

- [x] **Phase 1: Project Scaffolding & Backend Configuration**
  - Scaffolding backend & frontend directory trees
  - Defining `requirements.txt` with pinned dependencies
  - Constructing `backend/configs/classes.yaml` with 5 canonical classes, Roboflow alias normalization, deterministic thresholds, and risk weights
  - Constructing `backend/configs/railway_dataset.yaml` for Ultralytics YOLO
  - Implementing Pydantic v2 `backend/engine/config_loader.py`
  - Passing all 67 test assertions in `backend/tests/test_config.py`
- [x] **Phase 2: Dataset Harmonization, Preprocessing & Augmentation**
  - Parsing raw YOLO/COCO datasets and resolving variant labels into canonical 5-class taxonomy
  - Clean unmapped label handling (dropping 144 normal `fastener_present` instances)
  - Clamping & validating YOLO bounding boxes strictly to `[0.0, 1.0]`
  - Deduplicating and merging `dataset_a` (2,651) and `dataset_b` (2,324) into 4,975 raw images with source tags (`dsa_`, `dsb_`)
  - Deterministic 70/20/10 split (Train: 3,482, Val: 995, Test: 498)
  - Drone-vision aerial augmentations (rotation ±15°, motion blur, fog/rain, illumination changes) adding 696 training samples (total: 5,671 processed images)
  - Generating comprehensive JSON report (`backend/data/processed/dataset_report.json`) and updating `backend/configs/railway_dataset.yaml`
  - Passing all 85 unit and integration tests in `backend/tests/`
- [x] **Phase 3: YOLO Model Training, Severity Engine & Vision Annotator**
  - Ultralytics YOLOv11/YOLOv8 training and validation engine with auto device detection (`cuda` / `cpu`)
  - Deterministic defect severity evaluation with geometric ratios (`area_ratio`, `diagonal_ratio`) and count escalation
  - Composite Track Risk Index (TRI) clamping to `[0.0, 100.0]` with actionable dispatch recommendations
  - High-contrast OpenCV visual annotation engine with telemetry HUD strip and thumbnail patch extraction
  - High-level inference orchestrator (`InferenceService`) for image paths, numpy arrays, and byte streams
  - Model checkpoint export (`backend/models/best.pt`, `last.pt`) and evaluation metrics report (`evaluation_metrics.json`)
  - Passing all 101 unit and integration tests in `backend/tests/`
- [x] **Phase 4: FastAPI REST Service & Inspection Backend**
  - Robust Pydantic v2 data schemas (`HealthResponse`, `DetectionItem`, `InspectionSummary`, `InspectionResponse`, `MetricsResponse`, `InspectionHistoryResponse`, `SampleListResponse`)
  - RESTful endpoints under `/api/v1` (`/health`, `/inspect`, `/metrics`, `/history`, `/samples`, `/samples/{filename}`)
  - Multipart image upload supporting `.jpg`, `.jpeg`, `.png`, `.webp` with base64 visual overlays and thumbnail crops
  - Pre-loaded dataset sample streaming and synthetic track reference generator for live faculty demos
  - Configurable query and form parameters with strict boundary clamping (`conf_threshold` between 0.05 and 0.95)
  - Thread-safe in-memory inspection history store with capacity limiting and eviction
  - Graceful fallback for offline execution or unpopulated weights without crashing
  - Comprehensive integration test suite using FastAPI `TestClient` passing all 18 tests
  - Passing all 119 unit and integration tests across the entire test suite
- [x] **Phase 5: React 18 + Vite + TypeScript Anveshana-Themed Dashboard**
  - Anveshana editorial design system (`#0B132B` Navy, `#1C2541` Elevated Navy, `#F5A623` Golden-Orange, `#FDFBF7` Warm Ivory)
  - Typography system loaded via Google Fonts (`Playfair Display`, `Inter`, `JetBrains Mono`)
  - Strict UI cleanliness: focused entirely on railway vision engineering without boxed white margins
  - Full-bleed edge-to-edge cinematic hero band and unboxed 5-class taxonomy cards
  - Dataset sample picker (1-click loading of real test split images) + custom drag-and-drop uploader
  - Dynamic class filtering checkboxes and three-way visual canvas (YOLO overlay, original, side-by-side)
  - Interactive lightbox, defect thumbnail zoom, and JSON/CSV triage export
  - Clean model evaluation and compact dataset partition summary
  - Compiles with 0 TypeScript and Vite build errors
  - 100% test preservation (119/119 backend tests green)

---

## ⚡ Quickstart & Verification

### 1. Install Dependencies
```bash
# Backend dependencies
pip install -r backend/requirements.txt

# Frontend dependencies
cd frontend
npm install
cd ..
```

### 2. Prepare & Harmonize Dataset
```bash
python backend/scripts/prepare_dataset.py
```

### 3. Train YOLO Model (Smoke Test / Full Training)
```bash
# Fast smoke test on CPU/GPU
python backend/scripts/train_yolo.py --epochs 2 --imgsz 320 --batch 16 --fraction 0.02

# Full production training run
python backend/scripts/train_yolo.py --epochs 50 --imgsz 640 --batch 16
```

### 4. Start FastAPI Backend Server
```bash
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation: [http://localhost:8000/docs](http://localhost:8000/docs).

### 5. Start React Frontend Client
```bash
cd frontend
npm run dev
```
Web Dashboard will be accessible at: [http://localhost:5173](http://localhost:5173).

### 6. Run Complete Test Suite
```bash
pytest backend/tests/ -v
```

All 119 tests will execute across configuration validation, dataset preprocessing & augmentation, geometric defect severity assessment, OpenCV vision annotation, YOLO model orchestration, FastAPI REST endpoints, and sample image streaming.
