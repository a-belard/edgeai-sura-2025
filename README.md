# Edge AI Benchmarking Environment

Lightweight virtual environment + scripts + notebooks for benchmarking compact CNN models (SqueezeNet, MobileNetV2) across  edge and cloud platforms.

## Folder Structure (Top Level)
```
edgeai-env/
├─ README.md                # (this file)
├─ mobilenetv2.tflite       # Base TFLite model (MobileNetV2 quantized)
├─ pyvenv.cfg               # Python virtual environment metadata
├─ bin/                     # Virtualenv executables (python, pip, jupyter, etc.)
├─ include/                 # Virtualenv headers
├─ lib/                     # Site-packages (installed dependencies)
├─ share/                   # Shared data (jupyter kernels, man pages)
├─ results/                 # Aggregated benchmark artifacts by platform
│  ├─ pc/                   # Laptop / desktop results (JSON, CSV)
│  ├─ rpi/                  # Raspberry Pi results (JSON, CSV) (note typo squuezenet.json)
│  └─ googlecollab/         # Colab captured JSON results
├─ inference_evaluation/    # Core benchmarking scripts & analysis notebooks
│  ├─ mobilenetv2_benchmark.py            # Comprehensive MobileNetV2 TFLite benchmark
│  ├─ squeezenet_benchmark.py             # Comprehensive SqueezeNet PyTorch benchmark
│  ├─ mobilenetv2_comprehensive_eval.ipynb# Notebook: end-to-end MobileNetV2 evaluation
│  ├─ squeezenet_comprehensive_eval.ipynb # Notebook: end-to-end SqueezeNet evaluation
│  ├─ squeezenet_platform_comparison.ipynb# Multi-platform visualization & export
│  ├─ device_model_comparison.ipynb       # Cross-model/device comparison exploratory
│  ├─ requirements.txt                    # Python deps for evaluation layer
│  ├─ README.md / README_*                # Model/platform specific summaries
│  └─ results/                            # (Optional) intra-module result staging
├─ jupyter/                 # Older exploratory notebooks & models folder
│  ├─ mobilenetv2.ipynb
│  ├─ squeezenet.ipynb
│  └─ models/               # Original downloaded / cached model assets
```

## Key Artifacts
- `results/*/*.json`  Detailed raw benchmark outputs (latency distributions, batch, cold/warm, network delay).
- `results/*/*_summary_*.csv`  Summarized KPIs (mean latency, p95, throughput, efficiency).
- `inference_evaluation/*_benchmark.py`  Reusable scripts for automated runs (CLI friendly).
- `inference_evaluation/*_comprehensive_eval.ipynb`  Notebooks reproducing script flow for interactive analysis.
- `inference_evaluation/squeezenet_platform_comparison.ipynb`  Generates JSON/CSV plus comparative plots (latency, resource use, network efficiency, cold vs warm).

## Quick Start
Activate environment (already a venv directory):
```
source bin/activate
pip install -r inference_evaluation/requirements.txt  # if updating deps
```
Run a benchmark (example):
```
python inference_evaluation/squeezenet_benchmark.py --runs 100 --output results/pc
```
View summary CSV & JSON in `results/pc/` after completion.

## Adding New Platform Results
1. Copy the two output files (`*_results_*.json`, `*_summary_*.csv`) into a new subfolder under `results/` (e.g., `results/edge_device/`).
2. Ensure naming pattern matches existing (`squeezenet_results_<timestamp>.json`).
3. Re-run the platform comparison notebook to include the new platform automatically (update folder mapping if needed).

## Known Cleanups / TODO
- Rename `results/rpi/squuezenet.json` -> `squeezenet_results_<timestamp>.json` for consistency.
- Add quantized runs for all platforms (only Colab currently shows quantization comparison in JSON).
- Add GPU / mixed precision (FP16) results where hardware is available.
- Introduce energy measurement (e.g., RAPL on PC, INA219 on Pi) into result schema.