# Edge AI Benchmarking Environment

Virtual environment with scripts and notebooks for benchmarking compact CNN models (SqueezeNet, MobileNetV2) across edge and cloud platforms.

## Project Structure
```
edgeai-env/
├─ README.md                          # Main documentation
├─ README_RTT_Simulation.md           # Network delay simulation guide
├─ mobilenetv2.tflite                 # Base TFLite model (MobileNetV2 quantized)
├─ docker-compose.yml                 # Docker configuration for server deployment
├─ pyvenv.cfg                         # Python virtual environment metadata
├─ bin/                               # Virtual environment executables
├─ include/                           # Virtual environment headers
├─ lib/                               # Installed Python dependencies
├─ share/                             # Shared data (jupyter kernels, documentation)
├─ results/                           # Benchmark results by platform
│  ├─ pc/                             # Desktop/laptop benchmark results
│  ├─ rpi/                            # Raspberry Pi benchmark results
│  └─ googlecollab/                   # Google Colab benchmark results
├─ inference_evaluation/              # Local inference benchmarking
│  ├─ mobilenetv2_benchmark.py        # MobileNetV2 TFLite benchmark script
│  ├─ squeezenet_benchmark.py         # SqueezeNet PyTorch benchmark script
│  ├─ mobilenetv2_comprehensive_eval.ipynb     # MobileNetV2 analysis notebook
│  ├─ squeezenet_comprehensive_eval.ipynb      # SqueezeNet analysis notebook
│  ├─ squeezenet_platform_comparison.ipynb     # Cross-platform comparison
│  ├─ device_model_comparison.ipynb            # Device and model comparison
│  ├─ aggregate_results_to_excel.py            # Result aggregation utility
│  ├─ requirements.txt                         # Python dependencies
│  └─ results/                                 # Local benchmark results
├─ network_simulation/                # Real network delay testing
│  ├─ fastapi_server/                 # FastAPI inference server
│  └─ benchmark_client/               # Network benchmark client
│     ├─ client.py                    # HTTP benchmark client script
│     ├─ rtt_analysis.ipynb           # Network delay analysis notebook
│     ├─ results-*/                   # RTT benchmark results by delay scenario
│     └─ requirements.txt             # Client dependencies
└─ jupyter/                           # Legacy exploratory notebooks
   ├─ mobilenetv2.ipynb
   ├─ squeezenet.ipynb
   └─ models/                         # Cached model files
```

## Key Components

### Local Inference Benchmarking
- **Benchmark Scripts**: Automated performance testing for MobileNetV2 and SqueezeNet models
- **Analysis Notebooks**: Interactive evaluation and visualization of benchmark results
- **Platform Comparison**: Cross-platform performance analysis across different hardware

### Network Simulation Environment
- **FastAPI Server**: HTTP inference server for real network testing
- **Benchmark Client**: Network delay simulation and RTT measurement
- **RTT Analysis**: Real network delay impact analysis with multiple delay scenarios

### Result Storage
- **JSON Files**: Detailed benchmark data including latency distributions and system info
- **CSV Summaries**: Key performance indicators for quick analysis
- **Platform Results**: Organized by device type (PC, Raspberry Pi, Google Colab)

## Getting Started

### Environment Setup
```bash
# Activate the virtual environment
source bin/activate

# Install dependencies for local benchmarks
pip install -r inference_evaluation/requirements.txt

# Install dependencies for network simulation
pip install -r network_simulation/benchmark_client/requirements.txt
```

### Running Local Benchmarks
```bash
# Run SqueezeNet benchmark
python inference_evaluation/squeezenet_benchmark.py --runs 100 --output results/pc

# Run MobileNetV2 benchmark
python inference_evaluation/mobilenetv2_benchmark.py --runs 100 --output results/pc
```

### Running Network Simulation
```bash
# Start the FastAPI server (in one terminal)
cd network_simulation/fastapi_server
python main.py

# Run network delay benchmark (in another terminal)
cd network_simulation/benchmark_client
python client.py http://localhost:8000
```

## Analysis and Visualization

### Platform Comparison
Use the `squeezenet_platform_comparison.ipynb` notebook to:
- Compare performance across different platforms
- Generate visualization charts
- Export results to CSV format

### Network Delay Analysis
Use the `rtt_analysis.ipynb` notebook to:
- Analyze real network delay impacts
- Compare efficiency across delay scenarios
- Generate network performance reports

## File Formats

### Benchmark Results
- **JSON Files**: Complete benchmark data with detailed statistics
  - Latency distributions (mean, median, p95, p99)
  - System information and configuration
  - Individual request timings
- **CSV Files**: Summary statistics for quick analysis
  - Key performance metrics
  - Aggregated results across runs

### Network Simulation Results
- **RTT Results**: Real round-trip time measurements
  - Total response time statistics
  - Network overhead calculations
  - Efficiency ratios relative to baseline

## Adding New Platform Results

1. Run benchmarks on the new platform using the provided scripts
2. Copy result files to a new folder under `results/` (example: `results/edge_device/`)
3. Ensure file naming follows the pattern: `model_results_timestamp.json`
4. Update analysis notebooks to include the new platform data

## Docker Deployment

The project includes Docker configuration for server deployment:
```bash
# Start the inference server using Docker
docker-compose up
```

## Dependencies

Main Python packages used:
- **PyTorch**: Deep learning framework for SqueezeNet
- **TensorFlow Lite**: Lightweight inference for MobileNetV2
- **FastAPI**: HTTP server for network simulation
- **Pandas**: Data analysis and manipulation
- **Matplotlib/Seaborn**: Visualization and plotting
- **Jupyter**: Interactive notebook environment

## Notes

- Results folder contains benchmark data from multiple platforms
- Network simulation provides real-world deployment insights
- All notebooks are self-contained with installation instructions
- Benchmark scripts support command-line options for automation