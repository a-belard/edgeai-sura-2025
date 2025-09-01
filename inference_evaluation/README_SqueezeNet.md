# SqueezeNet Comprehensive Evaluation

This directory contains comprehensive benchmarking tools for SqueezeNet model evaluation on laptop/desktop systems for edge AI research.

## Overview

SqueezeNet is a lightweight convolutional neural network architecture that achieves AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size when compressed. This evaluation suite provides detailed performance analysis for edge AI deployment scenarios.

### Key Features
- **Lightweight Architecture**: Fire modules with squeeze and expand layers
- **Small Model Size**: ~4.8 MB for SqueezeNet 1.1
- **Efficient Inference**: Optimized for edge deployment
- **Cross-Platform**: PyTorch implementation with device flexibility

## Files

### Core Evaluation Tools
- `squeezenet_benchmark.py` - Standalone Python benchmarking script
- `squeezenet_comprehensive_eval.ipynb` - Interactive Jupyter notebook with visualizations
- `requirements.txt` - Python dependencies (updated with PyTorch)

### Usage

#### Option 1: Standalone Script
```bash
# Install dependencies
pip install -r requirements.txt

# Run basic benchmark
python squeezenet_benchmark.py

# Run with custom parameters
python squeezenet_benchmark.py --runs 200 --output my_results/

# Run with quantization
python squeezenet_benchmark.py --quantized --runs 100
```

#### Option 2: Jupyter Notebook
```bash
# Launch Jupyter
jupyter notebook

# Open squeezenet_comprehensive_eval.ipynb
# Run all cells for complete evaluation with visualizations
```

## Benchmark Categories

### 1. Core Performance
- **Inference Latency**: Mean, std, percentiles (P95, P99)
- **Throughput**: Images per second
- **Resource Usage**: CPU, memory, GPU (if available)
- **Device Optimization**: Apple Silicon (MPS), CUDA, CPU

### 2. Batch Processing
- **Batch Sizes**: 1, 5, 10, 20, 50 images
- **Efficiency Analysis**: Time per image scaling
- **Memory Scaling**: Resource usage vs batch size
- **Throughput Optimization**: Maximum sustainable throughput

### 3. Network Latency Tolerance
- **Simulated Delays**: 0ms to 1000ms network latency
- **Efficiency Ratios**: Computation vs communication time
- **Edge Scenarios**: Real-world deployment simulation
- **Delay-Tolerant Performance**: Practical edge AI evaluation

### 4. Model Comparison
- **vs MobileNetV2**: Direct performance comparison
- **Size Efficiency**: Model size vs accuracy trade-offs
- **Speed Comparison**: Inference time analysis
- **Framework Differences**: PyTorch vs TensorFlow Lite

### 5. Quantization Analysis
- **FP32 vs INT8**: Precision trade-offs
- **Model Compression**: Size reduction analysis
- **Performance Impact**: Speed vs accuracy
- **Edge Deployment**: Memory-constrained scenarios

## Results Structure

```
results/
├── squeezenet_results_YYYYMMDD_HHMMSS.json          # Comprehensive results
├── squeezenet_summary_YYYYMMDD_HHMMSS.csv           # Summary metrics
├── squeezenet_detailed_YYYYMMDD_HHMMSS.csv          # Per-run detailed data
└── squeezenet_comprehensive_YYYYMMDD_HHMMSS.json    # Notebook results
```

### Key Metrics Tracked
- **Performance**: Mean inference time, throughput, latency percentiles
- **Resources**: CPU usage, memory consumption, GPU utilization
- **Efficiency**: Network tolerance, batch processing efficiency
- **Deployment**: Cold start overhead, quantization effects

## Device Support

### Optimized Platforms
- **Apple Silicon**: MPS (Metal Performance Shaders) acceleration
- **NVIDIA GPU**: CUDA acceleration with FP16 support
- **Intel/AMD CPU**: Optimized CPU inference
- **Edge Devices**: Quantized model support

### Device Selection Logic
1. Apple Silicon (M1/M2/M3) → MPS
2. NVIDIA GPU → CUDA
3. Fallback → CPU

## Model Architecture Details

### SqueezeNet 1.1 Specifications
- **Parameters**: ~1.24M
- **Model Size**: ~4.8 MB (FP32)
- **Input Size**: 224×224×3
- **Output Classes**: 1000 (ImageNet)
- **Key Innovation**: Fire modules (squeeze + expand)

### Fire Module Design
```
Input → Squeeze (1×1 conv) → [Expand 1×1 | Expand 3×3] → Concatenate → Output
```

## Comparison with MobileNetV2

| Metric | SqueezeNet 1.1 | MobileNetV2 | Notes |
|--------|----------------|-------------|--------|
| Parameters | ~1.24M | ~3.5M | SqueezeNet is smaller |
| Model Size | ~4.8 MB | ~14 MB | Significant size advantage |
| Framework | PyTorch | TensorFlow Lite | Different optimization paths |
| Architecture | Fire modules | Inverted residuals | Different design philosophies |

## Advanced Features

### 1. Precision Mode Testing
- **FP32**: Standard floating-point precision
- **FP16**: Half-precision (GPU only)
- **INT8**: Quantized inference for edge deployment

### 2. Cold Start Analysis
- **Model Loading Time**: Initialization overhead
- **First Inference**: Cold vs warm start comparison
- **Memory Allocation**: Initial resource requirements

### 3. Batch Efficiency
- **Optimal Batch Size**: Performance sweet spot identification
- **Memory Constraints**: Batch size vs memory usage
- **Throughput Scaling**: Linear vs sublinear scaling analysis

## Edge AI Research Applications

### Deployment Scenarios
1. **Mobile Devices**: iOS/Android on-device inference
2. **IoT Devices**: Raspberry Pi, Jetson Nano
3. **Edge Servers**: Local processing nodes
4. **Real-time Systems**: Low-latency requirements

### Research Use Cases
- **Model Compression**: Architecture efficiency studies
- **Edge Optimization**: Hardware-software co-design
- **Latency Analysis**: Real-time system design
- **Resource Constraints**: Limited compute environments

## Troubleshooting

### Common Issues

1. **PyTorch Installation**
   ```bash
   # For Apple Silicon
   pip install torch torchvision

   # For CUDA
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. **MPS Device Errors**
   ```python
   # Check MPS availability
   torch.backends.mps.is_available()
   ```

3. **Memory Issues**
   - Reduce batch sizes for large models
   - Use CPU for memory-constrained systems
   - Enable model quantization

### Performance Tips
- **Warm-up Runs**: Always include warm-up for accurate timing
- **Device Selection**: Use appropriate device for your hardware
- **Batch Optimization**: Find optimal batch size for your use case
- **Quantization**: Consider INT8 for deployment scenarios

## Future Enhancements

### Planned Features
- **ONNX Export**: Cross-framework compatibility
- **TensorRT**: NVIDIA optimization
- **Core ML**: Apple ecosystem integration
- **Model Pruning**: Further compression techniques

### Research Extensions
- **Architecture Variants**: SqueezeNet 1.0 comparison
- **Custom Fire Modules**: Architecture exploration
- **Mixed Precision**: Advanced quantization schemes
- **Hardware Profiling**: Detailed device-specific analysis

## Contributing

When extending this evaluation suite:
1. Follow the existing benchmarking patterns
2. Include comprehensive error handling
3. Add visualization for new metrics
4. Update documentation for new features
5. Maintain backward compatibility with existing results

## References

- **SqueezeNet Paper**: "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size"
- **PyTorch Documentation**: Official PyTorch model zoo
- **Edge AI Research**: Lightweight model deployment studies
