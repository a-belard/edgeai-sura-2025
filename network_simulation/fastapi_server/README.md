# SqueezeNet FastAPI Server

A FastAPI server that hosts the SqueezeNet model for real-time inference and RTT simulation testing.

## Features

- **RESTful API** for model inference
- **Base64 image encoding** for network transmission
- **File upload support** for direct image uploads
- **Health monitoring** endpoints
- **CUDA/MPS support** for GPU acceleration
- **Quantization support** for optimized inference
- **Detailed timing metrics** for performance analysis

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the server:
```bash
python server.py --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Image Inference (Base64)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image_data": "base64_encoded_image_here"}'
```

### File Upload Inference
```bash
curl -X POST http://localhost:8000/predict_file \
  -F "file=@test_image.jpg"
```

### Model Information
```bash
curl http://localhost:8000/model_info
```

## Command Line Options

- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port to bind to (default: 8000)
- `--quantized`: Use quantized model for faster inference
- `--workers`: Number of worker processes (default: 1)

## Usage Examples

### Basic server start:
```bash
python server.py
```

### GPU server with quantization:
```bash
python server.py --port 8080 --quantized
```

### Production deployment:
```bash
python server.py --host 0.0.0.0 --port 8000 --workers 4
```

## Response Format

All inference endpoints return:
```json
{
  "predictions": [281, 285, 282, 283, 287],
  "confidence_scores": [0.945, 0.032, 0.015, 0.005, 0.002],
  "processing_time_ms": 12.34,
  "model_info": {
    "name": "SqueezeNet 1.1",
    "size_mb": 4.71,
    "device": "cuda",
    "quantized": false
  },
  "timestamp": 1695456789.123
}
```

## Integration with Benchmark Client

This server is designed to work with the benchmark client for real RTT simulation testing, providing more accurate network delay measurements than synthetic sleep-based approaches.