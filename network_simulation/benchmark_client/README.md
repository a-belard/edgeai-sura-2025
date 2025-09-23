# RTT Benchmark Client

A Python client for testing real-world Round Trip Time (RTT) against the SqueezeNet FastAPI server. This provides accurate network latency measurements for edge AI benchmarking.

## Features

- **Real RTT measurement** - Actual network timing vs synthetic delays
- **Concurrent load testing** - Test server performance under multiple clients
- **Image size impact analysis** - Measure transfer time vs image dimensions
- **Comprehensive statistics** - P95, P99, mean, median latency metrics
- **Cross-platform support** - Works on Raspberry Pi, PC, cloud instances
- **JSON output format** - Compatible with existing benchmark aggregation

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic RTT Test
```bash
python client.py http://192.168.1.100:8000 --requests 50
```

### Concurrent Load Test
```bash
python client.py http://server:8000 --concurrent 1 2 4 8 --requests 100
```

### Full Benchmark Suite
```bash
python client.py http://server:8000 --requests 100 --concurrent 1 2 4 8 --image-sizes
```

## Command Line Options

- `server_url`: FastAPI server URL (required)
- `--requests`: Number of requests for latency test (default: 50)
- `--concurrent`: List of concurrent user counts to test (default: [1, 2, 4])
- `--output`: Output directory for results (default: "results")
- `--image-sizes`: Test different image sizes impact
- `--verbose`: Enable detailed logging

## Test Scenarios

### 1. Latency Benchmark
- Measures single-threaded request/response time
- Separates server processing from network overhead
- Provides percentile statistics (P95, P99)

### 2. Concurrent Load Test
- Tests multiple simultaneous clients
- Measures requests per second under load
- Identifies server saturation points

### 3. Image Size Impact
- Tests 224x224, 512x512, 1024x1024 images
- Measures network transfer time scaling
- Useful for bandwidth-limited scenarios

## Output Format

Results are saved in JSON format compatible with existing benchmark tools:

```json
{
  "system_info": {
    "timestamp": "2025-09-23T15:30:00",
    "client_info": {...},
    "server_info": {...}
  },
  "latency_benchmark": {
    "total_time_stats": {
      "mean_ms": 45.67,
      "median_ms": 42.31,
      "p95_ms": 78.45,
      "p99_ms": 125.67
    },
    "network_overhead_stats": {...}
  },
  "concurrent_benchmark": {...}
}
```

## Real-World Testing Scenarios

### Edge-to-Cloud RTT
```bash
# RPi client → Cloud server
python client.py http://cloud-server:8000 --requests 200
```

### LAN Performance
```bash
# PC client → Local server
python client.py http://192.168.1.100:8000 --concurrent 1 4 8 16
```

### Bandwidth Impact
```bash
# Test image size scaling
python client.py http://server:8000 --image-sizes --requests 50
```

## Integration with Existing Benchmarks

The client output format is designed to integrate with the existing SqueezeNet benchmark aggregation:

1. Run client against multiple server deployments
2. Collect JSON results in appropriate results/ subdirectories
3. Use existing Excel aggregation scripts to combine with local benchmarks
4. Compare real RTT vs synthetic delay measurements

## Performance Baselines

Expected RTT ranges for different deployment scenarios:

- **LAN (Ethernet)**: 1-5 ms network overhead
- **WiFi (local)**: 5-15 ms network overhead  
- **5G/LTE**: 20-100 ms network overhead
- **Satellite**: 500+ ms network overhead

## Troubleshooting

### Connection Issues
```bash
# Test server connectivity
curl http://server:8000/health
```

### High Latency
- Check network congestion
- Verify server resource availability
- Test with smaller concurrent loads

### Failed Requests
- Increase request timeout
- Check server logs for errors
- Verify image encoding/decoding