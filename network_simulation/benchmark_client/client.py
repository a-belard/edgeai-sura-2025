#!/usr/bin/env python3
"""
Real RTT Benchmark Client
Tests actual network latency against FastAPI server hosting SqueezeNet model.
"""

import os
import sys
import time
import json
import base64
import asyncio
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import argparse
import logging

import requests
import numpy as np
import pandas as pd
from PIL import Image
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RTTBenchmarkClient:
    def __init__(self, server_url: str, output_dir: str = "results"):
        self.server_url = server_url.rstrip('/')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Test connectivity
        self.verify_server_connection()
        
    def verify_server_connection(self):
        """Test server connectivity and get model info"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=10)
            response.raise_for_status()
            health_data = response.json()
            
            if not health_data.get('model_loaded', False):
                raise Exception("Model not loaded on server")
                
            logger.info(f"Connected to server: {health_data}")
            
        except Exception as e:
            logger.error(f"Server connection failed: {e}")
            raise
    
    def create_test_image(self, size: Tuple[int, int] = (224, 224)) -> str:
        """Create a test image and encode as base64"""
        # Create random RGB image
        image_array = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
        image = Image.fromarray(image_array, 'RGB')
        
        # Convert to base64
        import io
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return image_data
    
    def load_test_image(self, image_path: str) -> str:
        """Load image from file and encode as base64"""
        try:
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            return image_data
        except Exception as e:
            logger.warning(f"Failed to load {image_path}: {e}")
            return self.create_test_image()
    
    def single_inference_request(self, image_data: str) -> Dict:
        """Send single inference request and measure timing"""
        request_data = {
            "image_data": image_data,
            "include_timing": True
        }
        
        # Measure total request time
        start_time = time.perf_counter()
        
        try:
            response = requests.post(
                f"{self.server_url}/predict",
                json=request_data,
                timeout=30
            )
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None
            
        end_time = time.perf_counter()
        
        total_time_ms = (end_time - start_time) * 1000
        response_data = response.json()
        
        return {
            'total_time_ms': total_time_ms,
            'server_processing_ms': response_data.get('processing_time_ms', 0),
            'network_overhead_ms': total_time_ms - response_data.get('processing_time_ms', 0),
            'predictions': response_data.get('predictions', []),
            'confidence_scores': response_data.get('confidence_scores', []),
            'timestamp': time.time()
        }
    
    def benchmark_latency(self, num_requests: int = 50, warmup_requests: int = 5) -> Dict:
        """Benchmark inference latency with multiple requests"""
        logger.info(f"Starting latency benchmark: {warmup_requests} warmup + {num_requests} test requests")
        
        # Create test image
        image_data = self.create_test_image()
        
        # Warmup requests
        logger.info("Performing warmup requests...")
        for i in range(warmup_requests):
            result = self.single_inference_request(image_data)
            if result is None:
                logger.warning(f"Warmup request {i+1} failed")
        
        # Benchmark requests
        logger.info("Starting benchmark requests...")
        results = []
        failed_requests = 0
        
        for i in range(num_requests):
            result = self.single_inference_request(image_data)
            if result is not None:
                results.append(result)
            else:
                failed_requests += 1
                
            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i + 1}/{num_requests} requests")
        
        if not results:
            raise Exception("All requests failed")
        
        # Calculate statistics
        total_times = [r['total_time_ms'] for r in results]
        server_times = [r['server_processing_ms'] for r in results]
        network_times = [r['network_overhead_ms'] for r in results]
        
        stats = {
            'total_requests': num_requests,
            'successful_requests': len(results),
            'failed_requests': failed_requests,
            'success_rate': len(results) / num_requests * 100,
            
            # Total time statistics
            'total_time_stats': {
                'mean_ms': statistics.mean(total_times),
                'median_ms': statistics.median(total_times),
                'std_ms': statistics.stdev(total_times) if len(total_times) > 1 else 0,
                'min_ms': min(total_times),
                'max_ms': max(total_times),
                'p95_ms': np.percentile(total_times, 95),
                'p99_ms': np.percentile(total_times, 99)
            },
            
            # Server processing time statistics
            'server_processing_stats': {
                'mean_ms': statistics.mean(server_times),
                'median_ms': statistics.median(server_times),
                'std_ms': statistics.stdev(server_times) if len(server_times) > 1 else 0,
                'min_ms': min(server_times),
                'max_ms': max(server_times)
            },
            
            # Network overhead statistics
            'network_overhead_stats': {
                'mean_ms': statistics.mean(network_times),
                'median_ms': statistics.median(network_times),
                'std_ms': statistics.stdev(network_times) if len(network_times) > 1 else 0,
                'min_ms': min(network_times),
                'max_ms': max(network_times)
            },
            
            # Raw data
            'raw_results': results
        }
        
        return stats
    
    def benchmark_concurrent_requests(self, concurrent_users: List[int] = [1, 2, 4, 8], 
                                    requests_per_user: int = 10) -> Dict:
        """Benchmark with concurrent requests to test server load"""
        logger.info("Starting concurrent request benchmark...")
        
        import concurrent.futures
        import threading
        
        results = {}
        image_data = self.create_test_image()
        
        for num_users in concurrent_users:
            logger.info(f"Testing {num_users} concurrent users")
            
            def user_requests():
                user_results = []
                for _ in range(requests_per_user):
                    result = self.single_inference_request(image_data)
                    if result:
                        user_results.append(result)
                return user_results
            
            start_time = time.perf_counter()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_users) as executor:
                futures = [executor.submit(user_requests) for _ in range(num_users)]
                all_results = []
                
                for future in concurrent.futures.as_completed(futures):
                    user_results = future.result()
                    all_results.extend(user_results)
            
            end_time = time.perf_counter()
            
            if all_results:
                total_times = [r['total_time_ms'] for r in all_results]
                results[num_users] = {
                    'concurrent_users': num_users,
                    'total_requests': num_users * requests_per_user,
                    'successful_requests': len(all_results),
                    'total_duration_s': end_time - start_time,
                    'requests_per_second': len(all_results) / (end_time - start_time),
                    'mean_response_time_ms': statistics.mean(total_times),
                    'p95_response_time_ms': np.percentile(total_times, 95)
                }
        
        return results
    
    def benchmark_different_image_sizes(self, sizes: List[Tuple[int, int]] = None) -> Dict:
        """Test with different image sizes to measure impact on network transfer"""
        if sizes is None:
            sizes = [(224, 224), (512, 512), (1024, 1024)]
        
        logger.info("Starting image size benchmark...")
        results = {}
        
        for size in sizes:
            logger.info(f"Testing image size: {size[0]}x{size[1]}")
            
            image_data = self.create_test_image(size)
            image_size_kb = len(image_data.encode('utf-8')) / 1024
            
            # Run small benchmark for this size
            size_results = []
            for _ in range(10):
                result = self.single_inference_request(image_data)
                if result:
                    size_results.append(result)
            
            if size_results:
                total_times = [r['total_time_ms'] for r in size_results]
                network_times = [r['network_overhead_ms'] for r in size_results]
                
                results[f"{size[0]}x{size[1]}"] = {
                    'image_size_pixels': size,
                    'image_size_kb': image_size_kb,
                    'mean_total_time_ms': statistics.mean(total_times),
                    'mean_network_overhead_ms': statistics.mean(network_times),
                    'samples': len(size_results)
                }
        
        return results
    
    def run_comprehensive_benchmark(self, config: Dict = None) -> Dict:
        """Run complete benchmark suite"""
        if config is None:
            config = {
                'latency_requests': 50,
                'concurrent_users': [1, 2, 4],
                'requests_per_user': 10,
                'test_image_sizes': True
            }
        
        logger.info("Starting comprehensive RTT benchmark...")
        
        # Get system info
        system_info = {
            'timestamp': datetime.now().isoformat(),
            'client_info': {
                'hostname': os.uname().nodename,
                'system': os.uname().sysname,
                'release': os.uname().release,
                'machine': os.uname().machine,
                'cpu_cores': psutil.cpu_count(),
                'total_ram_gb': psutil.virtual_memory().total / (1024**3)
            },
            'server_url': self.server_url
        }
        
        # Get server info
        try:
            server_response = requests.get(f"{self.server_url}/model_info", timeout=10)
            system_info['server_info'] = server_response.json()
        except Exception as e:
            logger.warning(f"Could not get server info: {e}")
            system_info['server_info'] = {}
        
        benchmark_results = {
            'system_info': system_info,
            'latency_benchmark': self.benchmark_latency(config['latency_requests']),
            'concurrent_benchmark': self.benchmark_concurrent_requests(
                config['concurrent_users'], 
                config['requests_per_user']
            )
        }
        
        if config.get('test_image_sizes', False):
            benchmark_results['image_size_benchmark'] = self.benchmark_different_image_sizes()
        
        return benchmark_results
    
    def save_results(self, results: Dict, filename: str = None) -> str:
        """Save benchmark results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rtt_benchmark_results_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {filepath}")
        return str(filepath)
    
    def generate_summary_report(self, results: Dict) -> str:
        """Generate human-readable summary report"""
        latency = results['latency_benchmark']
        concurrent = results['concurrent_benchmark']
        
        report = f"""
RTT Benchmark Summary Report
Generated: {results['system_info']['timestamp']}
Server: {results['system_info']['server_url']}

=== Latency Performance ===
Successful Requests: {latency['successful_requests']}/{latency['total_requests']} ({latency['success_rate']:.1f}%)
Mean Total Time: {latency['total_time_stats']['mean_ms']:.2f} ms
Median Total Time: {latency['total_time_stats']['median_ms']:.2f} ms
P95 Total Time: {latency['total_time_stats']['p95_ms']:.2f} ms
P99 Total Time: {latency['total_time_stats']['p99_ms']:.2f} ms

Mean Server Processing: {latency['server_processing_stats']['mean_ms']:.2f} ms
Mean Network Overhead: {latency['network_overhead_stats']['mean_ms']:.2f} ms
Network Overhead %: {(latency['network_overhead_stats']['mean_ms'] / latency['total_time_stats']['mean_ms'] * 100):.1f}%

=== Concurrent Load Performance ==="""
        
        for users, data in concurrent.items():
            report += f"""
{users} Users: {data['requests_per_second']:.2f} req/s, {data['mean_response_time_ms']:.2f}ms avg"""
        
        if 'image_size_benchmark' in results:
            report += "\n\n=== Image Size Impact ==="
            for size_key, data in results['image_size_benchmark'].items():
                report += f"""
{size_key}: {data['image_size_kb']:.1f}KB → {data['mean_total_time_ms']:.2f}ms total, {data['mean_network_overhead_ms']:.2f}ms network"""
        
        return report

def main():
    parser = argparse.ArgumentParser(description="RTT Benchmark Client for SqueezeNet Server")
    parser.add_argument("server_url", help="FastAPI server URL (e.g., http://192.168.1.100:8000)")
    parser.add_argument("--requests", type=int, default=50, help="Number of requests for latency test")
    parser.add_argument("--concurrent", nargs='+', type=int, default=[1, 2, 4], help="Concurrent user counts to test")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--image-sizes", action="store_true", help="Test different image sizes")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create client
    client = RTTBenchmarkClient(args.server_url, args.output)
    
    # Configure benchmark
    config = {
        'latency_requests': args.requests,
        'concurrent_users': args.concurrent,
        'requests_per_user': 10,
        'test_image_sizes': args.image_sizes
    }
    
    # Run benchmark
    try:
        results = client.run_comprehensive_benchmark(config)
        
        # Save results
        results_file = client.save_results(results)
        
        # Print summary
        summary = client.generate_summary_report(results)
        print(summary)
        
        # Save summary
        summary_file = Path(args.output) / f"rtt_benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        print(f"\nDetailed results: {results_file}")
        print(f"Summary report: {summary_file}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())