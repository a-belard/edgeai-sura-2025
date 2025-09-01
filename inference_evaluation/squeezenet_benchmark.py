#!/usr/bin/env python3
"""
SqueezeNet Automated Benchmarking Script
========================================

Comprehensive performance evaluation for edge AI research.
Focus on latency, delay-tolerance, and inference performance.

Usage:
    python squeezenet_benchmark.py [--runs 100] [--output results/]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


class SqueezeNetBenchmarker:
    """Comprehensive SqueezeNet benchmarking for edge AI evaluation."""
    
    def __init__(self, use_quantized=False):
        self.model = None
        self.device = None
        self.test_input = None
        self.model_size_mb = 0
        self.use_quantized = use_quantized
        self.transform = None
        
    def setup(self):
        """Initialize model and test data."""
        print("Setting up SqueezeNet benchmarker...")
        
        # Set device (prioritize Apple Silicon GPU if available)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("✓ Using Apple Silicon GPU (MPS)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("✓ Using CUDA GPU")
        else:
            self.device = torch.device("cpu")
            print("✓ Using CPU")
        
        # Load model
        print("Loading SqueezeNet model...")
        self.model = models.squeezenet1_1(pretrained=True)
        self.model.eval()
        
        # Apply quantization if requested
        if self.use_quantized:
            print("Applying dynamic quantization...")
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
            )
        
        self.model.to(self.device)
        
        # Calculate model size
        self.model_size_mb = self._calculate_model_size()
        
        # Setup image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Prepare test data
        self._prepare_test_data()
        
        print(f"✓ Model loaded: {self.model_size_mb:.2f} MB")
        print(f"✓ Input shape: [1, 3, 224, 224]")
        print(f"✓ Quantized: {self.use_quantized}")
        print(f"✓ Device: {self.device}")
        
    def _calculate_model_size(self):
        """Calculate model size in MB."""
        param_size = 0
        buffer_size = 0
        
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        model_size = (param_size + buffer_size) / (1024 * 1024)
        return model_size
    
    def _prepare_test_data(self):
        """Prepare test image data."""
        # Create a synthetic test image
        test_image = torch.randn(1, 3, 224, 224).to(self.device)
        self.test_input = test_image
    
    def benchmark_core_performance(self, num_runs=100, warmup_runs=10):
        """Core inference performance benchmark."""
        print(f"Running core performance benchmark ({warmup_runs} warmup + {num_runs} runs)...")
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self.model(self.test_input)
        
        # Synchronize if using GPU
        if self.device.type in ['cuda', 'mps']:
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
        
        # Benchmark
        inference_times = []
        cpu_usage = []
        memory_usage = []
        gpu_memory_usage = []
        process = psutil.Process()
        
        for _ in tqdm(range(num_runs), desc="Benchmarking"):
            # Measure resources
            cpu_before = process.cpu_percent()
            mem_before = process.memory_info().rss / (1024 * 1024)
            
            # GPU memory if available
            gpu_mem_before = 0
            if self.device.type == 'cuda':
                gpu_mem_before = torch.cuda.memory_allocated() / (1024 * 1024)
            
            # Run inference
            start_time = time.perf_counter()
            with torch.no_grad():
                output = self.model(self.test_input)
            
            # Synchronize if using GPU
            if self.device.type in ['cuda', 'mps']:
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
            
            end_time = time.perf_counter()
            
            # Record measurements
            inference_times.append((end_time - start_time) * 1000)  # ms
            cpu_usage.append(process.cpu_percent())
            memory_usage.append(process.memory_info().rss / (1024 * 1024))  # MB
            
            # GPU memory if available
            gpu_mem_after = 0
            if self.device.type == 'cuda':
                gpu_mem_after = torch.cuda.memory_allocated() / (1024 * 1024)
            gpu_memory_usage.append(gpu_mem_after)
        
        return {
            'inference_times': inference_times,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'gpu_memory_usage': gpu_memory_usage,
            'statistics': {
                'mean_inference_time': np.mean(inference_times),
                'std_inference_time': np.std(inference_times),
                'min_inference_time': np.min(inference_times),
                'max_inference_time': np.max(inference_times),
                'p95_inference_time': np.percentile(inference_times, 95),
                'p99_inference_time': np.percentile(inference_times, 99),
                'mean_memory_usage': np.mean(memory_usage),
                'peak_memory_usage': np.max(memory_usage),
                'mean_cpu_usage': np.mean(cpu_usage),
                'peak_cpu_usage': np.max(cpu_usage),
                'mean_gpu_memory': np.mean(gpu_memory_usage) if gpu_memory_usage else 0,
                'peak_gpu_memory': np.max(gpu_memory_usage) if gpu_memory_usage else 0
            }
        }
    
    def benchmark_latency_scenarios(self, network_delays=[0, 50, 100, 200, 500, 1000]):
        """Benchmark with simulated network latency."""
        print("Testing delay-tolerant scenarios...")
        
        latency_results = {}
        
        for delay in network_delays:
            times = []
            
            for _ in tqdm(range(20), desc=f"Delay {delay}ms"):
                # Simulate network delay before processing
                start_total = time.perf_counter()
                time.sleep(delay / 1000.0)  # Simulate network delay
                
                # Actual inference
                start_inference = time.perf_counter()
                with torch.no_grad():
                    output = self.model(self.test_input)
                
                # Synchronize if using GPU
                if self.device.type in ['cuda', 'mps']:
                    torch.cuda.synchronize() if self.device.type == 'cuda' else None
                
                end_inference = time.perf_counter()
                
                # Simulate network delay after processing
                time.sleep(delay / 1000.0)
                end_total = time.perf_counter()
                
                times.append({
                    'inference_time': (end_inference - start_inference) * 1000,
                    'total_time': (end_total - start_total) * 1000,
                    'network_overhead': (end_total - start_total) * 1000 - (end_inference - start_inference) * 1000
                })
            
            inference_times = [t['inference_time'] for t in times]
            total_times = [t['total_time'] for t in times]
            network_overhead = [t['network_overhead'] for t in times]
            
            latency_results[delay] = {
                'mean_inference': np.mean(inference_times),
                'mean_total': np.mean(total_times),
                'mean_network_overhead': np.mean(network_overhead),
                'efficiency_ratio': np.mean(inference_times) / np.mean(total_times) * 100
            }
        
        return latency_results
    
    def benchmark_batch_processing(self, batch_sizes=[1, 5, 10, 20, 50]):
        """Test batch processing performance."""
        print("Testing batch processing performance...")
        
        batch_results = {}
        
        for batch_size in batch_sizes:
            times = []
            memory_usage = []
            process = psutil.Process()
            
            # Create batch input
            batch_input = torch.randn(batch_size, 3, 224, 224).to(self.device)
            
            for _ in range(10):
                start_time = time.perf_counter()
                mem_before = process.memory_info().rss / (1024 * 1024)
                
                # Process batch
                with torch.no_grad():
                    output = self.model(batch_input)
                
                # Synchronize if using GPU
                if self.device.type in ['cuda', 'mps']:
                    torch.cuda.synchronize() if self.device.type == 'cuda' else None
                
                end_time = time.perf_counter()
                mem_after = process.memory_info().rss / (1024 * 1024)
                
                batch_time = (end_time - start_time) * 1000
                times.append(batch_time)
                memory_usage.append(mem_after)
            
            avg_time = np.mean(times)
            avg_time_per_image = avg_time / batch_size
            throughput = batch_size / (avg_time / 1000)
            
            batch_results[batch_size] = {
                'total_time': avg_time,
                'time_per_image': avg_time_per_image,
                'throughput': throughput,
                'memory_usage': np.mean(memory_usage)
            }
        
        return batch_results
    
    def benchmark_cold_warm_start(self, num_trials=10):
        """Compare cold start vs warm start performance."""
        print("Testing cold start vs warm start...")
        
        cold_starts = []
        warm_starts = []
        
        for _ in range(num_trials):
            # Cold start: reload model
            model_cold = models.squeezenet1_1(pretrained=True)
            model_cold.eval()
            
            if self.use_quantized:
                model_cold = torch.quantization.quantize_dynamic(
                    model_cold, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
                )
            
            model_cold.to(self.device)
            
            # Cold start inference
            start_time = time.perf_counter()
            with torch.no_grad():
                output = model_cold(self.test_input)
            
            if self.device.type in ['cuda', 'mps']:
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
            
            cold_time = (time.perf_counter() - start_time) * 1000
            cold_starts.append(cold_time)
            
            # Warm start inference
            start_time = time.perf_counter()
            with torch.no_grad():
                output = model_cold(self.test_input)
            
            if self.device.type in ['cuda', 'mps']:
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
            
            warm_time = (time.perf_counter() - start_time) * 1000
            warm_starts.append(warm_time)
        
        return {
            'cold_starts': cold_starts,
            'warm_starts': warm_starts,
            'statistics': {
                'cold_start_mean': np.mean(cold_starts),
                'cold_start_std': np.std(cold_starts),
                'warm_start_mean': np.mean(warm_starts),
                'warm_start_std': np.std(warm_starts),
                'startup_overhead': np.mean(cold_starts) - np.mean(warm_starts)
            }
        }
    
    def benchmark_precision_modes(self):
        """Test different precision modes if available."""
        print("Testing precision modes...")
        
        precision_results = {}
        
        # Test FP32 (default)
        times_fp32 = []
        for _ in range(50):
            start_time = time.perf_counter()
            with torch.no_grad():
                output = self.model(self.test_input)
            
            if self.device.type in ['cuda', 'mps']:
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
            
            times_fp32.append((time.perf_counter() - start_time) * 1000)
        
        precision_results['fp32'] = {
            'mean_time': np.mean(times_fp32),
            'std_time': np.std(times_fp32)
        }
        
        # Test FP16 if GPU available
        if self.device.type == 'cuda':
            try:
                model_fp16 = self.model.half()
                test_input_fp16 = self.test_input.half()
                
                times_fp16 = []
                for _ in range(50):
                    start_time = time.perf_counter()
                    with torch.no_grad():
                        output = model_fp16(test_input_fp16)
                    torch.cuda.synchronize()
                    times_fp16.append((time.perf_counter() - start_time) * 1000)
                
                precision_results['fp16'] = {
                    'mean_time': np.mean(times_fp16),
                    'std_time': np.std(times_fp16)
                }
                
                # Restore original model
                self.model = self.model.float()
                
            except Exception as e:
                print(f"FP16 testing failed: {e}")
        
        return precision_results
    
    def run_comprehensive_benchmark(self, num_runs=100):
        """Run all benchmarks and compile results."""
        print("=== Starting Comprehensive SqueezeNet Benchmark ===")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'device_info': self._get_device_info(),
            'model_info': {
                'name': 'SqueezeNet',
                'type': 'Lightweight CNN',
                'size_mb': self.model_size_mb,
                'input_shape': [1, 3, 224, 224],
                'output_shape': [1, 1000],
                'quantized': self.use_quantized,
                'device': str(self.device)
            }
        }
        
        # Run all benchmarks
        results['core_performance'] = self.benchmark_core_performance(num_runs)
        results['latency_analysis'] = self.benchmark_latency_scenarios()
        results['batch_processing'] = self.benchmark_batch_processing()
        results['cold_warm_analysis'] = self.benchmark_cold_warm_start()
        results['precision_analysis'] = self.benchmark_precision_modes()
        
        return results
    
    def _get_device_info(self):
        """Get device information."""
        import platform
        
        device_info = {
            'system': platform.system(),
            'release': platform.release(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'cpu_cores': psutil.cpu_count(),
            'total_ram_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'device_type': str(self.device)
        }
        
        # Add GPU info if available
        if self.device.type == 'cuda':
            device_info['cuda_version'] = torch.version.cuda
            device_info['gpu_name'] = torch.cuda.get_device_name()
            device_info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        elif self.device.type == 'mps':
            device_info['mps_available'] = True
        
        return device_info
    
    def save_results(self, results, output_dir="results"):
        """Save benchmark results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            else:
                return obj
        
        # Save comprehensive JSON
        json_file = output_path / f"squeezenet_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(convert_numpy_types(results), f, indent=2)
        
        # Save summary CSV
        core_stats = results['core_performance']['statistics']
        cold_warm_stats = results['cold_warm_analysis']['statistics']
        
        summary_data = {
            'Model': ['SqueezeNet'],
            'Device_Tier': ['Laptop (Tier 2)'],
            'Device_Type': [results['model_info']['device']],
            'Model_Size_MB': [results['model_info']['size_mb']],
            'Quantized': [results['model_info']['quantized']],
            'Mean_Inference_Time_ms': [core_stats['mean_inference_time']],
            'Std_Inference_Time_ms': [core_stats['std_inference_time']],
            'P95_Inference_Time_ms': [core_stats['p95_inference_time']],
            'Mean_Memory_Usage_MB': [core_stats['mean_memory_usage']],
            'Peak_Memory_Usage_MB': [core_stats['peak_memory_usage']],
            'Mean_CPU_Usage_Percent': [core_stats['mean_cpu_usage']],
            'GPU_Memory_MB': [core_stats['peak_gpu_memory']],
            'Cold_Start_Overhead_ms': [cold_warm_stats['startup_overhead']],
            'Max_Throughput_imgs_per_sec': [max([results['batch_processing'][bs]['throughput'] 
                                               for bs in results['batch_processing'].keys()])],
            'Efficiency_No_Network_Percent': [results['latency_analysis'][0]['efficiency_ratio']],
            'Efficiency_500ms_Network_Percent': [results['latency_analysis'][500]['efficiency_ratio']]
        }
        
        summary_df = pd.DataFrame(summary_data)
        csv_file = output_path / f"squeezenet_summary_{timestamp}.csv"
        summary_df.to_csv(csv_file, index=False)
        
        print(f"\n=== Results Saved ===")
        print(f"JSON: {json_file}")
        print(f"CSV:  {csv_file}")
        
        return json_file, csv_file
    
    def print_summary(self, results):
        """Print benchmark summary."""
        core_stats = results['core_performance']['statistics']
        cold_warm_stats = results['cold_warm_analysis']['statistics']
        
        print("\n=== SQUEEZENET BENCHMARK SUMMARY ===")
        print(f"Device: {results['device_info']['system']} ({results['device_info']['cpu_cores']} cores)")
        print(f"Compute Device: {results['model_info']['device']}")
        print(f"Model Size: {results['model_info']['size_mb']:.2f} MB")
        print(f"Quantized: {results['model_info']['quantized']}")
        print()
        print("PERFORMANCE METRICS:")
        print(f"  Mean Inference Time: {core_stats['mean_inference_time']:.2f} ± {core_stats['std_inference_time']:.2f} ms")
        print(f"  95th Percentile: {core_stats['p95_inference_time']:.2f} ms")
        print(f"  Memory Usage: {core_stats['mean_memory_usage']:.1f} MB (peak: {core_stats['peak_memory_usage']:.1f} MB)")
        print(f"  CPU Usage: {core_stats['mean_cpu_usage']:.1f}% (peak: {core_stats['peak_cpu_usage']:.1f}%)")
        if core_stats['peak_gpu_memory'] > 0:
            print(f"  GPU Memory: {core_stats['peak_gpu_memory']:.1f} MB")
        print()
        print("DELAY-TOLERANT ANALYSIS:")
        print(f"  Cold Start Overhead: {cold_warm_stats['startup_overhead']:.2f} ms")
        print(f"  Efficiency (no network): {results['latency_analysis'][0]['efficiency_ratio']:.1f}%")
        print(f"  Efficiency (500ms network): {results['latency_analysis'][500]['efficiency_ratio']:.1f}%")
        print(f"  Max Throughput: {max([results['batch_processing'][bs]['throughput'] for bs in results['batch_processing'].keys()]):.1f} images/sec")
        
        # Precision comparison
        if 'precision_analysis' in results and 'fp16' in results['precision_analysis']:
            fp32_time = results['precision_analysis']['fp32']['mean_time']
            fp16_time = results['precision_analysis']['fp16']['mean_time']
            speedup = fp32_time / fp16_time
            print(f"  FP16 Speedup: {speedup:.2f}x")


def main():
    """Main benchmarking function."""
    parser = argparse.ArgumentParser(description='SqueezeNet Comprehensive Benchmark')
    parser.add_argument('--runs', type=int, default=100, help='Number of benchmark runs')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--quantized', action='store_true', help='Use quantized model')
    
    args = parser.parse_args()
    
    # Initialize benchmarker
    benchmarker = SqueezeNetBenchmarker(use_quantized=args.quantized)
    benchmarker.setup()
    
    # Run comprehensive benchmark
    results = benchmarker.run_comprehensive_benchmark(args.runs)
    
    # Save and display results
    benchmarker.save_results(results, args.output)
    benchmarker.print_summary(results)
    
    print("\n=== Benchmark Complete ===")


if __name__ == "__main__":
    main()
