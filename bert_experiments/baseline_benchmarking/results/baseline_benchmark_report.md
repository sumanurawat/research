# BERT Baseline Performance Report

**Generated:** 2025-09-04 09:38:37

## System Configuration

- **Model:** bert-base-uncased
- **Device:** cpu
- **PyTorch Version:** 2.8.0
- **Transformers Version:** 4.56.0
- **CUDA Available:** False
- **Measurement Runs:** 50
- **Warmup Runs:** 10

## Performance Analysis

![Performance Analysis](performance_analysis.png)

<div align="center">
<img src="performance_analysis.png" alt="Performance Analysis" width="800"/>
</div>

## Performance Results

### Short Text Performance

| Batch Size | Mean Latency (ms) | P95 Latency (ms) | P99 Latency (ms) | Throughput (req/s) | GPU Memory (MB) |
|------------|-------------------|------------------|------------------|-------------------|------------------|
| 1 | 53.81 | 58.55 | 75.21 | 18.6 | 0.0 |
| 2 | 50.96 | 59.61 | 59.83 | 39.2 | 0.0 |
| 4 | 42.70 | 48.17 | 48.60 | 93.7 | 0.0 |
| 8 | 46.22 | 47.21 | 47.30 | 173.1 | 0.0 |
| 16 | 59.33 | 60.24 | 60.37 | 269.7 | 0.0 |
| 32 | 87.39 | 87.39 | 87.39 | 366.2 | 0.0 |

### Medium Text Performance

| Batch Size | Mean Latency (ms) | P95 Latency (ms) | P99 Latency (ms) | Throughput (req/s) | GPU Memory (MB) |
|------------|-------------------|------------------|------------------|-------------------|------------------|
| 1 | 48.98 | 56.08 | 61.70 | 20.4 | 0.0 |
| 2 | 66.81 | 93.60 | 106.79 | 29.9 | 0.0 |
| 4 | 83.51 | 86.40 | 86.90 | 47.9 | 0.0 |
| 8 | 135.81 | 137.29 | 137.29 | 58.9 | 0.0 |
| 16 | 247.36 | 253.95 | 254.71 | 64.7 | 0.0 |
| 32 | 704.23 | 704.23 | 704.23 | 45.4 | 0.0 |

### Long Text Performance

| Batch Size | Mean Latency (ms) | P95 Latency (ms) | P99 Latency (ms) | Throughput (req/s) | GPU Memory (MB) |
|------------|-------------------|------------------|------------------|-------------------|------------------|
| 1 | 81.38 | 94.55 | 102.32 | 12.3 | 0.0 |
| 2 | 100.39 | 111.55 | 114.76 | 19.9 | 0.0 |
| 4 | 164.10 | 169.05 | 169.11 | 24.4 | 0.0 |
| 8 | 293.24 | 301.84 | 303.07 | 27.3 | 0.0 |
| 16 | 517.35 | 520.09 | 520.33 | 30.9 | 0.0 |
| 32 | 1014.51 | 1014.51 | 1014.51 | 31.5 | 0.0 |

## Performance Summary

**Short Texts:**
- Best Latency: 42.70ms (batch size 4)
- Best Throughput: 366.2 req/s (batch size 32)
- Memory Range: 0.0 - 0.0 MB

**Medium Texts:**
- Best Latency: 48.98ms (batch size 1)
- Best Throughput: 64.7 req/s (batch size 16)
- Memory Range: 0.0 - 0.0 MB

**Long Texts:**
- Best Latency: 81.38ms (batch size 1)
- Best Throughput: 31.5 req/s (batch size 32)
- Memory Range: 0.0 - 0.0 MB

