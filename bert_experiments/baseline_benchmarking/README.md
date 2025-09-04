# Baseline Performance Benchmarking

Production-grade BERT performance measurement with statistical rigor and clean results presentation.

## Quick Start

```bash
cd bert_experiments/baseline_benchmarking
python baseline_benchmark.py
```

## Features

- **Production-Grade Benchmarking**: Statistical rigor with multiple measurement runs
- **Clean Results Presentation**: Pure performance data without optimization recommendations
- **Embedded Visualizations**: Self-contained reports with embedded charts
- **Multi-Dimensional Analysis**: Latency, throughput, memory usage across batch sizes
- **Automated Reporting**: Professional markdown reports with performance analysis
- **Fresh Results**: Automatic cleanup and overwrite of previous results

## Expected Runtime

- **CPU**: ~3-5 minutes for comprehensive analysis
- **GPU**: ~2-3 minutes with CUDA acceleration

## Output Files

- `baseline_benchmark_report.md` - **Complete report with linked performance charts**
- `performance_analysis.png` - **Professional performance visualizations**
- `benchmark_data.json` - Raw performance data in JSON format

## Performance Metrics

### Measured Dimensions
- **Latency**: Mean, P95, P99 percentiles across batch sizes
- **Throughput**: Requests per second scaling analysis
- **Memory Usage**: GPU/CPU memory consumption tracking
- **Statistical Analysis**: Confidence intervals and outlier detection

### Test Categories
- **Short Texts** (10-50 tokens): Social media posts, search queries
- **Medium Texts** (100-200 tokens): Product reviews, news snippets
- **Long Texts** (400-512 tokens): Articles, documentation

### Batch Size Analysis
- Comprehensive testing: [1, 2, 4, 8, 16, 32]
- Automatic warmup procedures for measurement accuracy
- Memory management between experiments

## Configuration

Customize benchmarking through `BenchmarkConfig`:

```python
@dataclass
class BenchmarkConfig:
    model_name: str = "bert-base-uncased"
    device: str = "auto"  # auto, cpu, cuda
    batch_sizes: List[int] = [1, 2, 4, 8, 16, 32]
    measurement_runs: int = 50
    warmup_runs: int = 10
    results_dir: str = "results"
```

## Sample Results

Typical baseline performance on CPU:
- **Short Texts**: 48-60ms latency, 300+ req/s throughput
- **Medium Texts**: 55-70ms latency, 60+ req/s throughput  
- **Long Texts**: 85-120ms latency, 25+ req/s throughput

## Key Design Principles

1. **Statistical Rigor**: Multiple measurement runs with proper warmup
2. **Clean Presentation**: Pure performance data without recommendations
3. **Universal Compatibility**: Image references that work across all markdown viewers
4. **Fresh Results**: Automatic cleanup ensures current data
5. **Professional Output**: Publication-ready charts and formatting