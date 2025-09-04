#!/usr/bin/env python3
"""
Vertex AI Job Submission Interface

Usage:
    python vertex_gpu_service/job_submitter.py --experiment baseline --dry-run
    python vertex_gpu_service/job_submitter.py --experiment quantization --wait
"""

import argparse
import json
import sys
import time
import yaml
from pathlib import Path
from datetime import datetime

from .vertex_manager import VertexAIManager


def load_config(config_path: str = None) -> dict:
    """Load Vertex AI configuration."""
    if not config_path:
        # Try to find config in service folder
        service_dir = Path(__file__).parent
        config_path = service_dir / "config.yaml"
    
    config_file = Path(config_path)
    
    if not config_file.exists():
        print(f"❌ Configuration file not found: {config_file}")
        sys.exit(1)
    
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def print_job_templates(config: dict):
    """Print available job templates."""
    print("\\n📋 Available BERT Experiments:")
    print("=" * 50)
    
    for template_name, template_config in config['job_templates'].items():
        print(f"\\n{template_name.upper()}:")
        print(f"   Description: {template_config.get('description', 'No description')}")
        print(f"   Runtime: {template_config['max_runtime_hours']} hours")
        print(f"   Cost: ${template_config['estimated_cost']:.2f}")
        print(f"   Machine: {template_config['machine_type']}")
        print(f"   GPUs: {template_config['gpu_count']}x T4")


def validate_experiment_type(experiment_type: str, config: dict) -> bool:
    """Validate that experiment type exists in configuration."""
    available_experiments = list(config['job_templates'].keys())
    
    # Remove 'bert_' prefix if provided
    if experiment_type.startswith('bert_'):
        experiment_type = experiment_type[5:]
    
    # Check if experiment exists
    template_key = f"bert_{experiment_type}"
    if template_key not in available_experiments:
        print(f"❌ Unknown experiment type: {experiment_type}")
        print(f"Available experiments: {[exp.replace('bert_', '') for exp in available_experiments]}")
        return False
    
    return True


def check_prerequisites() -> bool:
    """Check if prerequisites are met."""
    print("🔍 Checking prerequisites...")
    
    # Check if authenticated with gcloud
    try:
        import subprocess
        result = subprocess.run(['gcloud', 'auth', 'list', '--filter=status:ACTIVE'], 
                              capture_output=True, text=True, check=True)
        
        if 'ACTIVE' in result.stdout:
            print("✅ Authenticated with Google Cloud")
        else:
            print("❌ Not authenticated with Google Cloud")
            print("   Run: gcloud auth login")
            return False
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Google Cloud CLI not found or authentication failed")
        return False
    
    print("✅ Prerequisites check passed")
    return True


def submit_experiment(experiment_type: str, config: dict, **kwargs) -> dict:
    """Submit BERT experiment to Vertex AI."""
    
    # Get experiment template
    template_key = f"bert_{experiment_type}"
    template = config['job_templates'][template_key]
    
    # Override template values with kwargs
    experiment_config = {
        "experiment_type": experiment_type,
        "gpu_count": kwargs.get('gpu_count', template['gpu_count']),
        "machine_type": kwargs.get('machine_type', template['machine_type']),
        "max_runtime_hours": kwargs.get('max_runtime', template['max_runtime_hours']),
        "container_uri": kwargs.get('container_uri', template['container_uri'])
    }
    
    # Initialize manager
    region = kwargs.get('region', config['defaults']['region'])
    project_id = kwargs.get('project_id', config['defaults']['project_id'])
    
    manager = VertexAIManager(project_id=project_id, region=region)
    
    # Calculate cost estimate
    cost_estimate = manager.get_cost_estimate(
        runtime_hours=experiment_config["max_runtime_hours"],
        gpu_count=experiment_config["gpu_count"],
        machine_type=experiment_config["machine_type"]
    )
    
    # Print job summary
    print(f"\\n📊 Job Summary:")
    print(f"   Experiment: {experiment_type}")
    print(f"   Description: {template.get('description', 'No description')}")
    print(f"   Region: {region}")
    print(f"   Machine Type: {experiment_config['machine_type']}")
    print(f"   GPUs: {experiment_config['gpu_count']}x T4")
    print(f"   Max Runtime: {experiment_config['max_runtime_hours']} hours")
    print(f"   Container: {experiment_config['container_uri']}")
    print(f"   Estimated Cost: ${cost_estimate:.2f}")
    
    # Check budget
    daily_budget = config['cost_controls']['daily_budget']
    if cost_estimate > daily_budget:
        print(f"⚠️  Warning: Estimated cost (${cost_estimate:.2f}) exceeds daily budget (${daily_budget:.2f})")
        
        if not kwargs.get('dry_run', False):
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                print("❌ Job submission cancelled")
                return {"success": False, "error": "Cancelled by user"}
    
    if kwargs.get('dry_run', False):
        print(f"\\n🔍 DRY RUN - Would submit job with above configuration")
        return {"success": True, "dry_run": True, "cost_estimate": cost_estimate}
    
    # Generate unique job name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"{experiment_type}-{timestamp}"
    
    try:
        # Submit job
        print(f"\\n🚀 Submitting job: {job_name}")
        job_resource_name = manager.submit_bert_experiment(
            job_name=job_name,
            region=region,
            **experiment_config
        )
        
        print(f"✅ Job submitted successfully!")
        print(f"   Job Name: {job_name}")
        print(f"   Resource Name: {job_resource_name}")
        print(f"   Region: {region}")
        
        result = {
            "success": True,
            "job_name": job_name,
            "job_resource_name": job_resource_name,
            "region": region,
            "estimated_cost": cost_estimate,
            "config": experiment_config
        }
        
        # Wait for completion if requested
        if kwargs.get('wait', False):
            print(f"\\n⏳ Waiting for job completion...")
            completion_results = manager.wait_for_completion(
                job_resource_name, 
                timeout_hours=experiment_config['max_runtime_hours'] + 1
            )
            result.update(completion_results)
        
        return result
        
    except Exception as e:
        print(f"\\n❌ Failed to submit job: {e}")
        return {"success": False, "error": str(e)}


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Submit BERT experiment to Vertex AI")
    parser.add_argument("--experiment", help="Type of experiment to run")
    parser.add_argument("--gpu-count", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--region", help="GCP region")
    parser.add_argument("--project-id", default="phoenix-project-386", help="GCP project ID")
    parser.add_argument("--wait", action="store_true", help="Wait for job completion")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be submitted")
    parser.add_argument("--list-experiments", action="store_true", help="List available experiments")
    parser.add_argument("--max-runtime", type=int, help="Override max runtime hours")
    parser.add_argument("--machine-type", help="Override machine type")
    parser.add_argument("--container-uri", help="Override container URI")
    
    args = parser.parse_args()
    
    print("🚀 Vertex AI BERT Experiment Submission Tool")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    
    # Handle list experiments
    if args.list_experiments:
        print_job_templates(config)
        return
    
    if not args.experiment:
        print("❌ No experiment specified. Use --list-experiments to see available options.")
        return
    
    # Validate experiment type
    if not validate_experiment_type(args.experiment, config):
        return
    
    # Check prerequisites
    if not check_prerequisites():
        return
    
    # Submit experiment
    result = submit_experiment(
        experiment_type=args.experiment,
        config=config,
        gpu_count=args.gpu_count,
        region=args.region,
        project_id=args.project_id,
        wait=args.wait,
        dry_run=args.dry_run,
        max_runtime=args.max_runtime,
        machine_type=args.machine_type,
        container_uri=args.container_uri
    )
    
    if result['success']:
        if result.get('dry_run'):
            print("\\n🔍 Dry run completed successfully")
        else:
            print("\\n✅ Job submission completed")
    else:
        print(f"\\n❌ Job submission failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()