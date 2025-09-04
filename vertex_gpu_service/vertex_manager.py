"""
Vertex AI GPU Manager for BERT Optimization Experiments

This manager handles:
- Submitting custom training jobs with T4 GPUs
- Automatic resource cleanup and cost tracking
- Results collection from Cloud Storage
- Multi-region failover for quota management
"""

import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

from google.cloud import aiplatform
from google.cloud import storage
from google.cloud.aiplatform import gapic
from google.api_core import exceptions
import yaml


class VertexAIManager:
    """Manages BERT experiments on Vertex AI with T4 GPUs."""
    
    def __init__(self, project_id: str, region: str = "us-central1"):
        """
        Initialize Vertex AI Manager
        
        Args:
            project_id: GCP project ID (phoenix-project-386)
            region: Primary region for jobs (us-central1)
        """
        self.project_id = project_id
        self.region = region
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=region)
        
        # Initialize clients
        self.job_client = gapic.JobServiceClient(
            client_options={"api_endpoint": f"{region}-aiplatform.googleapis.com"}
        )
        self.storage_client = storage.Client()
        
        # Available regions with T4 quota (from your approval)
        self.available_regions = {
            "us-central1": 2,    # Primary - 2 GPUs
            "us-east1": 1,
            "us-west1": 1,
            "europe-west2": 1,
            "europe-west4": 1,
            "asia-northeast1": 1,
            "asia-northeast3": 1,
            "asia-south1": 1,
            "asia-southeast1": 1
        }
        
        # Cost tracking
        self.t4_preemptible_rate = 0.105  # USD per hour per GPU
        self.machine_rates = {
            "n1-standard-4": 0.19,   # USD per hour
            "n1-standard-8": 0.38,   # USD per hour
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def get_cost_estimate(self, runtime_hours: float, gpu_count: int = 1, 
                         machine_type: str = "n1-standard-4") -> float:
        """
        Calculate estimated cost for job
        
        Args:
            runtime_hours: Expected runtime
            gpu_count: Number of GPUs
            machine_type: Machine type
            
        Returns:
            Estimated cost in USD
        """
        gpu_cost = runtime_hours * gpu_count * self.t4_preemptible_rate
        machine_cost = runtime_hours * self.machine_rates.get(machine_type, 0.19)
        return gpu_cost + machine_cost
    
    def create_job_spec(self, 
                       job_name: str,
                       container_uri: str,
                       experiment_type: str,
                       gpu_count: int = 1,
                       machine_type: str = "n1-standard-4",
                       max_runtime_hours: int = 2,
                       results_bucket: str = None,
                       **kwargs) -> Dict[str, Any]:
        """
        Create Vertex AI Custom Job specification
        
        Args:
            job_name: Unique name for the job
            container_uri: Container image URI
            experiment_type: Type of experiment (baseline, quantization, etc.)
            gpu_count: Number of T4 GPUs
            machine_type: Machine type
            max_runtime_hours: Maximum job runtime
            results_bucket: GCS bucket for results
            
        Returns:
            Job specification dictionary
        """
        if not results_bucket:
            results_bucket = f"{self.project_id}-bert-results"
        
        # Environment variables for the container
        env_vars = {
            "EXPERIMENT_TYPE": experiment_type,
            "RESULTS_BUCKET": results_bucket,
            "AIP_JOB_NAME": job_name,
            "PROJECT_ID": self.project_id,
            "REGION": self.region,
            "GPU_COUNT": str(gpu_count)
        }
        
        # Add any additional environment variables
        env_vars.update(kwargs.get('env_vars', {}))
        
        # Convert to Vertex AI format
        env_list = [{"name": k, "value": v} for k, v in env_vars.items()]
        
        job_spec = {
            "display_name": f"BERT-{experiment_type}-{job_name}",
            "job_spec": {
                "worker_pool_specs": [
                    {
                        "machine_spec": {
                            "machine_type": machine_type,
                            "accelerator_type": "NVIDIA_TESLA_T4",
                            "accelerator_count": gpu_count
                        },
                        "replica_count": 1,
                        "container_spec": {
                            "image_uri": container_uri,
                            "env": env_list
                        }
                    }
                ],
                "scheduling": {
                    "timeout": f"{max_runtime_hours * 3600}s",
                    "restart_job_on_worker_restart": False
                },
                "service_account": f"vertex-ai-runner@{self.project_id}.iam.gserviceaccount.com"
            }
        }
        
        return job_spec
    
    def submit_bert_experiment(self,
                              job_name: str,
                              experiment_type: str,
                              container_uri: str = None,
                              gpu_count: int = 1,
                              machine_type: str = "n1-standard-4",
                              max_runtime_hours: int = 2,
                              results_bucket: str = None,
                              region: str = None,
                              **kwargs) -> str:
        """
        Submit a BERT experiment as Vertex AI Custom Job
        
        Args:
            job_name: Unique name for the job
            experiment_type: Type of experiment
            container_uri: Container image URI
            gpu_count: Number of T4 GPUs (1-2 max)
            machine_type: Machine type
            max_runtime_hours: Maximum job runtime
            results_bucket: GCS bucket for results
            region: Region to submit job (defaults to self.region)
            
        Returns:
            Job resource name
        """
        if not region:
            region = self.region
            
        if not container_uri:
            container_uri = f"gcr.io/{self.project_id}/bert-{experiment_type}:latest"
            
        if not results_bucket:
            results_bucket = f"{self.project_id}-bert-results"
        
        # Estimate cost
        estimated_cost = self.get_cost_estimate(max_runtime_hours, gpu_count, machine_type)
        
        self.logger.info(f"🚀 Submitting BERT {experiment_type} experiment")
        self.logger.info(f"   Job: {job_name}")
        self.logger.info(f"   Region: {region}")
        self.logger.info(f"   GPUs: {gpu_count}x T4")
        self.logger.info(f"   Estimated cost: ${estimated_cost:.2f}")
        
        # Create job specification
        job_spec = self.create_job_spec(
            job_name=job_name,
            container_uri=container_uri,
            experiment_type=experiment_type,
            gpu_count=gpu_count,
            machine_type=machine_type,
            max_runtime_hours=max_runtime_hours,
            results_bucket=results_bucket,
            **kwargs
        )
        
        try:
            # Submit job
            parent = f"projects/{self.project_id}/locations/{region}"
            
            # Use the job client to create the job
            job_client = gapic.JobServiceClient(
                client_options={"api_endpoint": f"{region}-aiplatform.googleapis.com"}
            )
            
            operation = job_client.create_custom_job(
                parent=parent,
                custom_job=job_spec
            )
            
            job_resource_name = operation.name
            
            self.logger.info(f"✅ Job submitted successfully!")
            self.logger.info(f"   Resource name: {job_resource_name}")
            
            return job_resource_name
            
        except exceptions.GoogleAPICallError as e:
            self.logger.error(f"❌ Failed to submit job: {e}")
            
            # Try fallback region if quota exceeded
            if "quota" in str(e).lower() and region == self.region:
                self.logger.info("🔄 Trying fallback region...")
                fallback_regions = [r for r in self.available_regions.keys() if r != region]
                
                for fallback_region in fallback_regions:
                    try:
                        return self.submit_bert_experiment(
                            job_name=job_name,
                            experiment_type=experiment_type,
                            container_uri=container_uri,
                            gpu_count=gpu_count,
                            machine_type=machine_type,
                            max_runtime_hours=max_runtime_hours,
                            results_bucket=results_bucket,
                            region=fallback_region,
                            **kwargs
                        )
                    except exceptions.GoogleAPICallError:
                        continue
                        
            raise e
    
    def get_job_status(self, job_resource_name: str) -> Dict[str, Any]:
        """
        Get current status of a Vertex AI job
        
        Args:
            job_resource_name: Full resource name of the job
            
        Returns:
            Job status information
        """
        try:
            # Extract region from resource name
            region = job_resource_name.split('/')[3]
            
            job_client = gapic.JobServiceClient(
                client_options={"api_endpoint": f"{region}-aiplatform.googleapis.com"}
            )
            
            job = job_client.get_custom_job(name=job_resource_name)
            
            status_info = {
                "name": job.name,
                "display_name": job.display_name,
                "state": job.state.name,
                "create_time": job.create_time,
                "start_time": job.start_time,
                "end_time": job.end_time,
                "error": job.error.message if job.error else None
            }
            
            return status_info
            
        except exceptions.GoogleAPICallError as e:
            self.logger.error(f"❌ Failed to get job status: {e}")
            return {"error": str(e)}
    
    def wait_for_completion(self, 
                           job_resource_name: str, 
                           timeout_hours: int = 4,
                           poll_interval_seconds: int = 30) -> Dict[str, Any]:
        """
        Wait for job completion and collect results
        
        Args:
            job_resource_name: Full resource name of the job
            timeout_hours: Maximum time to wait
            poll_interval_seconds: How often to check status
            
        Returns:
            Job results and metadata
        """
        self.logger.info(f"⏳ Waiting for job completion: {job_resource_name}")
        
        start_time = datetime.now()
        timeout = timedelta(hours=timeout_hours)
        
        while datetime.now() - start_time < timeout:
            status = self.get_job_status(job_resource_name)
            
            if "error" in status:
                return {"success": False, "error": status["error"]}
            
            state = status.get("state", "UNKNOWN")
            
            self.logger.info(f"   Status: {state}")
            
            if state == "JOB_STATE_SUCCEEDED":
                self.logger.info("✅ Job completed successfully!")
                
                # Calculate actual runtime and cost
                if status.get("start_time") and status.get("end_time"):
                    runtime = status["end_time"] - status["start_time"]
                    runtime_hours = runtime.total_seconds() / 3600
                    actual_cost = self.get_cost_estimate(runtime_hours)
                else:
                    runtime_hours = 0
                    actual_cost = 0
                
                return {
                    "success": True,
                    "status": status,
                    "runtime_hours": runtime_hours,
                    "actual_cost": actual_cost
                }
                
            elif state in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED"]:
                error_msg = status.get("error", f"Job {state}")
                self.logger.error(f"❌ Job failed: {error_msg}")
                
                return {
                    "success": False,
                    "status": status,
                    "error": error_msg
                }
            
            # Still running, wait and check again
            time.sleep(poll_interval_seconds)
        
        # Timeout reached
        self.logger.warning(f"⏰ Timeout reached after {timeout_hours} hours")
        return {
            "success": False,
            "error": f"Timeout after {timeout_hours} hours",
            "status": status
        }
    
    def list_jobs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List recent Vertex AI jobs
        
        Args:
            limit: Maximum number of jobs to return
            
        Returns:
            List of job information
        """
        try:
            parent = f"projects/{self.project_id}/locations/{self.region}"
            
            request = gapic.ListCustomJobsRequest(
                parent=parent,
                page_size=limit
            )
            
            jobs = []
            for job in self.job_client.list_custom_jobs(request=request):
                job_info = {
                    "name": job.name,
                    "display_name": job.display_name,
                    "state": job.state.name,
                    "create_time": job.create_time,
                    "start_time": job.start_time,
                    "end_time": job.end_time
                }
                jobs.append(job_info)
            
            return jobs
            
        except exceptions.GoogleAPICallError as e:
            self.logger.error(f"❌ Failed to list jobs: {e}")
            return []
    
    def cancel_job(self, job_resource_name: str) -> bool:
        """
        Cancel a running Vertex AI job
        
        Args:
            job_resource_name: Full resource name of the job
            
        Returns:
            True if cancellation was successful
        """
        try:
            # Extract region from resource name
            region = job_resource_name.split('/')[3]
            
            job_client = gapic.JobServiceClient(
                client_options={"api_endpoint": f"{region}-aiplatform.googleapis.com"}
            )
            
            job_client.cancel_custom_job(name=job_resource_name)
            
            self.logger.info(f"🛑 Job cancellation requested: {job_resource_name}")
            return True
            
        except exceptions.GoogleAPICallError as e:
            self.logger.error(f"❌ Failed to cancel job: {e}")
            return False
    
    def run_experiment_pipeline(self,
                               experiment_name: str,
                               experiment_config: Dict[str, Any],
                               wait_for_completion: bool = True) -> Dict[str, Any]:
        """
        High-level method to run complete BERT experiment
        
        Args:
            experiment_name: Name of experiment (e.g., "week2-quantization")
            experiment_config: Experiment configuration
            wait_for_completion: Whether to wait for job completion
            
        Returns:
            Complete experiment results
        """
        # Generate unique job name
        timestamp = int(time.time())
        job_name = f"{experiment_name}-{timestamp}"
        
        # Submit job
        job_resource_name = self.submit_bert_experiment(
            job_name=job_name,
            **experiment_config
        )
        
        if not wait_for_completion:
            return {
                "job_name": job_name,
                "job_resource_name": job_resource_name,
                "status": "submitted"
            }
        
        # Wait for completion
        results = self.wait_for_completion(job_resource_name)
        results["job_name"] = job_name
        results["job_resource_name"] = job_resource_name
        
        return results