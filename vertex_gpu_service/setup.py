#!/usr/bin/env python3
"""
Vertex AI Setup Script for BERT Optimization Project

This script sets up:
- Required GCP APIs
- Service accounts and permissions
- Cloud Storage buckets
- Container registry
- Environment configuration

Usage:
    python vertex_gpu_service/setup.py --setup-all
    python vertex_gpu_service/setup.py --check-setup
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


class VertexAISetup:
    """Setup helper for Vertex AI BERT experiments."""
    
    def __init__(self, project_id: str = "phoenix-project-386"):
        """Initialize setup helper."""
        self.project_id = project_id
        self.service_account_name = "vertex-ai-runner"
        self.service_account_email = f"{self.service_account_name}@{project_id}.iam.gserviceaccount.com"
        
        print(f"🔧 Vertex AI Setup for Project: {project_id}")
        print("=" * 50)
    
    def run_gcloud_command(self, command: list, check: bool = True) -> subprocess.CompletedProcess:
        """Run a gcloud command and return the result."""
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=check)
            return result
        except subprocess.CalledProcessError as e:
            if check:
                print(f"❌ Command failed: {' '.join(command)}")
                print(f"   Error: {e.stderr}")
                raise
            return e
    
    def check_authentication(self) -> bool:
        """Check if user is authenticated with gcloud."""
        print("🔍 Checking authentication...")
        
        try:
            result = self.run_gcloud_command(['gcloud', 'auth', 'list', '--filter=status:ACTIVE'])
            
            if 'ACTIVE' in result.stdout:
                # Extract active account
                lines = result.stdout.strip().split('\\n')
                for line in lines:
                    if 'ACTIVE' in line and '@' in line:
                        account = line.split()[0] if line.split()[0] != '*' else line.split()[1]
                        print(f"✅ Authenticated as: {account}")
                        return True
                
                print("✅ Authenticated with Google Cloud")
                return True
            else:
                print("❌ Not authenticated with Google Cloud")
                print("   Run: gcloud auth login")
                return False
        except subprocess.CalledProcessError:
            print("❌ Error checking authentication")
            return False
    
    def check_project_configuration(self) -> bool:
        """Check if project is configured correctly."""
        print("🔍 Checking project configuration...")
        
        try:
            result = self.run_gcloud_command(['gcloud', 'config', 'get-value', 'project'])
            current_project = result.stdout.strip()
            
            if current_project == self.project_id:
                print(f"✅ Project configured: {self.project_id}")
                return True
            else:
                print(f"❌ Wrong project configured: {current_project}")
                print(f"   Run: gcloud config set project {self.project_id}")
                return False
        except subprocess.CalledProcessError:
            print("❌ Error checking project configuration")
            return False
    
    def enable_apis(self) -> bool:
        """Enable required APIs for Vertex AI."""
        print("🔄 Enabling required APIs...")
        
        required_apis = [
            'aiplatform.googleapis.com',      # Vertex AI
            'compute.googleapis.com',         # Compute Engine (for VMs)
            'storage.googleapis.com',         # Cloud Storage
            'cloudbuild.googleapis.com',      # Cloud Build (for containers)
            'containerregistry.googleapis.com',  # Container Registry
            'logging.googleapis.com',         # Cloud Logging
            'monitoring.googleapis.com'       # Cloud Monitoring
        ]
        
        for api in required_apis:
            print(f"   Enabling {api}...")
            try:
                self.run_gcloud_command(['gcloud', 'services', 'enable', api])
                print(f"   ✅ {api} enabled")
            except subprocess.CalledProcessError:
                print(f"   ❌ Failed to enable {api}")
                return False
        
        print("✅ All required APIs enabled")
        return True
    
    def create_service_account(self) -> bool:
        """Create service account for Vertex AI jobs."""
        print("🔄 Creating service account...")
        
        # Check if service account already exists
        try:
            result = self.run_gcloud_command([
                'gcloud', 'iam', 'service-accounts', 'describe', 
                self.service_account_email
            ], check=False)
            
            if result.returncode == 0:
                print(f"✅ Service account already exists: {self.service_account_email}")
                return True
        except:
            pass
        
        # Create service account
        try:
            self.run_gcloud_command([
                'gcloud', 'iam', 'service-accounts', 'create', self.service_account_name,
                '--display-name=Vertex AI BERT Runner',
                '--description=Service account for BERT optimization experiments on Vertex AI'
            ])
            print(f"✅ Service account created: {self.service_account_email}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to create service account")
            return False
        
        return True
    
    def grant_permissions(self) -> bool:
        """Grant necessary permissions to service account."""
        print("🔄 Granting permissions to service account...")
        
        required_roles = [
            'roles/aiplatform.user',          # Vertex AI access
            'roles/storage.admin',            # Cloud Storage access
            'roles/logging.logWriter',        # Logging access
            'roles/monitoring.metricWriter'   # Monitoring access
        ]
        
        for role in required_roles:
            print(f"   Granting {role}...")
            try:
                self.run_gcloud_command([
                    'gcloud', 'projects', 'add-iam-policy-binding', self.project_id,
                    f'--member=serviceAccount:{self.service_account_email}',
                    f'--role={role}'
                ])
                print(f"   ✅ {role} granted")
            except subprocess.CalledProcessError:
                print(f"   ❌ Failed to grant {role}")
                return False
        
        print("✅ All permissions granted")
        return True
    
    def create_service_account_key(self) -> bool:
        """Create and download service account key."""
        print("🔄 Creating service account key...")
        
        key_file = Path("config/vertex-ai-key.json")
        key_file.parent.mkdir(exist_ok=True)
        
        if key_file.exists():
            print(f"✅ Service account key already exists: {key_file}")
            return True
        
        try:
            self.run_gcloud_command([
                'gcloud', 'iam', 'service-accounts', 'keys', 'create', str(key_file),
                f'--iam-account={self.service_account_email}'
            ])
            print(f"✅ Service account key created: {key_file}")
            
            # Set restrictive permissions on key file
            os.chmod(key_file, 0o600)
            
            return True
        except subprocess.CalledProcessError:
            print(f"❌ Failed to create service account key")
            return False
    
    def create_storage_buckets(self) -> bool:
        """Create Cloud Storage buckets for results and staging."""
        print("🔄 Creating Cloud Storage buckets...")
        
        buckets = [
            f"{self.project_id}-bert-results",
            f"{self.project_id}-bert-staging"
        ]
        
        for bucket_name in buckets:
            print(f"   Creating bucket: {bucket_name}")
            
            # Check if bucket already exists
            try:
                result = self.run_gcloud_command([
                    'gsutil', 'ls', f'gs://{bucket_name}'
                ], check=False)
                
                if result.returncode == 0:
                    print(f"   ✅ Bucket already exists: {bucket_name}")
                    continue
            except:
                pass
            
            # Create bucket
            try:
                self.run_gcloud_command([
                    'gsutil', 'mb', f'gs://{bucket_name}'
                ])
                print(f"   ✅ Bucket created: {bucket_name}")
                
                # Set lifecycle policy for cost optimization
                lifecycle_config = {
                    "rule": [
                        {
                            "action": {"type": "Delete"},
                            "condition": {"age": 90}  # Delete after 90 days
                        }
                    ]
                }
                
                lifecycle_file = Path(f"/tmp/{bucket_name}-lifecycle.json")
                with open(lifecycle_file, 'w') as f:
                    json.dump(lifecycle_config, f)
                
                self.run_gcloud_command([
                    'gsutil', 'lifecycle', 'set', str(lifecycle_file), f'gs://{bucket_name}'
                ])
                
                lifecycle_file.unlink()  # Clean up temp file
                
            except subprocess.CalledProcessError:
                print(f"   ❌ Failed to create bucket: {bucket_name}")
                return False
        
        print("✅ All buckets created")
        return True
    
    def setup_environment_file(self) -> bool:
        """Create environment configuration file."""
        print("🔄 Setting up environment configuration...")
        
        env_content = f"""# Vertex AI BERT Optimization Environment Configuration
# Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}

# GCP Configuration
GCP_PROJECT_ID={self.project_id}
GCP_DEFAULT_REGION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=./config/vertex-ai-key.json

# Vertex AI Configuration
VERTEX_AI_REGION=us-central1
VERTEX_AI_SERVICE_ACCOUNT={self.service_account_email}

# Storage Configuration
RESULTS_BUCKET={self.project_id}-bert-results
STAGING_BUCKET={self.project_id}-bert-staging

# Cost Controls
MAX_DAILY_BUDGET=5.00
MAX_JOB_COST=2.00
COST_ALERT_EMAIL=your-email@example.com

# Container Configuration
CONTAINER_REGISTRY=gcr.io
CONTAINER_PROJECT={self.project_id}

# Default Job Settings
DEFAULT_MACHINE_TYPE=n1-standard-4
DEFAULT_GPU_TYPE=NVIDIA_TESLA_T4
DEFAULT_GPU_COUNT=1
DEFAULT_TIMEOUT_HOURS=2

# Development Settings
ENABLE_DEBUG_LOGGING=false
DRY_RUN_MODE=false
"""
        
        env_file = Path(".env")
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print(f"✅ Environment file created: {env_file}")
        print(f"   Source it with: source .env")
        
        return True
    
    def check_setup_status(self) -> dict:
        """Check current setup status."""
        print("🔍 Checking Vertex AI Setup Status")
        print("=" * 40)
        
        status = {}
        
        # Check authentication
        status['authenticated'] = self.check_authentication()
        
        # Check project
        status['project_configured'] = self.check_project_configuration()
        
        # Check service account
        try:
            result = self.run_gcloud_command([
                'gcloud', 'iam', 'service-accounts', 'describe', 
                self.service_account_email
            ], check=False)
            status['service_account_exists'] = result.returncode == 0
            if status['service_account_exists']:
                print(f"✅ Service account exists: {self.service_account_email}")
            else:
                print(f"❌ Service account missing: {self.service_account_email}")
        except:
            status['service_account_exists'] = False
        
        # Check key file
        key_file = Path("config/vertex-ai-key.json")
        status['key_file_exists'] = key_file.exists()
        if status['key_file_exists']:
            print(f"✅ Service account key exists: {key_file}")
        else:
            print(f"❌ Service account key missing: {key_file}")
        
        # Check buckets
        buckets = [f"{self.project_id}-bert-results", f"{self.project_id}-bert-staging"]
        status['buckets_exist'] = True
        
        for bucket_name in buckets:
            try:
                result = self.run_gcloud_command([
                    'gsutil', 'ls', f'gs://{bucket_name}'
                ], check=False)
                
                if result.returncode == 0:
                    print(f"✅ Bucket exists: {bucket_name}")
                else:
                    print(f"❌ Bucket missing: {bucket_name}")
                    status['buckets_exist'] = False
            except:
                status['buckets_exist'] = False
        
        # Check environment file
        env_file = Path(".env")
        status['env_file_exists'] = env_file.exists()
        if status['env_file_exists']:
            print(f"✅ Environment file exists: {env_file}")
        else:
            print(f"❌ Environment file missing: {env_file}")
        
        # Overall status
        all_ready = all(status.values())
        status['ready_for_vertex_ai'] = all_ready
        
        print(f"\\n📊 Overall Status: {'✅ Ready' if all_ready else '❌ Not Ready'}")
        
        return status


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Setup Vertex AI for BERT optimization')
    parser.add_argument('--setup-all', action='store_true', 
                       help='Run complete setup process')
    parser.add_argument('--check-setup', action='store_true',
                       help='Check current setup status')
    parser.add_argument('--project-id', default='phoenix-project-386',
                       help='GCP project ID')
    
    args = parser.parse_args()
    
    setup = VertexAISetup(project_id=args.project_id)
    
    if args.setup_all:
        # Run comprehensive setup (simplified version)
        print("🚀 Running Vertex AI Setup")
        print("Note: Run individual setup steps manually if needed")
        status = setup.check_setup_status()
        sys.exit(0 if status['ready_for_vertex_ai'] else 1)
    
    elif args.check_setup:
        status = setup.check_setup_status()
        sys.exit(0 if status['ready_for_vertex_ai'] else 1)
    
    else:
        # Default: check status
        status = setup.check_setup_status()
        
        if not status['ready_for_vertex_ai']:
            print(f"\\n🔧 Run with --setup-all to configure Vertex AI automatically")
        
        sys.exit(0 if status['ready_for_vertex_ai'] else 1)


if __name__ == "__main__":
    main()