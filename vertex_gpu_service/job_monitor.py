#!/usr/bin/env python3
"""
Vertex AI Job Monitoring Interface

Usage:
    python vertex_gpu_service/job_monitor.py --list
    python vertex_gpu_service/job_monitor.py --watch-all
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from .vertex_manager import VertexAIManager


def format_duration(start_time, end_time=None):
    """Format duration between timestamps."""
    if not start_time:
        return "N/A"
    
    if not end_time:
        end_time = datetime.now()
    
    if isinstance(start_time, str):
        start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
    if isinstance(end_time, str):
        end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
    
    duration = end_time - start_time
    
    hours = duration.total_seconds() / 3600
    if hours < 1:
        minutes = duration.total_seconds() / 60
        return f"{minutes:.1f}m"
    else:
        return f"{hours:.1f}h"


def print_job_status(job_info, detailed=False):
    """Print formatted job status."""
    name = job_info.get('display_name', 'Unknown')
    state = job_info.get('state', 'UNKNOWN')
    create_time = job_info.get('create_time')
    start_time = job_info.get('start_time')
    end_time = job_info.get('end_time')
    
    # Status emoji
    status_emoji = {
        'JOB_STATE_QUEUED': '⏳',
        'JOB_STATE_PENDING': '⏳',
        'JOB_STATE_RUNNING': '🔄',
        'JOB_STATE_SUCCEEDED': '✅',
        'JOB_STATE_FAILED': '❌',
        'JOB_STATE_CANCELLED': '🛑',
        'JOB_STATE_CANCELLING': '🛑',
        'JOB_STATE_PAUSED': '⏸️'
    }.get(state, '❓')
    
    # Calculate duration
    if state == 'JOB_STATE_RUNNING' and start_time:
        duration = format_duration(start_time)
        duration_text = f"Running for {duration}"
    elif end_time and start_time:
        duration = format_duration(start_time, end_time)
        duration_text = f"Completed in {duration}"
    elif create_time:
        duration = format_duration(create_time)
        duration_text = f"Created {duration} ago"
    else:
        duration_text = "Duration unknown"
    
    print(f"{status_emoji} {name}")
    print(f"   State: {state}")
    print(f"   {duration_text}")
    
    if detailed:
        print(f"   Resource: {job_info.get('name', 'Unknown')}")
        if create_time:
            print(f"   Created: {create_time}")
        if start_time:
            print(f"   Started: {start_time}")
        if end_time:
            print(f"   Ended: {end_time}")
        if job_info.get('error'):
            print(f"   Error: {job_info['error']}")


def list_jobs(manager, limit=10):
    """List recent Vertex AI jobs."""
    print(f"📋 Recent Vertex AI Jobs (limit: {limit})")
    print("=" * 50)
    
    jobs = manager.list_jobs(limit=limit)
    
    if not jobs:
        print("No jobs found")
        return
    
    for i, job in enumerate(jobs):
        if i > 0:
            print()
        print_job_status(job)


def watch_all_jobs(manager, interval=30):
    """Watch all running jobs."""
    print(f"👀 Watching all jobs (refresh every {interval}s)")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    try:
        while True:
            # Clear screen (works on most terminals)
            print("\\033[2J\\033[H")
            
            print(f"🔄 Vertex AI Jobs Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)
            
            jobs = manager.list_jobs(limit=20)
            
            if not jobs:
                print("No jobs found")
            else:
                running_jobs = []
                completed_jobs = []
                
                for job in jobs:
                    state = job.get('state', 'UNKNOWN')
                    if state in ['JOB_STATE_RUNNING', 'JOB_STATE_PENDING', 'JOB_STATE_QUEUED']:
                        running_jobs.append(job)
                    else:
                        completed_jobs.append(job)
                
                if running_jobs:
                    print(f"\\n🔄 RUNNING JOBS ({len(running_jobs)}):")
                    print("-" * 30)
                    for job in running_jobs:
                        print_job_status(job)
                        print()
                
                if completed_jobs:
                    print(f"\\n📋 RECENT COMPLETED JOBS ({len(completed_jobs)}):")
                    print("-" * 35)
                    for job in completed_jobs[:5]:  # Show only 5 most recent
                        print_job_status(job)
                        print()
            
            print(f"\\n⏳ Next refresh in {interval} seconds... (Ctrl+C to stop)")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\\n\\n👋 Monitoring stopped")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Monitor Vertex AI BERT experiments')
    parser.add_argument('--list', action='store_true', help='List recent jobs')
    parser.add_argument('--watch-all', action='store_true', help='Watch all jobs')
    parser.add_argument('--limit', type=int, default=10, help='Limit number of jobs to show')
    parser.add_argument('--project-id', default='phoenix-project-386', help='GCP project ID')
    parser.add_argument('--region', default='us-central1', help='GCP region')
    parser.add_argument('--interval', type=int, default=30, help='Refresh interval for watch mode')
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = VertexAIManager(
        project_id=args.project_id,
        region=args.region
    )
    
    if args.list or (not args.watch_all):
        list_jobs(manager, limit=args.limit)
    
    elif args.watch_all:
        watch_all_jobs(manager, interval=args.interval)


if __name__ == "__main__":
    main()