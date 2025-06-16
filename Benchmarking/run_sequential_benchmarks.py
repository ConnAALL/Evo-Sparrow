#!/usr/bin/env python3
"""
Sequential Benchmark Runner
Runs multiple benchmark scripts one after another for overnight execution.
"""

import subprocess
import sys
import time
from datetime import datetime

def log_message(message):
    """Print a timestamped message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def run_script(script_name):
    """Run a single benchmark script and return success status."""
    log_message(f"Starting {script_name}")
    start_time = time.time()
    
    try:
        # Run the script using python
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, 
                              text=True, 
                              cwd=".")
        
        end_time = time.time()
        duration = end_time - start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        
        if result.returncode == 0:
            log_message(f"✓ {script_name} completed successfully in {hours}h {minutes}m")
            return True
        else:
            log_message(f"✗ {script_name} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        log_message(f"✗ {script_name} failed with exception after {hours}h {minutes}m: {str(e)}")
        return False

def main():
    # List of scripts to run in order
    scripts = [
        "ppo_ppo_rule.py",
        "ppo_ppo_ppo.py",
    ]
    
    log_message("Starting sequential benchmark execution")
    log_message(f"Will run {len(scripts)} benchmark scripts")
    
    overall_start = time.time()
    successful_runs = 0
    failed_runs = 0
    
    for i, script in enumerate(scripts, 1):
        log_message(f"=== Running script {i}/{len(scripts)}: {script} ===")
        
        if run_script(script):
            successful_runs += 1
        else:
            failed_runs += 1
            log_message(f"WARNING: {script} failed, but continuing with remaining scripts")
        
        # Add a small delay between scripts
        if i < len(scripts):
            log_message("Waiting 10 seconds before next script...")
            time.sleep(10)
    
    overall_end = time.time()
    total_duration = overall_end - overall_start
    total_hours = int(total_duration // 3600)
    total_minutes = int((total_duration % 3600) // 60)
    
    log_message("=== BENCHMARK EXECUTION SUMMARY ===")
    log_message(f"Total time: {total_hours}h {total_minutes}m")
    log_message(f"Successful runs: {successful_runs}/{len(scripts)}")
    log_message(f"Failed runs: {failed_runs}/{len(scripts)}")
    
    if failed_runs > 0:
        log_message("Some scripts failed. Check the output above for details.")
        sys.exit(1)
    else:
        log_message("All benchmark scripts completed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main() 