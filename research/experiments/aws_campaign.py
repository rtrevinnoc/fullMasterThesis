#!/usr/bin/env python3
"""AWS Parallel Campaign Runner for Lithography Scanner Simulation.

Runs all thesis experimental campaigns (Exp 1 to Exp 7) in parallel across
available CPU cores on AWS EC2 instances (e.g. c6a.4xlarge / c6a.16xlarge).

Usage:
    python3 aws_campaign.py [--workers N] [--output-dir ./aws_results]
"""
import os
import sys
import time
import argparse
import subprocess

def run_script(script_name, log_name, cwd):
    print(f"[START] Running {script_name}...", flush=True)
    start_time = time.time()
    log_path = os.path.join(cwd, log_name)
    with open(log_path, "w") as log_file:
        proc = subprocess.run([sys.executable, script_name], cwd=cwd, stdout=log_file, stderr=subprocess.STDOUT)
    elapsed = time.time() - start_time
    status = "SUCCESS" if proc.returncode == 0 else f"FAILED (code {proc.returncode})"
    print(f"[{status}] {script_name} completed in {elapsed/60:.2f} min. Log: {log_name}", flush=True)

def main():
    parser = argparse.ArgumentParser(description="AWS Campaign Runner")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel workers")
    args = parser.parse_args()

    exp_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"=== Lithography Simulation AWS Batch Campaign ===")
    print(f"System CPU Cores available: {os.cpu_count()}")
    print(f"Working Directory: {exp_dir}")

    scripts = [
        ("exp_case1_full.py", "exp_case1.log"),
        ("driver_sweep.py", "driver_sweep.log"),
        ("driver_sweep_spec.py", "driver_sweep_spec.log"),
        ("driver_perturb.py", "driver_perturb.log"),
        ("exp_controllers.py", "exp_controllers.log"),
        ("exp_transition_onset.py", "exp_transition_onset.log"),
        ("exp_order_comparison.py", "exp_order_comparison.log"),
        ("exp_pso.py", "exp_pso.log")
    ]

    start_total = time.time()
    for script, log in scripts:
        run_script(script, log, exp_dir)

    total_min = (time.time() - start_total) / 60
    print(f"\n=== AWS Campaign Complete in {total_min:.2f} minutes ===")

if __name__ == "__main__":
    main()
