#!/usr/bin/env python3
"""
Run the complete TracIn influence analysis pipeline for a trained model.

This script executes steps 3-7 of the pipeline in sequence:
  3. Detect mispredictions
  4. Compute TracIn influence scores
  5a. Generate analysis dashboards
  5b. Cross-reference analysis
  6. Inspect mislabeled candidates
  7. Inspect influential samples

Usage:
    python scripts/run_full_pipeline.py --version 1
    python scripts/run_full_pipeline.py --version 1 --top_k 50 --batch_size 16
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_script(script_name: str, args: list, description: str):
    """Run a Python script and check for errors."""
    print(f"\n{'='*70}")
    print(f"STEP: {description}")
    print(f"{'='*70}")
    
    cmd = [sys.executable, script_name] + args
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    if result.returncode != 0:
        print(f"\n[ERROR] {script_name} failed with return code {result.returncode}")
        sys.exit(result.returncode)
    
    print(f"\n[SUCCESS] {description} complete")


def main():
    parser = argparse.ArgumentParser(
        description='Run complete TracIn influence analysis pipeline'
    )
    parser.add_argument('--version', type=int, required=True, 
                        help='Model version to analyze')
    parser.add_argument('--top_k', type=int, default=100, 
                        help='Top-K influences per test sample')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for influence computation')
    parser.add_argument('--top_n', type=int, default=20, 
                        help='Number of top samples to inspect')
    parser.add_argument('--skip_influence', action='store_true',
                        help='Skip influence computation (use existing results)')
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    version = args.version
    
    print("="*70)
    print("TracIn Influence Analysis Pipeline")
    print("="*70)
    print(f"Model Version: v{version}")
    print(f"Top-K: {args.top_k}")
    print(f"Batch Size: {args.batch_size}")
    print("="*70)
    
    # Step 3: Detect mispredictions
    run_script(
        str(script_dir / '3_detect_mispredictions.py'),
        ['--version', str(version)],
        'Detecting mispredictions'
    )
    
    # Step 4: Compute TracIn influence scores
    if not args.skip_influence:
        run_script(
            str(script_dir / '4_compute_influence.py'),
            ['--version', str(version), 
             '--top_k', str(args.top_k),
             '--batch_size', str(args.batch_size)],
            'Computing TracIn influence scores'
        )
    else:
        print("\n[SKIP] Skipping influence computation (--skip_influence)")
    
    # Step 5a: Generate dashboards
    run_script(
        str(script_dir / '5a_generate_dashboards.py'),
        ['--version', str(version)],
        'Generating analysis dashboards'
    )
    
    # Step 5b: Cross-reference analysis
    run_script(
        str(script_dir / '5b_cross_reference_analysis.py'),
        ['--version', str(version)],
        'Cross-referencing mispredictions with influences'
    )
    
    # Step 6: Inspect mislabeled candidates
    run_script(
        str(script_dir / '6_inspect_mislabeled.py'),
        ['--version', str(version), '--top_n', str(args.top_n)],
        'Inspecting mislabeled candidates'
    )
    
    # Step 7: Inspect influential samples
    run_script(
        str(script_dir / '7_inspect_influential.py'),
        ['--version', str(version), '--top_n', str(args.top_n)],
        'Inspecting most influential samples'
    )
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"\nAll outputs saved to: outputs/v{version}/")
    print(f"  - mispredictions/")
    print(f"  - influence_analysis/")
    print(f"  - inspection/")
    print("\n[DONE] Full pipeline complete!")


if __name__ == '__main__':
    main()
