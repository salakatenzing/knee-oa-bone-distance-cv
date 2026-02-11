#!/usr/bin/env python3
"""
Compare results between Octave and CUDA implementations
"""

import numpy as np
import sys
import os

def load_csv(filename):
    """Load CSV file with thickness measurements"""
    try:
        data = np.loadtxt(filename, delimiter=',')
        return data
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def calculate_statistics(diff):
    """Calculate difference statistics"""
    return {
        'max': np.max(np.abs(diff)),
        'min': np.min(np.abs(diff)),
        'mean': np.mean(np.abs(diff)),
        'std': np.std(diff),
        'median': np.median(np.abs(diff)),
        'percentile_95': np.percentile(np.abs(diff), 95),
        'percentile_99': np.percentile(np.abs(diff), 99)
    }

def print_comparison(octave_file, cuda_file):
    """Print detailed comparison between two result files"""

    print("=" * 70)
    print("CUDA vs Octave Results Comparison")
    print("=" * 70)
    print()

    # Load data
    octave_data = load_csv(octave_file)
    cuda_data = load_csv(cuda_file)

    if octave_data is None or cuda_data is None:
        print("Error: Could not load one or both files")
        return False

    # Check shapes
    print(f"Octave results shape: {octave_data.shape}")
    print(f"CUDA results shape:   {cuda_data.shape}")
    print()

    if octave_data.shape != cuda_data.shape:
        print("ERROR: Shape mismatch! Results cannot be compared.")
        return False

    # Calculate differences
    diff = cuda_data - octave_data
    abs_diff = np.abs(diff)

    # Overall statistics
    print("Overall Difference Statistics:")
    print("-" * 70)
    stats = calculate_statistics(diff)
    print(f"  Maximum absolute difference:  {stats['max']:.6f}")
    print(f"  Minimum absolute difference:  {stats['min']:.6f}")
    print(f"  Mean absolute difference:     {stats['mean']:.6f}")
    print(f"  Median absolute difference:   {stats['median']:.6f}")
    print(f"  Standard deviation:           {stats['std']:.6f}")
    print(f"  95th percentile:              {stats['percentile_95']:.6f}")
    print(f"  99th percentile:              {stats['percentile_99']:.6f}")
    print()

    # Per-region statistics
    num_regions = octave_data.shape[1] if len(octave_data.shape) > 1 else 1
    if num_regions > 1:
        print("Per-Region Statistics:")
        print("-" * 70)
        region_names = ['Anterior', 'Middle', 'Posterior']
        for i in range(min(num_regions, len(region_names))):
            region_diff = diff[:, i]
            print(f"\n  {region_names[i]} Region:")
            print(f"    Mean absolute diff: {np.mean(np.abs(region_diff)):.6f}")
            print(f"    Max absolute diff:  {np.max(np.abs(region_diff)):.6f}")
            print(f"    Std deviation:      {np.std(region_diff):.6f}")
        print()

    # Tolerance checks
    print("Tolerance Checks:")
    print("-" * 70)
    tolerances = [0.01, 0.1, 0.5, 1.0, 5.0]
    for tol in tolerances:
        within_tol = np.sum(abs_diff <= tol)
        total = abs_diff.size
        percentage = (within_tol / total) * 100
        status = "✓" if percentage > 95 else "✗"
        print(f"  {status} Within ±{tol:5.2f}: {within_tol:6d}/{total} ({percentage:6.2f}%)")
    print()

    # Find largest differences
    print("Top 10 Largest Differences:")
    print("-" * 70)
    flat_diff = abs_diff.flatten()
    top_indices = np.argsort(flat_diff)[-10:][::-1]

    for rank, idx in enumerate(top_indices, 1):
        if num_regions > 1:
            case_idx = idx // num_regions
            region_idx = idx % num_regions
            region_name = region_names[region_idx] if region_idx < len(region_names) else f"Region {region_idx}"
            octave_val = octave_data[case_idx, region_idx]
            cuda_val = cuda_data[case_idx, region_idx]
            diff_val = diff[case_idx, region_idx]
            print(f"  {rank:2d}. Case {case_idx+1:3d}, {region_name:9s}: "
                  f"Octave={octave_val:7.3f}, CUDA={cuda_val:7.3f}, Diff={diff_val:+7.3f}")
        else:
            octave_val = octave_data.flatten()[idx]
            cuda_val = cuda_data.flatten()[idx]
            diff_val = diff.flatten()[idx]
            print(f"  {rank:2d}. Index {idx:4d}: "
                  f"Octave={octave_val:7.3f}, CUDA={cuda_val:7.3f}, Diff={diff_val:+7.3f}")
    print()

    # Overall assessment
    print("=" * 70)
    if stats['mean'] < 0.1 and stats['percentile_95'] < 1.0:
        print("✓ EXCELLENT: Results are very close (mean diff < 0.1)")
    elif stats['mean'] < 0.5 and stats['percentile_95'] < 2.0:
        print("✓ GOOD: Results are acceptably close (mean diff < 0.5)")
    elif stats['mean'] < 1.0:
        print("⚠ ACCEPTABLE: Results have some differences (mean diff < 1.0)")
    else:
        print("✗ WARNING: Results show significant differences (mean diff >= 1.0)")
    print("=" * 70)
    print()

    # Note about differences
    print("Note: Small differences are expected due to:")
    print("  - Floating-point precision differences between CPU and GPU")
    print("  - Different implementations of connected components")
    print("  - Rounding in intermediate calculations")
    print()

    return True

def main():
    """Main function"""
    if len(sys.argv) < 3:
        print("Usage: python3 compare_results.py <octave_results.csv> <cuda_results.csv>")
        print()
        print("Example:")
        print("  python3 compare_results.py octave_output.csv test_distance_vector.csv")
        sys.exit(1)

    octave_file = sys.argv[1]
    cuda_file = sys.argv[2]

    # Check if files exist
    if not os.path.exists(octave_file):
        print(f"Error: File not found: {octave_file}")
        sys.exit(1)

    if not os.path.exists(cuda_file):
        print(f"Error: File not found: {cuda_file}")
        sys.exit(1)

    # Perform comparison
    success = print_comparison(octave_file, cuda_file)

    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
