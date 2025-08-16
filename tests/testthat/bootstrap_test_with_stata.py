#!/usr/bin/env python3
"""
Bootstrap Test with Stata Data

This script validates Python statistical corrections using bootstrap data exported from Stata.
It replicates the functionality of statatest.R from the R package.

Usage:
    python bootstrap_test_with_stata.py

This script automatically:
    1. Runs export_bootstrap.do in Stata to generate bootstrap data
    2. Loads the generated CSV files (abregact.csv, abregboot.csv, pact.csv, pboot.csv)
    3. Validates Python statistical corrections against Stata bootstrap results
"""

import numpy as np
import pandas as pd
import sys
import os
import subprocess
import time

# Add the main package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mhtexp2.corrections import (
    calculate_alphasin, 
    calculate_alpha_unified,
    validate_statistics_with_external_data
)

def run_stata_export_bootstrap():
    """Run Stata run_test.do script to generate CSV files with 5000 bootstrap samples."""
    print("Step 1: Running Stata run_test.do...")
    print("=" * 60)
    
    # Check if run_test.do exists
    if not os.path.exists('run_test.do'):
        print("ERROR: run_test.do not found in current directory")
        return False
    
    # Try to find Stata executable
    stata_paths = [
        "/Applications/StataNow/StataMP.app/Contents/MacOS/stata-mp",
        "/Applications/Stata/StataMP.app/Contents/MacOS/stata-mp",
        "/Applications/StataNow/StataSE.app/Contents/MacOS/stata-se",
        "/Applications/Stata/StataSE.app/Contents/MacOS/stata-se"
    ]
    
    stata_path = None
    for path in stata_paths:
        if os.path.exists(path):
            stata_path = path
            break
    
    if stata_path is None:
        print("ERROR: Stata not found. Please run manually:")
        print("  1. Open Stata GUI")
        print("  2. In Stata, type: do run_test.do")
        print("  3. Wait for completion, then rerun this script")
        return False
    
    try:
        print(f"Found Stata at: {stata_path}")
        start_time = time.time()
        
        # Run Stata script
        result = subprocess.run([
            stata_path, '-b', 'do', 'run_test.do'
        ], capture_output=True, text=True, cwd='.')
        
        execution_time = time.time() - start_time
        print(f"Stata execution time: {execution_time:.2f} seconds")
        
        if result.returncode != 0:
            print("ERROR: Stata execution failed")
            print("STDERR:", result.stderr)
            return False
            
        print("✓ Stata export completed successfully")
        return True
        
    except Exception as e:
        print(f"ERROR running Stata: {e}")
        return False

def csv_to_array(file_path, has_bootstrap=False):
    """
    Convert CSV to numpy array (equivalent to R csv_to_array function).
    
    Parameters
    ----------
    file_path : str
        Path to CSV file
    has_bootstrap : bool
        Whether data has bootstrap dimension (4D vs 3D)
        
    Returns
    -------
    np.ndarray
        Converted array
    """
    print(f"Loading {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"WARNING: File not found: {file_path}")
        print("   Please run the Stata export_bootstrap.do script first")
        return None
    
    # Read CSV
    data = pd.read_csv(file_path)
    print(f"   Loaded {len(data)} rows")
    
    if has_bootstrap:
        # 4D array with bootstrap dimension
        max_l = int(data['l'].max())
        max_i = int(data['i'].max()) 
        max_j = int(data['j'].max())
        max_k = int(data['k'].max())
        
        print(f"   Creating 4D array: ({max_l}, {max_i}, {max_j}, {max_k})")
        result = np.full((max_l, max_i, max_j, max_k), np.nan)
        
        # Fill the array (using 0-based indexing)
        for _, row in data.iterrows():
            l = int(row['l']) - 1  # Convert to 0-based
            i = int(row['i']) - 1  # Convert to 0-based 
            j = int(row['j']) - 1  # Convert to 0-based
            k = int(row['k']) - 1  # Convert to 0-based
            result[l, i, j, k] = row['value']
            
    else:
        # 3D array without bootstrap dimension
        max_i = int(data['i'].max())
        max_j = int(data['j'].max()) 
        max_k = int(data['k'].max())
        
        print(f"   Creating 3D array: ({max_i}, {max_j}, {max_k})")
        result = np.full((max_i, max_j, max_k), np.nan)
        
        # Fill the array (using 0-based indexing)
        for _, row in data.iterrows():
            i = int(row['i']) - 1  # Convert to 0-based
            j = int(row['j']) - 1  # Convert to 0-based  
            k = int(row['k']) - 1  # Convert to 0-based
            result[i, j, k] = row['value']
    
    print(f"   Array shape: {result.shape}")
    return result

def load_stata_bootstrap_data(base_path="."):
    """
    Load Stata bootstrap data from CSV files.
    
    Parameters
    ----------
    base_path : str
        Base directory containing CSV files
        
    Returns
    -------
    dict
        Dictionary with loaded arrays
    """
    print("Loading Stata bootstrap data from CSV files...")
    print("=" * 60)
    
    # File paths  
    files = {
        'abregact': os.path.join(base_path, 'abregact.csv'),
        'abregboot': os.path.join(base_path, 'abregboot.csv'), 
        'pact': os.path.join(base_path, 'pact.csv'),
        'pboot': os.path.join(base_path, 'pboot.csv')
    }
    
    # Load arrays
    data = {}
    data['abregact'] = csv_to_array(files['abregact'], has_bootstrap=False)
    data['abregboot'] = csv_to_array(files['abregboot'], has_bootstrap=True) 
    data['pact'] = csv_to_array(files['pact'], has_bootstrap=False)
    data['pboot'] = csv_to_array(files['pboot'], has_bootstrap=True)
    
    # Check if any failed to load
    if any(arr is None for arr in data.values()):
        print("\nFailed to load some files. Please check paths and run Stata export_bootstrap.do first.")
        return None
    
    # Create combo matrix (treatment pairs)
    # From statatest.R: combo_mat <- matrix(c(0, 1, 0, 2, 1, 2), nrow = 3, byrow = TRUE)
    data['combo'] = np.array([[0, 1], [0, 2], [1, 2]])
    
    # Create select array (all hypotheses selected)
    num_outcomes, num_subgroups, num_comparisons = data['abregact'].shape
    data['select'] = np.ones((num_outcomes, num_subgroups, num_comparisons))
    
    print(f"\nSuccessfully loaded all data:")
    print(f"   - Outcomes: {num_outcomes}")
    print(f"   - Subgroups: {num_subgroups}") 
    print(f"   - Comparisons: {num_comparisons}")
    print(f"   - Bootstrap samples: {data['abregboot'].shape[0]}")
    
    return data

def run_python_corrections(data):
    """
    Run Python statistical corrections using Stata's exact p-values.
    
    Parameters
    ----------
    data : dict
        Dictionary with loaded Stata data
        
    Returns
    -------
    pd.DataFrame
        Results DataFrame from Python corrections
    """
    print("\nRunning Python statistical corrections...")
    print("=" * 50)
    
    # Use Stata's exact p-value calculations (pact and pboot)
    pvals = data['pact']  # Stata-calculated observed p-values
    pboot = data['pboot']  # Stata-calculated bootstrap p-values
    
    # Step 1: Calculate single hypothesis corrections (Remark 3.2) using Stata p-values
    alpha_sin = calculate_alphasin(pvals, pboot)
    
    # Step 2: Calculate multiple hypothesis corrections using Stata p-values
    unified_results = calculate_alpha_unified(
        pvals=pvals,
        coefficients=data['abregact'],
        pboot=pboot,
        combo=data['combo'],
        alphasin=alpha_sin,
        select=data['select'],
        transitivitycheck=True
    )
    
    alpha_mul = unified_results['alphamul']
    alpha_mulm = unified_results['alphamulm']
    
    # Step 3: Build output DataFrame
    from mhtexp2.corrections import build_output
    output_df = build_output(
        stat=data['abregact'],
        coef=data['abregact'],
        combo=data['combo'],
        alpha_sin=alpha_sin,
        alpha_mul=alpha_mul,
        pvals=pvals,
        alpha_mulm=alpha_mulm,
        select=data['select']
    )
    
    print(f"✓ Generated output DataFrame with {len(output_df)} rows")
    return output_df

def build_results_dataframe(data, corrections):
    """Build results DataFrame matching R/Stata format."""
    print("\nBuilding results DataFrame...")
    
    num_outcomes, num_subgroups, num_comparisons = data['abregact'].shape
    
    # Create indices (matching R convention: outcome varies slowest)
    results_list = []
    
    for outcome in range(num_outcomes):
        for subgroup in range(num_subgroups):
            for comparison in range(num_comparisons):
                idx = outcome * num_subgroups * num_comparisons + subgroup * num_comparisons + comparison
                
                results_list.append({
                    'outcome': outcome + 1,      # 1-based for output
                    'subgroup': subgroup + 1,    # 1-based for output  
                    'comparison': comparison + 1, # 1-based for output
                    't1': data['combo'][comparison, 0],
                    't2': data['combo'][comparison, 1], 
                    'coefficient': data['abregact'][outcome, subgroup, comparison],
                    'test_stat': data['abregact'][outcome, subgroup, comparison],
                    'Remark3_2': corrections['alpha1'][outcome, subgroup, comparison],
                    'Thm3_1': corrections['alpha2'][outcome, subgroup, comparison],
                    'Remark3_8': corrections['alpha3'][outcome, subgroup, comparison],
                    'Bonf': corrections['bonf_flat'][idx],
                    'Holm': corrections['holm_flat'][idx]
                })
    
    df = pd.DataFrame(results_list)
    print(f"   Created DataFrame with {len(df)} rows")
    
    return df

def main():
    """Main validation function."""
    print("Python Validation Test for mhtexp2")
    print("Replicating statatest.R functionality in Python")
    print("="*70)
    
    # Step 1: Load Stata bootstrap data
    # Use current test directory for CSV files
    base_path = os.path.dirname(__file__)
    
    data = load_stata_bootstrap_data(base_path)
    if data is None:
        return
    
    # Step 2: Run Python corrections
    print("DEBUG: About to call run_python_corrections...")
    corrections = run_python_corrections(data)
    print(f"DEBUG: Corrections keys: {list(corrections.keys())}")
    
    # Step 3: Use corrected output DataFrame
    if 'output_df' in corrections and corrections['output_df'] is not None:
        results_df = corrections['output_df']
        print(f"✓ Using corrected output DataFrame with {len(results_df)} hypotheses")
        print("DEBUG: Output DataFrame columns:", list(results_df.columns))
        print("DEBUG: First few rows:")
        print(results_df.head().to_string(index=False))
    else:
        print("ERROR: No output_df found in corrections, falling back to manual build")
        results_df = build_results_dataframe(data, corrections)
    
    # Step 4: Display results
    print("\nPython Results (first 10 rows):")
    print("-" * 60)
    print(results_df.head(10).to_string(index=False, float_format='%.6f'))
    
    # Step 5: Summary statistics
    print(f"\nSummary Statistics:")
    print("-" * 30)
    print(f"Total hypotheses: {len(results_df)}")
    print(f"Remark3_2 range: [{results_df['Remark3_2'].min():.6f}, {results_df['Remark3_2'].max():.6f}]")
    print(f"Thm3_1 range: [{results_df['Thm3_1'].min():.6f}, {results_df['Thm3_1'].max():.6f}]") 
    print(f"Remark3_8 range: [{results_df['Remark3_8'].min():.6f}, {results_df['Remark3_8'].max():.6f}]")
    
    # Step 6: Save results for comparison
    output_file = os.path.join(os.path.dirname(__file__), 'python_results.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    print("\nNext Steps:")
    print("   1. Compare python_results.csv with R/Stata output")
    print("   2. Check for numerical differences")
    print("   3. Investigate any discrepancies")
    
    return results_df

def load_stata_final_results():
    """Load Stata final statistical results from stata_results.csv."""
    results_file = "stata_results.csv"
    
    if not os.path.exists(results_file):
        print(f"ERROR: {results_file} not found")
        print("The export_bootstrap.do script should have generated this file")
        return None
    
    try:
        stata_results = pd.read_csv(results_file)
        print(f"✓ Loaded {len(stata_results)} Stata final results")
        print("Stata results preview:")
        print(stata_results.head().to_string(index=False))
        return stata_results
    except Exception as e:
        print(f"ERROR loading Stata results: {e}")
        return None

def compare_stata_python_results(stata_df, python_results):
    """Compare Stata and Python final results and create comparison table."""
    
    # Handle case where python_results is a DataFrame
    if isinstance(python_results, pd.DataFrame):
        python_df = python_results
    else:
        print("ERROR: Expected DataFrame from Python corrections but got:", type(python_results))
        return None
    
    if len(stata_df) != len(python_df):
        print(f"WARNING: Different number of results - Stata: {len(stata_df)}, Python: {len(python_df)}")
        print(f"Stata results shape: {stata_df.shape}")
        print(f"Python results shape: {python_df.shape}")
        print("This suggests the core issue persists - Python is not generating all hypotheses")
        return None
    
    # Create comparison DataFrame
    comparison_data = []
    for i in range(len(stata_df)):
        stata_row = stata_df.iloc[i]
        python_row = python_df.iloc[i]
        
        comparison_data.append({
            'Hypothesis': f"H{i+1}: Out{stata_row['outcome']} Sub{stata_row['subgroup']} T{stata_row['t1']}vs{stata_row['t2']}",
            'Stata_Coef': stata_row['coefficient'],
            'Python_Coef': python_row['coefficient'],
            'Diff_Coef': abs(stata_row['coefficient'] - python_row['coefficient']),
            'Stata_R3.2': stata_row['Remark3_2'],
            'Python_R3.2': python_row['Remark3_2'],
            'Diff_R3.2': abs(stata_row['Remark3_2'] - python_row['Remark3_2']),
            'Stata_T3.1': stata_row['Thm3_1'],
            'Python_T3.1': python_row['Thm3_1'],
            'Diff_T3.1': abs(stata_row['Thm3_1'] - python_row['Thm3_1']),
            'Stata_R3.8': stata_row['Remark3_8'],
            'Python_R3.8': python_row['Remark3_8'],
            'Diff_R3.8': abs(stata_row['Remark3_8'] - python_row['Remark3_8']),
            'Match_R3.2': "✓" if abs(stata_row['Remark3_2'] - python_row['Remark3_2']) < 0.001 else "✗",
            'Match_T3.1': "✓" if abs(stata_row['Thm3_1'] - python_row['Thm3_1']) < 0.001 else "✗",
            'Match_R3.8': "✓" if abs(stata_row['Remark3_8'] - python_row['Remark3_8']) < 0.001 else "✗"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display comparison table
    print("STATA vs PYTHON BOOTSTRAP VALIDATION RESULTS:")
    print("="*120)
    print(comparison_df[['Hypothesis', 'Stata_R3.2', 'Python_R3.2', 'Diff_R3.2', 'Match_R3.2',
                        'Stata_T3.1', 'Python_T3.1', 'Diff_T3.1', 'Match_T3.1',
                        'Stata_R3.8', 'Python_R3.8', 'Diff_R3.8', 'Match_R3.8']].to_string(index=False))
    
    # Summary statistics
    total_matches_r32 = sum(comparison_df['Match_R3.2'] == "✓")
    total_matches_t31 = sum(comparison_df['Match_T3.1'] == "✓")
    total_matches_r38 = sum(comparison_df['Match_R3.8'] == "✓")
    total_hypotheses = len(comparison_df)
    
    print(f"\nSUMMARY:")
    print(f"Remark 3.2 matches: {total_matches_r32}/{total_hypotheses} ({total_matches_r32/total_hypotheses*100:.0f}%)")
    print(f"Theorem 3.1 matches: {total_matches_t31}/{total_hypotheses} ({total_matches_t31/total_hypotheses*100:.0f}%)")
    print(f"Remark 3.8 matches: {total_matches_r38}/{total_hypotheses} ({total_matches_r38/total_hypotheses*100:.0f}%)")
    
    print(f"\nMAX DIFFERENCES:")
    print(f"Remark 3.2: {comparison_df['Diff_R3.2'].max():.6f}")
    print(f"Theorem 3.1: {comparison_df['Diff_T3.1'].max():.6f}")
    print(f"Remark 3.8: {comparison_df['Diff_R3.8'].max():.6f}")
    
    # Show cases with significant differences
    significant_diffs = comparison_df[
        (comparison_df['Diff_R3.2'] > 0.001) | 
        (comparison_df['Diff_T3.1'] > 0.001) | 
        (comparison_df['Diff_R3.8'] > 0.001)
    ]
    
    if len(significant_diffs) > 0:
        print(f"\nHYPOTHESES WITH SIGNIFICANT DIFFERENCES (>0.001):")
        print(significant_diffs[['Hypothesis', 'Diff_R3.2', 'Diff_T3.1', 'Diff_R3.8']].to_string(index=False))
    else:
        print("\n✅ ALL DIFFERENCES ARE NEGLIGIBLE (<0.001)")
    
    return comparison_df

def main():
    """Main execution function."""
    print("Bootstrap Test with Stata Data")
    print("="*60)
    print("Validating Python statistical corrections using Stata bootstrap data\n")
    
    # Step 1: Run Stata export script
    if not run_stata_export_bootstrap():
        print("\nFailed to generate Stata bootstrap data. Exiting.")
        return None
    
    # Step 2: Load Stata bootstrap data  
    print("\nStep 2: Loading Stata bootstrap data...")
    print("="*60)
    data = load_stata_bootstrap_data()
    if data is None:
        print("Failed to load bootstrap data. Exiting.")
        return None
    
    # Step 3: Load Stata final results
    print("\nStep 3: Loading Stata final results...")
    print("="*60)
    stata_final_results = load_stata_final_results()
    if stata_final_results is None:
        print("Failed to load Stata final results. Exiting.")
        return None
    
    # Step 4: Run Python corrections
    print("\nStep 4: Running Python statistical corrections...")
    print("="*60)
    python_results = run_python_corrections(data)
    
    
    # Step 5: Compare results
    print("\nStep 5: Comparing Stata vs Python results...")
    print("="*60)
    comparison = compare_stata_python_results(stata_final_results, python_results)
    
    print("\n" + "="*60)
    print("Bootstrap validation completed!")
    return {
        'stata_results': stata_final_results,
        'python_results': python_results,
        'comparison': comparison
    }

if __name__ == "__main__":
    results = main()