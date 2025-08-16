#!/usr/bin/env python3
"""
Validation Script: Python vs Stata Comparison

This script runs both Python and Stata implementations of mhtexp2 and compares results.
Supports both generated test data and user-provided datasets.
"""

import numpy as np
import pandas as pd
import subprocess
import os
import sys
import tempfile
import time
from pathlib import Path

# Add the mhtexp2 package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from mhtexp2 import mhtexp2


def find_stata():
    """Find Stata executable automatically"""
    # Common Stata locations on different systems (newest versions first)
    possible_paths = [
        # macOS locations - Stata 19 first
        "/Applications/Stata/Stata19.app/Contents/MacOS/stata-mp",
        "/Applications/Stata/Stata19.app/Contents/MacOS/stata-se", 
        "/Applications/Stata/Stata19.app/Contents/MacOS/stata-ic",
        "/Applications/StataNow/Stata19.app/Contents/MacOS/stata-mp",
        "/Applications/StataNow/Stata19.app/Contents/MacOS/stata-se",
        "/Applications/StataNow/Stata19.app/Contents/MacOS/stata-ic",
        # macOS Stata 18, 17, 16
        "/Applications/Stata/StataMP.app/Contents/MacOS/stata-mp",
        "/Applications/Stata/StataSE.app/Contents/MacOS/stata-se", 
        "/Applications/Stata/StataIC.app/Contents/MacOS/stata-ic",
        "/Applications/StataNow/StataMP.app/Contents/MacOS/stata-mp",
        "/Applications/StataNow/StataSE.app/Contents/MacOS/stata-se",
        "/Applications/StataNow/StataIC.app/Contents/MacOS/stata-ic",
        # Windows locations
        "C:/Program Files/Stata19/StataMP-64.exe",
        "C:/Program Files/Stata18/StataMP-64.exe",
        "C:/Program Files/Stata17/StataMP-64.exe",
        "C:/Program Files/Stata16/StataMP-64.exe", 
        "C:/Program Files/Stata15/StataMP-64.exe",
        # Linux locations
        "/usr/local/stata19/stata-mp",
        "/usr/local/stata18/stata-mp",
        "/usr/local/stata17/stata-mp",
        "/usr/local/stata16/stata-mp",
        "/usr/local/stata15/stata-mp"
    ]
    
    # Check if stata is in PATH
    try:
        result = subprocess.run(["which", "stata"], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    
    # Check common installation paths
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # If not found, return None
    return None


def generate_test_data(n=1000):
    """Generate test data for validation"""
    
    # Create treatment effects that match R version
    treat = np.random.choice([0, 1, 2], size=n, replace=True)
    subgroup = np.random.choice([1, 2], size=n, replace=True)
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    
    # Create outcome variables with treatment effects
    treatment_effects_y1 = np.array([0, 0.5, 1.0])
    treatment_effects_y2 = np.array([0, 0.3, 0.8])
    
    y1 = np.random.normal(0, 1, n) + 0.1 * treatment_effects_y1[treat]
    y2 = np.random.normal(0, 1, n) + 0.1 * treatment_effects_y2[treat]
    
    return pd.DataFrame({
        'treat': treat,
        'subgroup': subgroup,
        'x1': x1,
        'x2': x2,
        'y1': y1,
        'y2': y2
    })


def run_stata_mhtexp2(data, combo="pairwise", bootstrap=5000, studentized=True, 
                     transitivity_check=True, stata_path=None, verbose=True):
    """Run Stata mhtexp2 and return results"""
    
    # Use temporary directory for files
    import tempfile
    temp_dir = tempfile.mkdtemp()
    temp_data_file = os.path.join(temp_dir, "temp_validation_data.csv")
    data.to_csv(temp_data_file, index=False)
    
    # Create Stata script
    # Get paths to the Stata ado files
    project_root = Path(__file__).parent.parent.parent  
    stata_ado_path = project_root / "Stata"
    
    stata_script_lines = [
        "clear all",
        "set more off",
        f"adopath + \"{stata_ado_path}\"",  # Add Stata directory to ado path
        f"import delimited \"{os.path.basename(temp_data_file)}\"",  # Use basename since we're in temp dir
        (f"mhtexp2 y1 y2, treatment(treat) controls(x1 x2) subgroup(subgroup) "
         f"combo({combo}) bootstrap({bootstrap}) "
         f"studentized({int(studentized)}) "
         f"transitivitycheck({int(transitivity_check)})"),
        "mata:",
        "result_matrix = st_matrix(\"results\")",
        "fh = fopen(\"temp_stata_results.csv\", \"w\")",
        "fput(fh, \"outcome,subgroup,t1,t2,coefficient,Remark3_2,Thm3_1,Remark3_8,Bonf,Holm\")",
        "for (i = 1; i <= rows(result_matrix); i++) {",
        "  row_str = strofreal(result_matrix[i,1])",
        "  for (j = 2; j <= cols(result_matrix); j++) {",
        "    row_str = row_str + \",\" + strofreal(result_matrix[i,j])",
        "  }",
        "  fput(fh, row_str)",
        "}",
        "fclose(fh)",
        "end",
        "exit"
    ]
    
    stata_script = "\n".join(stata_script_lines)
    
    # Write and execute Stata script
    temp_script_file = os.path.join(temp_dir, "temp_stata_script.do")
    temp_results_file = os.path.join(temp_dir, "temp_stata_results.csv")
    
    with open(temp_script_file, "w") as f:
        f.write(stata_script)
    
    # Change to temp directory for execution
    old_cwd = os.getcwd()
    os.chdir(temp_dir)
    
    start_time = time.time()
    result = subprocess.run([stata_path, "-b", "do", "temp_stata_script.do"], 
                          capture_output=not verbose, text=True)
    stata_time = time.time() - start_time
    
    if verbose:
        print(f"Stata execution time: {stata_time:.2f} seconds")
    
    # Read results
    if os.path.exists(temp_results_file):
        stata_results = pd.read_csv(temp_results_file)
        
        # Add comparison column to match Python output
        def get_comparison(row):
            if row['t1'] == 0 and row['t2'] == 1:
                return 1
            elif row['t1'] == 0 and row['t2'] == 2:
                return 2
            elif row['t1'] == 1 and row['t2'] == 2:
                return 3
            else:
                return None
        
        stata_results['comparison'] = stata_results.apply(get_comparison, axis=1)
        
        # Return to original directory
        os.chdir(old_cwd)
        
        # Clean up temporary directory
        import shutil
        shutil.rmtree(temp_dir)
        
        return stata_results
    else:
        # Return to original directory
        os.chdir(old_cwd)
        # Clean up temporary directory
        import shutil
        shutil.rmtree(temp_dir)
        raise RuntimeError("Stata execution failed - no results file generated")


def compare_results(python_output, stata_output, verbose=True):
    """Compare Python and Stata results"""
    
    # Sort both results by (outcome, subgroup, t1, t2)
    python_sorted = python_output.sort_values(['outcome', 'subgroup', 't1', 't2']).reset_index(drop=True)
    stata_sorted = stata_output.sort_values(['outcome', 'subgroup', 't1', 't2']).reset_index(drop=True)
    
    if len(python_sorted) != len(stata_sorted):
        print(f"Warning: Different number of hypotheses: Python={len(python_sorted)}, Stata={len(stata_sorted)}")
    
    # Create comparison table
    n_rows = min(len(python_sorted), len(stata_sorted))
    
    comparison = pd.DataFrame({
        'hypothesis': range(1, n_rows + 1),
        'outcome': python_sorted['outcome'][:n_rows],
        'subgroup': python_sorted['subgroup'][:n_rows],
        't1': python_sorted['t1'][:n_rows],
        't2': python_sorted['t2'][:n_rows],
        
        # Python results
        'coef_Python': python_sorted['coefficient'][:n_rows],
        'Remark3_2_Python': python_sorted['Remark3_2'][:n_rows],
        'Thm3_1_Python': python_sorted['Thm3_1'][:n_rows],
        'Remark3_8_Python': python_sorted['Remark3_8'][:n_rows],
        
        # Stata results
        'coef_Stata': stata_sorted['coefficient'][:n_rows],
        'Remark3_2_Stata': stata_sorted['Remark3_2'][:n_rows],
        'Thm3_1_Stata': stata_sorted['Thm3_1'][:n_rows],
        'Remark3_8_Stata': stata_sorted['Remark3_8'][:n_rows],
    })
    
    # Calculate differences
    comparison['diff_coef'] = comparison['coef_Python'] - comparison['coef_Stata']
    comparison['diff_Remark3_2'] = comparison['Remark3_2_Python'] - comparison['Remark3_2_Stata']
    comparison['diff_Thm3_1'] = comparison['Thm3_1_Python'] - comparison['Thm3_1_Stata']
    comparison['diff_Remark3_8'] = comparison['Remark3_8_Python'] - comparison['Remark3_8_Stata']
    
    return comparison


def validate_mhtexp2_python(data_source="generate", n_obs=1000, combo="pairwise", 
                           bootstrap=5000, studentized=True, transitivity_check=True,
                           stata_path=None, verbose=True):
    """
    Main validation function
    
    Parameters
    ----------
    data_source : str
        "generate" to create test data, or path to CSV file
    n_obs : int
        Number of observations (if generating data)
    combo : str
        Comparison type: "pairwise" or "treatmentcontrol" 
    bootstrap : int
        Number of bootstrap replications
    studentized : bool
        Whether to studentize test statistics
    transitivity_check : bool
        Whether to apply transitivity corrections
    stata_path : str, optional
        Path to Stata executable
    verbose : bool
        Whether to print detailed output
        
    Returns
    -------
    dict
        Dictionary with comparison results, raw results, and parameters
    """
    
    if verbose:
        print("=== MHTEXP2 PYTHON vs STATA VALIDATION ===\n")
    
    # Auto-detect Stata if path not provided
    if stata_path is None:
        stata_path = find_stata()
        if stata_path is None:
            raise RuntimeError("Stata not found. Please install Stata or provide stata_path parameter.")
        if verbose:
            print(f"Found Stata at: {stata_path}")
    
    # 1. Load or generate data
    if data_source == "generate":
        if verbose:
            print(f"Generating test data with {n_obs} observations...")
        test_data = generate_test_data(n_obs)
    else:
        if verbose:
            print(f"Loading data from: {data_source}")
        test_data = pd.read_csv(data_source)
        n_obs = len(test_data)
    
    # Validate data structure
    required_cols = ["y1", "y2", "treat", "x1", "x2", "subgroup"]
    missing_cols = set(required_cols) - set(test_data.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
    
    if verbose:
        print("Data summary:")
        print(f"  Observations: {n_obs}")
        print(f"  Treatments: {', '.join(map(str, sorted(test_data['treat'].unique())))}")
        print(f"  Subgroups: {', '.join(map(str, sorted(test_data['subgroup'].unique())))}")
        print(f"  Y1 mean: {test_data['y1'].mean():.4f}")
        print(f"  Y2 mean: {test_data['y2'].mean():.4f}\n")
    
    # 2. Run Python implementation
    if verbose:
        print("Running Python implementation...")
    
    start_time = time.time()
    python_results = mhtexp2(
        Y=test_data[['y1', 'y2']].values,
        treatment=test_data['treat'].values,
        controls=test_data[['x1', 'x2']].values,
        subgroup=test_data['subgroup'].values,
        combo=combo,
        bootstrap=bootstrap,
        studentized=studentized,
        transitivity_check=transitivity_check
    )
    python_time = time.time() - start_time
    
    if verbose:
        print(f"Python execution time: {python_time:.2f} seconds")
    
    # 3. Run Stata implementation
    if verbose:
        print("Running Stata implementation...")
    
    stata_results = run_stata_mhtexp2(
        data=test_data,
        combo=combo,
        bootstrap=bootstrap,
        studentized=studentized,
        transitivity_check=transitivity_check,
        stata_path=stata_path,
        verbose=verbose
    )
    
    # 4. Compare results
    if verbose:
        print("Comparing results...")
    
    # Get Python output table
    python_output = python_results['output_table']
    
    comparison = compare_results(python_output, stata_results, verbose=verbose)
    
    # 5. Summary
    if verbose:
        print(f"\nComparison complete: {len(comparison)} hypotheses tested")
        print("Use result['comparison'] to see the full comparison table")
    
    return {
        'comparison': comparison,
        'python_results': python_results,
        'stata_results': stata_results,
        'data': test_data,
        'params': {
            'combo': combo,
            'bootstrap': bootstrap,
            'studentized': studentized,
            'transitivity_check': transitivity_check,
            'n_obs': n_obs
        }
    }


if __name__ == "__main__":
    # Example usage
    print("Running validation with default parameters...")
    result = validate_mhtexp2_python(
        n_obs=500, 
        bootstrap=5000,  # Larger bootstrap for more stable results
        verbose=True
    )
    
    print("\nALL HYPOTHESES COMPARISON:")
    print(result['comparison'].to_string(index=False))
    
    # Show summary of differences
    comparison = result['comparison']
    print("\nSummary of differences:")
    print(f"Max absolute coefficient difference: {abs(comparison['diff_coef']).max():.6f}")
    print(f"Max absolute Remark 3.2 difference: {abs(comparison['diff_Remark3_2']).max():.6f}")
    print(f"Max absolute Theorem 3.1 difference: {abs(comparison['diff_Thm3_1']).max():.6f}")
    print(f"Max absolute Remark 3.8 difference: {abs(comparison['diff_Remark3_8']).max():.6f}")