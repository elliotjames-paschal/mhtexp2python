"""
Basic usage examples for mhtexp2 package.

This script demonstrates how to use the mhtexp2 package for multiple
hypothesis testing in experimental economics.
"""

import numpy as np
import pandas as pd
from mhtexp2 import mhtexp2


def example_basic_usage():
    """Basic example with synthetic data."""
    print("=== Basic Usage Example ===")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic experimental data
    n_obs = 200
    n_outcomes = 3
    
    # Create outcome variables with some treatment effects
    Y = np.random.randn(n_obs, n_outcomes)
    
    # Treatment assignment (0 = control, 1 = treatment)
    treatment = np.concatenate([np.zeros(100), np.ones(100)])
    
    # Add treatment effects to outcomes 1 and 3 (indices 0 and 2)
    Y[treatment == 1, [0, 2]] += 0.3
    
    # Run basic analysis
    results = mhtexp2(
        Y=Y,
        treatment=treatment,
        combo="treatmentcontrol",
        bootstrap=1000  # Reduced for faster example
    )
    
    print(f"Number of outcomes: {results['summary']['n_outcomes']}")
    print(f"Number of observations: {results['summary']['n_observations']}")
    print(f"Raw p-values: {results['pvalues_raw']}")
    print(f"Bonferroni-corrected p-values: {results['pvalues_corrected'].get('bonferroni', 'Not available')}")
    print()


def example_with_controls():
    """Example with control variables."""
    print("=== Example with Control Variables ===")
    
    np.random.seed(123)
    
    # Generate data with controls
    n_obs = 150
    n_outcomes = 4
    n_controls = 2
    
    # Control variables
    controls = np.random.randn(n_obs, n_controls)
    
    # Treatment assignment
    treatment = np.random.choice([0, 1, 2], n_obs)  # 3 treatment groups
    
    # Outcome variables (influenced by controls and treatment)
    Y = np.random.randn(n_obs, n_outcomes)
    
    # Add control effects
    for i in range(n_outcomes):
        Y[:, i] += 0.2 * controls[:, 0] + 0.1 * controls[:, 1]
    
    # Add treatment effects
    treatment_effects = {0: 0.0, 1: 0.25, 2: 0.4}
    for treat_val, effect in treatment_effects.items():
        mask = treatment == treat_val
        Y[mask, :] += effect
    
    # Run analysis with controls
    results = mhtexp2(
        Y=Y,
        treatment=treatment,
        controls=controls,
        combo="pairwise",
        bootstrap=1000
    )
    
    print(f"Number of treatment groups: {results['summary']['n_treatments']}")
    print(f"Raw p-values: {results['pvalues_raw']}")
    print(f"Holm-corrected p-values: {results['pvalues_corrected'].get('holm', 'Not available')}")
    print()


def example_with_pandas():
    """Example using pandas DataFrames."""
    print("=== Example with Pandas DataFrames ===")
    
    np.random.seed(456)
    
    # Create data as pandas DataFrame
    n_obs = 100
    
    data = pd.DataFrame({
        'outcome_1': np.random.randn(n_obs),
        'outcome_2': np.random.randn(n_obs),
        'outcome_3': np.random.randn(n_obs),
        'treatment': np.random.choice(['control', 'treatment_a', 'treatment_b'], n_obs),
        'age': np.random.randint(18, 65, n_obs),
        'gender': np.random.choice([0, 1], n_obs),
        'baseline_score': np.random.randn(n_obs)
    })
    
    # Add treatment effects
    treatment_effects = {'control': 0, 'treatment_a': 0.3, 'treatment_b': 0.5}
    for treatment_name, effect in treatment_effects.items():
        mask = data['treatment'] == treatment_name
        data.loc[mask, ['outcome_1', 'outcome_2']] += effect
    
    # Convert treatment to numeric
    treatment_mapping = {'control': 0, 'treatment_a': 1, 'treatment_b': 2}
    data['treatment_numeric'] = data['treatment'].map(treatment_mapping)
    
    # Define outcome and control variables
    outcome_vars = ['outcome_1', 'outcome_2', 'outcome_3']
    control_vars = ['age', 'gender', 'baseline_score']
    
    # Run analysis
    results = mhtexp2(
        Y=data[outcome_vars],
        treatment=data['treatment_numeric'],
        controls=data[control_vars],
        combo="pairwise",
        bootstrap=1000,
        studentized=True
    )
    
    print("Analysis with pandas DataFrame:")
    print(f"Outcome variables: {outcome_vars}")
    print(f"Control variables: {control_vars}")
    print(f"Raw p-values: {results['pvalues_raw']}")
    print()


def example_subgroup_analysis():
    """Example with subgroup analysis."""
    print("=== Example with Subgroup Analysis ===")
    
    np.random.seed(789)
    
    n_obs = 200
    n_outcomes = 2
    
    # Generate data
    Y = np.random.randn(n_obs, n_outcomes)
    treatment = np.random.choice([0, 1], n_obs)
    subgroup = np.random.choice([0, 1], n_obs)  # e.g., male=0, female=1
    
    # Add different treatment effects by subgroup
    # Subgroup 0: moderate treatment effect
    mask_treat_sub0 = (treatment == 1) & (subgroup == 0)
    Y[mask_treat_sub0, :] += 0.2
    
    # Subgroup 1: larger treatment effect
    mask_treat_sub1 = (treatment == 1) & (subgroup == 1)
    Y[mask_treat_sub1, :] += 0.5
    
    # Run analysis with subgroup
    results = mhtexp2(
        Y=Y,
        treatment=treatment,
        subgroup=subgroup,
        combo="treatmentcontrol",
        bootstrap=1000
    )
    
    print("Subgroup analysis:")
    print(f"Raw p-values: {results['pvalues_raw']}")
    print(f"Bootstrap replications: {results['summary']['bootstrap_replications']}")
    print()


def example_compare_corrections():
    """Example comparing different correction methods."""
    print("=== Comparison of Correction Methods ===")
    
    np.random.seed(100)
    
    # Generate data with varying effect sizes
    n_obs = 300
    n_outcomes = 5
    
    Y = np.random.randn(n_obs, n_outcomes)
    treatment = np.random.choice([0, 1], n_obs)
    
    # Add different effect sizes to different outcomes
    effect_sizes = [0.0, 0.1, 0.3, 0.5, 0.8]
    for i, effect in enumerate(effect_sizes):
        Y[treatment == 1, i] += effect
    
    # Run analysis
    results = mhtexp2(
        Y=Y,
        treatment=treatment,
        combo="treatmentcontrol",
        bootstrap=2000
    )
    
    print("Comparison of correction methods:")
    print(f"Raw p-values:        {np.round(results['pvalues_raw'], 4)}")
    
    for method, pvals in results['pvalues_corrected'].items():
        print(f"{method:15} p-values: {np.round(pvals, 4)}")
    
    print()


if __name__ == "__main__":
    print("mhtexp2 Package Examples")
    print("=" * 50)
    print()
    
    # Run all examples
    example_basic_usage()
    example_with_controls()
    example_with_pandas()
    example_subgroup_analysis()
    example_compare_corrections()
    
    print("All examples completed successfully!")
    print("Note: These examples use placeholder implementations.")
    print("Actual statistical procedures need to be implemented.")