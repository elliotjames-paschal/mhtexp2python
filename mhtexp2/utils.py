"""
Utility functions for mhtexp2 package.

Helper functions for data processing and statistical computations.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, List


def validate_data_dimensions(
    Y: np.ndarray,
    treatment: np.ndarray,
    controls: Optional[np.ndarray] = None,
    subgroup: Optional[np.ndarray] = None
) -> None:
    """
    Validate that all input arrays have compatible dimensions.
    
    Parameters
    ----------
    Y : np.ndarray
        Outcome variables matrix
    treatment : np.ndarray
        Treatment assignment vector
    controls : np.ndarray, optional
        Control variables matrix
    subgroup : np.ndarray, optional
        Subgroup identifiers
        
    Raises
    ------
    ValueError
        If dimensions are incompatible
    """
    n_obs = Y.shape[0]
    
    if len(treatment) != n_obs:
        raise ValueError(f"Treatment vector length ({len(treatment)}) must match "
                        f"number of observations in Y ({n_obs})")
    
    if controls is not None and controls.shape[0] != n_obs:
        raise ValueError(f"Controls matrix rows ({controls.shape[0]}) must match "
                        f"number of observations in Y ({n_obs})")
    
    if subgroup is not None and len(subgroup) != n_obs:
        raise ValueError(f"Subgroup vector length ({len(subgroup)}) must match "
                        f"number of observations in Y ({n_obs})")


def check_treatment_balance(
    treatment: np.ndarray,
    controls: Optional[np.ndarray] = None,
    alpha: float = 0.05
) -> dict:
    """
    Check balance of treatment assignment across control variables.
    
    Parameters
    ----------
    treatment : np.ndarray
        Treatment assignment vector
    controls : np.ndarray, optional
        Control variables matrix
    alpha : float, default 0.05
        Significance level for balance tests
        
    Returns
    -------
    dict
        Balance test results
    """
    balance_results = {
        'treatment_counts': np.bincount(treatment.astype(int)),
        'treatment_proportions': np.bincount(treatment.astype(int)) / len(treatment)
    }
    
    if controls is not None:
        # TODO: Implement actual balance tests
        balance_results['balance_pvalues'] = np.ones(controls.shape[1])
        balance_results['balanced'] = True
    
    return balance_results


def compute_effect_sizes(
    Y: np.ndarray,
    treatment: np.ndarray,
    controls: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute effect sizes for treatment comparisons.
    
    Parameters
    ----------
    Y : np.ndarray
        Outcome variables matrix
    treatment : np.ndarray
        Treatment assignment vector
    controls : np.ndarray, optional
        Control variables matrix
        
    Returns
    -------
    np.ndarray
        Effect sizes for each outcome variable
    """
    unique_treatments = np.unique(treatment)
    n_outcomes = Y.shape[1]
    n_comparisons = len(unique_treatments) * (len(unique_treatments) - 1) // 2
    
    effect_sizes = np.zeros((n_outcomes, n_comparisons))
    
    comparison_idx = 0
    for i, treat1 in enumerate(unique_treatments):
        for treat2 in unique_treatments[i+1:]:
            mask1 = treatment == treat1
            mask2 = treatment == treat2
            
            y1 = Y[mask1]
            y2 = Y[mask2]
            
            # Cohen's d effect size
            pooled_std = np.sqrt(((len(y1)-1)*np.var(y1, axis=0, ddof=1) + 
                                 (len(y2)-1)*np.var(y2, axis=0, ddof=1)) / 
                                (len(y1) + len(y2) - 2))
            
            effect_sizes[:, comparison_idx] = (np.mean(y1, axis=0) - 
                                             np.mean(y2, axis=0)) / pooled_std
            comparison_idx += 1
    
    return effect_sizes


def build_output(stat: np.ndarray, coef: np.ndarray, combo: np.ndarray, 
                 alpha_sin: np.ndarray, alpha_mul: np.ndarray, pvals: np.ndarray,
                 alpha_mulm: np.ndarray, select: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Build output data frame with multiple testing corrections.
    
    Based on R implementation (build_output.R):
    Creates a formatted DataFrame with all hypothesis test results and corrections.
    
    Parameters
    ----------
    stat : np.ndarray
        3D array of test statistics (outcomes x subgroups x comparisons)
    coef : np.ndarray
        3D array of treatment effect coefficients (outcomes x subgroups x comparisons)
    combo : np.ndarray
        Matrix of treatment comparison pairs (numpc x 2)
    alpha_sin : np.ndarray
        3D array of single hypothesis adjusted p-values (Remark 3.2)
    alpha_mul : np.ndarray
        3D array of multiple hypothesis adjusted p-values (Theorem 3.1)
    pvals : np.ndarray
        3D array of p-values (1-p format)
    alpha_mulm : np.ndarray
        3D array of transitivity-corrected p-values (Remark 3.8)
    select : np.ndarray, optional
        3D array indicating which hypotheses to include (default: all)
        
    Returns
    -------
    pd.DataFrame
        Formatted results table with all corrections
    """
    dims = stat.shape
    num_outcomes, num_subgroups, num_comparisons = dims
    
    if select is None:
        select = np.ones_like(stat)
    
    results_list = []
    
    for i in range(num_outcomes):
        for j in range(num_subgroups):
            for k in range(num_comparisons):
                # Only include if selected (like R/Stata)
                if select[i, j, k] == 1:
                    results_list.append({
                        'outcome': i + 1,  # 1-based indexing for output
                        'subgroup': j + 1,  # 1-based indexing for output
                        'comparison': k + 1,  # 1-based indexing for output
                        't1': combo[k, 0],
                        't2': combo[k, 1],
                        'coefficient': coef[i, j, k],
                        'test_stat': stat[i, j, k],
                        'Remark3_2': alpha_sin[i, j, k],
                        'Thm3_1': alpha_mul[i, j, k],
                        'Remark3_8': alpha_mulm[i, j, k]
                    })
    
    # Create DataFrame
    df = pd.DataFrame(results_list)
    
    if len(df) == 0:
        return df
    
    # Add Bonferroni and Holm corrections based on Remark 3.2 p-values
    pvec = df['Remark3_2'].values
    nh = len(pvec)
    
    # Bonferroni
    df['Bonf'] = np.minimum(1.0, pvec * nh)
    
    # Holm
    order_indices = np.argsort(pvec)
    pvec_sorted = pvec[order_indices]
    correction_factors = np.arange(nh, 0, -1)  # [nh, nh-1, ..., 1]
    holm_adjust = pvec_sorted * correction_factors
    holm_adjust = np.minimum(np.maximum.accumulate(holm_adjust), 1.0)  # cummax + cap at 1
    
    # Map back to original order
    holm_result = np.zeros(nh)
    holm_result[order_indices] = holm_adjust
    df['Holm'] = holm_result
    
    return df


def format_results_table(results: dict) -> pd.DataFrame:
    """
    Format results into a readable table.
    
    Parameters
    ----------
    results : dict
        Results dictionary from mhtexp2
        
    Returns
    -------
    pd.DataFrame
        Formatted results table
    """
    n_outcomes = len(results['pvalues_raw'])
    outcome_names = [f"Outcome_{i+1}" for i in range(n_outcomes)]
    
    table_data = {
        'Outcome': outcome_names,
        'Raw_pvalue': results['pvalues_raw']
    }
    
    # Add corrected p-values
    for method, pvals in results['pvalues_corrected'].items():
        table_data[f'{method}_pvalue'] = pvals
    
    # Add confidence intervals if available
    if 'confidence_intervals' in results:
        ci = results['confidence_intervals']
        table_data['CI_lower'] = ci[:, 0]
        table_data['CI_upper'] = ci[:, 1]
    
    return pd.DataFrame(table_data)


def generate_summary_stats(Y: np.ndarray, treatment: np.ndarray) -> dict:
    """
    Generate descriptive statistics by treatment group.
    
    Parameters
    ----------
    Y : np.ndarray
        Outcome variables matrix
    treatment : np.ndarray
        Treatment assignment vector
        
    Returns
    -------
    dict
        Summary statistics by treatment group
    """
    unique_treatments = np.unique(treatment)
    n_outcomes = Y.shape[1]
    
    summary = {}
    
    for treat in unique_treatments:
        mask = treatment == treat
        y_treat = Y[mask]
        
        summary[f'treatment_{treat}'] = {
            'n_obs': len(y_treat),
            'means': np.mean(y_treat, axis=0),
            'std_devs': np.std(y_treat, axis=0, ddof=1),
            'medians': np.median(y_treat, axis=0)
        }
    
    return summary