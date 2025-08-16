"""
P-value correction methods for multiple hypothesis testing.

Implements various correction procedures for controlling family-wise error rates.
"""

import numpy as np
from typing import Dict, Any, Optional
from scipy import stats


def calculate_pvals(observed: np.ndarray, bootstrap: np.ndarray) -> np.ndarray:
    """
    Calculate p-values from observed and bootstrap test statistics.
    
    Based on R implementation (pval_helpers.R lines 8-35):
    Calculates "1-p" values where p = (count of bootstrap >= observed) / B
    
    Parameters
    ----------
    observed : np.ndarray
        3D array (outcomes x subgroups x comparisons) of observed test statistics
    bootstrap : np.ndarray  
        4D array (B x outcomes x subgroups x comparisons) of bootstrap test statistics
        
    Returns
    -------
    np.ndarray
        3D array of "1-p" values (same shape as observed)
    """
    B = bootstrap.shape[0]
    pvals = np.zeros_like(observed)
    
    outcomes, subgroups, comparisons = observed.shape
    
    for i in range(outcomes):
        for j in range(subgroups):
            for k in range(comparisons):
                # Get observed test statistic for this hypothesis
                observed_stat = observed[i, j, k]
                
                # Get all bootstrap statistics for this hypothesis
                bootstrap_stats = bootstrap[:, i, j, k]
                
                # Count how many bootstrap stats >= observed stat
                exceed_count = np.sum(bootstrap_stats >= observed_stat)
                
                # Calculate 1 - p-value (matching R/Stata exactly)
                one_minus_p = 1 - (exceed_count / B)
                
                pvals[i, j, k] = one_minus_p
    
    return pvals


def apply_corrections(
    pvalues_raw: np.ndarray,
    method: str = "all",
    alpha: float = 0.05
) -> Dict[str, np.ndarray]:
    """
    Apply multiple hypothesis testing corrections.
    
    Parameters
    ----------
    pvalues_raw : np.ndarray
        Raw p-values from hypothesis tests
    method : str, default "all"
        Correction method to apply. Options: "all", "bonferroni", "holm", 
        "remark32", "theorem31", "remark38"
    alpha : float, default 0.05
        Significance level
        
    Returns
    -------
    dict
        Dictionary of corrected p-values for each method
    """
    
    corrections = {}
    
    if method == "all" or method == "bonferroni":
        corrections["bonferroni"] = bonferroni_correction(pvalues_raw, alpha)
    
    if method == "all" or method == "holm":
        corrections["holm"] = holm_correction(pvalues_raw, alpha)
    
    if method == "all" or method == "remark32":
        corrections["remark32"] = remark32_correction(pvalues_raw, alpha)
    
    if method == "all" or method == "theorem31":
        corrections["theorem31"] = theorem31_correction(pvalues_raw, alpha)
    
    if method == "all" or method == "remark38":
        corrections["remark38"] = remark38_correction(pvalues_raw, alpha)
    
    return corrections


def bonferroni_correction(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Apply Bonferroni correction for multiple comparisons.
    
    Based on Stata implementation: bon = rowmin((statsrank[.,7]*nh, J(nh,1,1)))
    
    Parameters
    ----------
    pvalues : np.ndarray
        Raw p-values (should be pre-sorted by significance)
    alpha : float
        Significance level
        
    Returns
    -------
    np.ndarray
        Bonferroni-corrected p-values
    """
    n_tests = len(pvalues)
    # Multiply each p-value by total number of tests, cap at 1.0
    corrected = np.minimum(pvalues * n_tests, 1.0)
    return corrected


def holm_correction(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Apply Holm step-down correction.
    
    Based on Stata implementation:
    holm = rowmin((statsrank[.,7]:*(nh::1), J(nh,1,1)))
    for (i = 2; i <= nh; i++) {
        holm[i] = holm[i] > holm[i-1] ? holm[i] : holm[i-1]
    }
    
    Parameters
    ----------
    pvalues : np.ndarray
        Raw p-values (should be pre-sorted by significance)
    alpha : float
        Significance level
        
    Returns
    -------
    np.ndarray
        Holm-corrected p-values
    """
    n_tests = len(pvalues)
    
    # Stata: holm = rowmin((statsrank[.,7]:*(nh::1), J(nh,1,1)))
    # This multiplies p-values by (n_tests, n_tests-1, n_tests-2, ..., 1)
    correction_factors = np.arange(n_tests, 0, -1)  # (nh::1) in Stata
    corrected = np.minimum(pvalues * correction_factors, 1.0)
    
    # Enforce monotonicity condition (Stata lines 443-445)
    for i in range(1, n_tests):
        if corrected[i] < corrected[i-1]:
            corrected[i] = corrected[i-1]
    
    return corrected


def calculate_alphasin(pact: np.ndarray, pboot: np.ndarray) -> np.ndarray:
    """
    Calculate single hypothesis adjusted thresholds (Remark 3.2).
    
    Based on R implementation (pval_helpers.R lines 38-82):
    For each hypothesis, compares actual "1-p" value against bootstrap distribution
    to find the quantile where actual value exceeds bootstrap values.
    
    Parameters
    ----------
    pact : np.ndarray
        3D array (outcomes x subgroups x comparisons) - observed "1-p" values
    pboot : np.ndarray
        4D array (B x outcomes x subgroups x comparisons) - bootstrap "1-p" values
        
    Returns
    -------
    np.ndarray
        3D array of adjusted alpha thresholds (same shape as pact)
    """
    B = pboot.shape[0]
    alphasin = np.zeros_like(pact)
    
    outcomes, subgroups, comparisons = pact.shape
    
    for i in range(outcomes):
        for j in range(subgroups):
            for k in range(comparisons):
                # Get observed "1-p" value for this hypothesis
                observed_1p = pact[i, j, k]
                
                # Get bootstrap "1-p" values for this hypothesis
                boot_1p = pboot[:, i, j, k]
                
                # Sort bootstrap values in descending order (largest first)
                sorted_boot = np.sort(boot_1p)[::-1]
                
                # Find where observed value is >= sorted bootstrap values
                v = observed_1p >= sorted_boot
                
                # Find first TRUE index (where observed >= bootstrap)
                # In R: which(v)[1] gives 1-based index of first TRUE
                # In Python: np.where(v)[0][0] gives 0-based index, so add 1
                indices = np.where(v)[0]
                if len(indices) == 0:
                    q = 1.0
                else:
                    # R uses 1-based indexing: first_idx ranges from 1 to B
                    first_idx = indices[0] + 1  
                    q = first_idx / B
                
                alphasin[i, j, k] = q
    
    return alphasin


def calculate_alpha_unified(pvals: np.ndarray, coefficients: np.ndarray, pboot: np.ndarray,
                           combo: np.ndarray, alphasin: np.ndarray, select: Optional[np.ndarray] = None,
                           transitivitycheck: bool = False) -> Dict[str, np.ndarray]:
    """
    Calculate multiple hypothesis testing thresholds with optional transitivity.
    
    Based on R implementation (fwer_control.R lines 13-337):
    Implements both Theorem 3.1 (standard) and Remark 3.8 (transitivity-aware) procedures.
    
    Parameters
    ----------
    pvals : np.ndarray
        3D array (outcomes x subgroups x comparisons) - observed "1-p" values
    coefficients : np.ndarray
        3D array (outcomes x subgroups x comparisons) - treatment effect coefficients  
    pboot : np.ndarray
        4D array (B x outcomes x subgroups x comparisons) - bootstrap "1-p" values
    combo : np.ndarray
        Matrix of treatment-control pairs
    alphasin : np.ndarray
        3D array (outcomes x subgroups x comparisons) - single-hypothesis adjusted p-values
    select : np.ndarray, optional
        3D array indicating which hypotheses to include (default = all)
    transitivitycheck : bool, default False
        Whether to apply transitivity correction (Remark 3.8)
        
    Returns
    -------
    dict
        Dictionary with 'alphamul' (Theorem 3.1) and 'alphamulm' (Remark 3.8) results
    """
    # Build dimensions
    outcomes, subgroups, comparisons = pvals.shape
    B = pboot.shape[0]
    
    if select is None:
        select = np.ones_like(pvals)
    
    nh = int(np.sum(select))  # Total number of hypotheses
    
    if nh == 0:
        return {'alphamul': np.zeros_like(pvals), 'alphamulm': np.zeros_like(pvals)}
    
    # Build statsall matrix - flatten selected hypotheses
    hypothesis_data = []
    for i in range(outcomes):
        for j in range(subgroups):
            for k in range(comparisons):
                if select[i, j, k] == 1:
                    hypothesis_data.append({
                        'outcome': i,
                        'subgroup': j, 
                        'comparison': k,
                        'coefficient': coefficients[i, j, k],
                        'alphasin': alphasin[i, j, k],
                        'pact': pvals[i, j, k],
                        'pboot': pboot[:, i, j, k]
                    })
    
    # Sort by single hypothesis p-values (alphasin)
    hypothesis_data.sort(key=lambda x: x['alphasin'])
    
    alphamul = np.zeros(nh)   # Standard results (Theorem 3.1)
    alphamulm = np.zeros(nh)  # Transitivity aware results (Remark 3.8)
    
    # Step-down procedure
    for i in range(nh):
        # ALWAYS calculate standard alphamul (Theorem 3.1)
        # Get remaining hypotheses (from i to end)
        remaining_pboot = np.array([hyp['pboot'] for hyp in hypothesis_data[i:]])  # (remaining_hyps, B)
        
        # For each bootstrap sample, take maximum across remaining hypotheses
        maxstats = np.max(remaining_pboot, axis=0)  # (B,)
        sortmaxstats = np.sort(maxstats)[::-1]  # Descending order
        
        # Compare observed to bootstrap max distribution
        observed_pval = hypothesis_data[i]['pact']
        v = observed_pval >= sortmaxstats
        indices = np.where(v)[0]
        
        alphamul[i] = 1.0 if len(indices) == 0 else (indices[0] + 1) / B
        
        # For transitivity (Remark 3.8) - full implementation
        if i == 0 or not transitivitycheck:
            # Copy standard result (matches R: alphamulm[i] = alphamul[i])
            alphamulm[i] = alphamul[i]
        else:
            # Full transitivity checking logic
            sortmaxstatsm = np.zeros(B)
            
            # Loop through subset sizes from largest to smallest
            for j in range(nh - i, 0, -1):  # (nh - i + 1):1 in R becomes range(nh - i, 0, -1)
                remaining_indices = list(range(i, nh))
                
                # Generate all subsets of size j
                if j > 0 and len(remaining_indices) >= j:
                    from itertools import combinations
                    subsets = list(combinations(remaining_indices, j))
                else:
                    continue
                
                if len(subsets) == 0:
                    continue
                
                sumcont = 0  # Count of contradictory subsets
                
                # Check each subset
                for subset_idx, subset in enumerate(subsets):
                    cont = 0  # Contradiction flag
                    
                    # Check against all previously rejected hypotheses
                    for l in range(i):  # 1:(i-1) in R becomes range(i)
                        
                        # Get outcome/subgroup info for subset and rejected hypothesis
                        subset_outcomes = [hypothesis_data[s]['outcome'] for s in subset]
                        subset_subgroups = [hypothesis_data[s]['subgroup'] for s in subset]
                        rejected_outcome = hypothesis_data[l]['outcome']
                        rejected_subgroup = hypothesis_data[l]['subgroup']
                        
                        # Find hypotheses in subset with same outcome/subgroup as rejected hypothesis l
                        sameocsub = []
                        for s_idx, s in enumerate(subset):
                            if (subset_outcomes[s_idx] == rejected_outcome and 
                                subset_subgroups[s_idx] == rejected_subgroup):
                                sameocsub.append(s)
                        
                        if len(sameocsub) <= 1:
                            # No transitivity constraint possible with ≤1 hypothesis
                            cont = 0
                            
                            # Process the ENTIRE subset (like R does)
                            subset_pboot = np.array([hypothesis_data[s]['pboot'] for s in subset])  # (subset_size, B)
                            
                            if len(subset) == 1:
                                maxstatsm = subset_pboot[0]  # Single hypothesis
                            else:
                                maxstatsm = np.max(subset_pboot, axis=0)  # Max across subset
                            
                            maxstatsm_sorted = np.sort(maxstatsm)[::-1]
                            sortmaxstatsm = np.maximum(sortmaxstatsm, maxstatsm_sorted)
                            break
                        
                        else:
                            # TRANSITIVITY ANALYSIS - Multiple matching hypotheses
                            
                            # Get treatment pairs for hypotheses with same outcome/subgroup
                            treatment_pairs = []
                            for s in sameocsub:
                                comp_idx = hypothesis_data[s]['comparison']
                                treatment_pairs.append(combo[comp_idx])
                            
                            # Build transitivity groups using transitive closure
                            tran_groups = _build_transitivity_groups(treatment_pairs)
                            
                            # Check for contradiction with rejected hypothesis l
                            rejected_comp_idx = hypothesis_data[l]['comparison']
                            rejected_treatments = combo[rejected_comp_idx]
                            
                            # Check if both treatments from rejected hypothesis are in same group
                            contradiction_found = False
                            for group in tran_groups:
                                if (rejected_treatments[0] in group and rejected_treatments[1] in group):
                                    # Both treatments from rejected hypothesis are in same group - contradiction!
                                    cont = 1
                                    contradiction_found = True
                                    break
                        
                        if cont == 1:
                            break  # Contradiction found, exit loop over rejected hypotheses
                    
                    sumcont += cont
                    
                    if cont == 0:
                        # Valid subset - process immediately (matches R exactly)
                        subset_pboot = np.array([hypothesis_data[s]['pboot'] for s in subset])
                        
                        if len(subset) == 1:
                            maxstats_subset = subset_pboot[0]
                        else:
                            maxstats_subset = np.max(subset_pboot, axis=0)
                        
                        maxstats_sorted = np.sort(maxstats_subset)[::-1]
                        sortmaxstatsm = np.maximum(sortmaxstatsm, maxstats_sorted)
                
                # Early termination if all subsets of size j are valid
                if sumcont == 0:
                    break  # Matches R's break condition exactly
            
            # Final p-value calculation
            observed_pval = hypothesis_data[i]['pact']
            v = observed_pval >= sortmaxstatsm
            indices = np.where(v)[0]
            
            if len(indices) == 0:
                qm = 1.0
            else:
                qm = (indices[0] + 1) / B
            
            alphamulm[i] = qm
    
    # Map results back to 3D arrays
    result_alphamul = np.zeros_like(pvals)
    result_alphamulm = np.zeros_like(pvals)
    
    for idx, hyp in enumerate(hypothesis_data):
        i, j, k = hyp['outcome'], hyp['subgroup'], hyp['comparison']
        result_alphamul[i, j, k] = alphamul[idx]
        result_alphamulm[i, j, k] = alphamulm[idx]
    
    return {
        'alphamul': result_alphamul,   # Theorem 3.1
        'alphamulm': result_alphamulm  # Remark 3.8
    }


def _build_transitivity_groups(treatment_pairs):
    """
    Build transitivity groups from treatment pairs using transitive closure.
    
    Based on R implementation (fwer_control.R lines 219-265):
    Iteratively merges treatment groups that share common treatments.
    
    Parameters
    ----------
    treatment_pairs : list
        List of treatment pairs [(t1, t2), (t2, t3), ...]
        
    Returns
    -------
    list
        List of transitivity groups, where each group is a set of treatments
    """
    if not treatment_pairs:
        return []
    
    # Initialize groups - each treatment pair starts as its own group
    tran_groups = [set(pair) for pair in treatment_pairs]
    
    # Iteratively merge groups until no more merging is possible (transitive closure)
    changed = True
    counter = 1
    
    while changed and counter < 100:  # Safety counter to prevent infinite loops
        changed = False
        counter += 1
        
        # Try to merge groups that share treatments
        new_groups = []
        used = set()
        
        for i, group_i in enumerate(tran_groups):
            if i in used:
                continue
                
            # Find all groups that can merge with group_i
            merged_group = group_i.copy()
            used.add(i)
            
            for j, group_j in enumerate(tran_groups):
                if j in used or i == j:
                    continue
                
                # Check if groups share any treatments (can be merged)
                if len(group_i.intersection(group_j)) > 0:
                    merged_group.update(group_j)
                    used.add(j)
                    changed = True
            
            new_groups.append(merged_group)
        
        tran_groups = new_groups
    
    return tran_groups


def validate_statistics_with_external_data(observed_stats: np.ndarray, bootstrap_stats: np.ndarray,
                                          observed_coef: np.ndarray, combo: np.ndarray,
                                          select: Optional[np.ndarray] = None,
                                          transitivitycheck: bool = True) -> dict:
    """
    Validate statistical methods using external bootstrap data (e.g., from Stata).
    
    This function allows testing our Python implementation against known bootstrap results
    from Stata or R to verify correctness of the statistical procedures.
    
    Parameters
    ----------
    observed_stats : np.ndarray
        3D array (outcomes x subgroups x comparisons) of observed test statistics
    bootstrap_stats : np.ndarray
        4D array (B x outcomes x subgroups x comparisons) of bootstrap test statistics
    observed_coef : np.ndarray
        3D array (outcomes x subgroups x comparisons) of treatment coefficients
    combo : np.ndarray
        Matrix of treatment comparison pairs (numpc x 2)
    select : np.ndarray, optional
        3D array indicating which hypotheses to include (default: all)
    transitivitycheck : bool, default True
        Whether to apply transitivity correction (Remark 3.8)
        
    Returns
    -------
    dict
        Complete results including all p-values and corrections
    """
    from .utils import build_output
    
    # Step 1: Calculate "1-p" values from bootstrap statistics
    pvals = calculate_pvals(observed_stats, bootstrap_stats)
    
    # Step 2: Build bootstrap "1-p" values for correction methods
    # For each bootstrap sample, calculate what its "1-p" value would be
    # relative to the full bootstrap distribution
    B = bootstrap_stats.shape[0]
    pboot = np.zeros_like(bootstrap_stats)
    
    outcomes, subgroups, comparisons = observed_stats.shape
    
    for i in range(outcomes):
        for j in range(subgroups):
            for k in range(comparisons):
                bootstrap_stats_ijk = bootstrap_stats[:, i, j, k]
                for b in range(B):
                    # For each bootstrap sample b, calculate its "1-p" value
                    # This is: 1 - (how many bootstrap stats >= this bootstrap sample / B)
                    exceed_count = np.sum(bootstrap_stats_ijk >= bootstrap_stats_ijk[b])
                    pboot[b, i, j, k] = 1 - (exceed_count / B)
    
    # Step 3: Calculate single hypothesis corrections (Remark 3.2)
    alpha_sin = calculate_alphasin(pvals, pboot)
    
    # Step 4: Calculate multiple hypothesis corrections (Theorem 3.1 + Remark 3.8)
    unified_results = calculate_alpha_unified(
        pvals=pvals,
        coefficients=observed_coef,
        pboot=pboot,
        combo=combo,
        alphasin=alpha_sin,
        select=select,
        transitivitycheck=transitivitycheck
    )
    
    alpha_mul = unified_results['alphamul']
    alpha_mulm = unified_results['alphamulm']
    
    # Step 5: Build formatted output
    output_df = build_output(
        stat=observed_stats,
        coef=observed_coef,
        combo=combo,
        alpha_sin=alpha_sin,
        alpha_mul=alpha_mul,
        pvals=pvals,
        alpha_mulm=alpha_mulm,
        select=select
    )
    
    # Return comprehensive results
    return {
        'output_df': output_df,
        'pvals': pvals,
        'pboot': pboot,
        'alpha_sin': alpha_sin,
        'alpha_mul': alpha_mul,
        'alpha_mulm': alpha_mulm,
        'observed_stats': observed_stats,
        'bootstrap_stats': bootstrap_stats
    }


def build_output(stat, coef, combo, alpha_sin, alpha_mul, pvals, alpha_mulm, select=None):
    """
    Build output DataFrame with all hypotheses.
    
    Creates a row for each outcome × subgroup × comparison combination.
    Based on R implementation in build_output.R
    """
    import pandas as pd
    
    outcomes, subgroups, comparisons = stat.shape
    
    if select is None:
        select = np.ones((outcomes, subgroups, comparisons))
    
    output_rows = []
    result_counter = 0  # Index into flattened vectors (like R)
    
    # Check if alpha_mul and alpha_mulm are flattened vectors as expected
    alpha_mul_flat = alpha_mul.flatten() if alpha_mul.ndim > 1 else alpha_mul
    alpha_mulm_flat = alpha_mulm.flatten() if alpha_mulm.ndim > 1 else alpha_mulm
    
    # Iterate through all hypotheses (like R: i, j, k loops)
    for outcome in range(outcomes):
        for subgroup in range(subgroups):
            for comparison in range(comparisons):
                # Only include if selected
                if select[outcome, subgroup, comparison] == 1:
                    if result_counter >= len(alpha_mul_flat):
                        break
                        
                    row = {
                        'outcome': outcome + 1,  # 1-based indexing
                        'subgroup': subgroup + 1,
                        'comparison': comparison + 1,
                        't1': int(combo[comparison, 0]),
                        't2': int(combo[comparison, 1]),
                        'coefficient': coef[outcome, subgroup, comparison],
                        'test_stat': stat[outcome, subgroup, comparison],
                        'Remark3_2': alpha_sin[outcome, subgroup, comparison],
                        'Thm3_1': alpha_mul_flat[result_counter],      # Use result_counter like R!
                        'Remark3_8': alpha_mulm_flat[result_counter],  # Use result_counter like R!
                        'Bonf': 0.0,  # Will be calculated below
                        'Holm': 0.0   # Will be calculated below
                    }
                    output_rows.append(row)
                    result_counter += 1  # Increment like R
    
    output_df = pd.DataFrame(output_rows)
    
    # Calculate Bonferroni and Holm corrections (like R)
    pvec = output_df['Remark3_2'].values
    nh = len(pvec)
    
    # Bonferroni
    output_df['Bonf'] = np.minimum(1.0, pvec * nh)
    
    # Holm
    order_indices = np.argsort(pvec)
    pvec_sorted = pvec[order_indices]
    holm_adjust = pvec_sorted * np.arange(nh, 0, -1)  # rev(seq_len(nh))
    holm_adjust = np.minimum(np.maximum.accumulate(holm_adjust), 1.0)
    holm_result = np.zeros(nh)
    holm_result[order_indices] = holm_adjust
    output_df['Holm'] = holm_result
    
    return output_df


def remark38_correction(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Apply Remark 3.8 correction (transitivity-corrected testing).
    
    This is a placeholder implementation. The actual method should be based
    on the specific procedure described in List, Shaikh, and Vayalinkal (2023).
    
    Parameters
    ----------
    pvalues : np.ndarray
        Raw p-values
    alpha : float
        Significance level
        
    Returns
    -------
    np.ndarray
        Remark 3.8 corrected p-values  
    """
    # TODO: Implement actual Remark 3.8 procedure
    return pvalues.copy()