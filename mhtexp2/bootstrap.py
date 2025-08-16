"""Bootstrap inference procedures for mhtexp2.

Implements bootstrap-based statistical inference methods.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional


def quadvariance(x):
    """Calculate variance using n divisor (not n-1)."""
    if x is None or len(x) <= 1:
        return 0
    
    if x.ndim == 2:  # Matrix case
        n = x.shape[0]
        result = np.zeros((x.shape[1], x.shape[1]))
        means = np.mean(x, axis=0)
        
        # Calculate covariance matrix with n divisor
        for i in range(x.shape[1]):
            for j in range(x.shape[1]):
                result[i, j] = np.sum((x[:, i] - means[i]) * (x[:, j] - means[j])) / n
        return result
    else:  # Vector case
        n = len(x)
        mean_x = np.mean(x)
        return np.sum((x - mean_x) ** 2) / n


def runreg(X, y, Xbar, Xvar, pi_2, pi_1, pi_z):
    """Run adjusted regression to estimate treatment effect.
    
    Parameters
    ----------
    X : np.ndarray
        Design matrix: first column is treatment dummy (1 for treatment, 0 for control), 
        rest are controls (optional)
    y : np.ndarray
        Outcome vector
    Xbar : np.ndarray or float
        Covariate means for the subgroup (can be 0)
    Xvar : np.ndarray or float
        Covariance matrix of covariates (can be 0)
    pi_2 : float
        Proportion of treatment group
    pi_1 : float
        Proportion of control group
    pi_z : float
        Proportion of the whole subgroup
        
    Returns
    -------
    dict
        Dictionary with ATE, SE, and other statistics
    """
    D = X[:, 0]
    n = len(y)
    D1_idx = D == 1
    D0_idx = D == 0
    y1 = y[D1_idx]
    y0 = y[D0_idx]
    n1 = len(y1)
    n0 = len(y0)
    
    # With covariates
    if X.shape[1] > 1:
        X1 = X[D1_idx, 1:]
        X0 = X[D0_idx, 1:]
        DX1 = np.column_stack([np.ones(n1), X1])
        DX0 = np.column_stack([np.ones(n0), X0])
    else:
        X1 = np.zeros((n1, 0))
        X0 = np.zeros((n0, 0))
        DX1 = np.ones((n1, 1))
        DX0 = np.ones((n0, 1))
    
    # Run separate regressions
    b1 = np.linalg.solve(DX1.T @ DX1, DX1.T @ y1)
    b0 = np.linalg.solve(DX0.T @ DX0, DX0.T @ y0)
    
    if X.shape[1] > 1:
        bX1 = b1[1:]
        bX0 = b0[1:]
        e1 = y1 - X1 @ bX1
        e0 = y0 - X0 @ bX0
        s1 = quadvariance(e1)
        s0 = quadvariance(e0)
    else:
        bX1 = bX0 = 0
        s1 = quadvariance(y1)
        s0 = quadvariance(y0)
    
    # Adjusted treatment effect estimate
    if isinstance(Xbar, (int, float)) and Xbar == 0:
        ATE = b1[0] - b0[0]
    else:
        ATE = (b1[0] - b0[0]) + np.sum(Xbar * (bX1 - bX0))
    
    # Covariance term
    if X.shape[1] > 1 and not isinstance(Xvar, (int, float)):
        cov_term = (bX1 - bX0).T @ Xvar @ (bX1 - bX0)
    else:
        cov_term = 0
    
    # Compute SE exactly as Stata does
    SE = np.sqrt((1 / pi_2) * s1 + (1 / pi_1) * s0 + (1 / pi_z) * cov_term)
    
    return {
        'ATE': ATE,
        'SE': SE,
        'pi_1': pi_1,
        'pi_2': pi_2,
        'pi_z': pi_z,
        's1': s1,
        's0': s0,
        'cov_term': cov_term
    }


def bootstrap_runreg(
    Y: np.ndarray,
    D: np.ndarray,
    DX: np.ndarray,
    combo: np.ndarray,
    B: int,
    subgroup: Optional[np.ndarray] = None,
    studentized: bool = True,
    idbootmat: Optional[np.ndarray] = None,
    select: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Bootstrap adjusted regression with subgroup support.
    
    Parameters
    ----------
    Y : np.ndarray
        Matrix of outcomes (n x outcomes)
    D : np.ndarray
        Matrix (n x 1) of treatment assignments
    DX : np.ndarray
        Matrix (n x k) of treatment + controls
    combo : np.ndarray
        Matrix of treatment-control pairs
    B : int
        Number of bootstrap samples
    subgroup : np.ndarray, optional
        Vector of subgroup IDs
    studentized : bool, default True
        Whether to divide by SE
    idbootmat : np.ndarray, optional
        Bootstrap resample matrix (n x B)
    select : np.ndarray, optional
        3D array specifying which outcome-subgroup-combo tests to run
        
    Returns
    -------
    dict
        Dictionary with 'stat' (test statistics), 'coef' (ATEs), and 'boot' arrays
    """
    n = Y.shape[0]
    num_outcomes = Y.shape[1]
    num_combos = combo.shape[0]
    
    if subgroup is None:
        subgroup = np.ones(n)
    
    subgroups = np.sort(np.unique(subgroup))
    num_subgroups = len(subgroups)
    
    # Generate bootstrap indices if not provided
    if idbootmat is None:
        idbootmat = np.random.choice(n, size=(n, B), replace=True)
    
    observed_stat = np.full((num_outcomes, num_subgroups, num_combos), np.nan)
    observed_coef = np.full((num_outcomes, num_subgroups, num_combos), np.nan)
    boot_stats = np.full((B, num_outcomes, num_subgroups, num_combos), np.nan)
    
    # First calculate observed statistics
    for i in range(num_outcomes):
        for s, sg in enumerate(subgroups):
            idx_sg = subgroup == sg
            
            Y_sg = Y[idx_sg]
            D_sg = D[idx_sg]
            DX_sg = DX[idx_sg]
            yi = Y_sg[:, i]
            
            for j in range(num_combos):
                if select is not None and select[i, s, j] != 1:
                    continue
                
                t1 = combo[j, 0]
                t2 = combo[j, 1]
                keep = np.isin(D_sg[:, 0], [t1, t2])
                
                if np.sum(keep) < 2:
                    continue
                
                treat = (D_sg[keep, 0] == t2).astype(int)
                cur_X = DX_sg[keep].copy()
                cur_X[:, 0] = treat
                cur_y = yi[keep]
                
                # Calculate controls for FULL subgroup (like Stata)
                if DX_sg.shape[1] > 1:
                    controls_full_sg = DX[idx_sg, 1:]
                    Xbar_sg = np.mean(controls_full_sg, axis=0)
                    Xvar_sg = quadvariance(controls_full_sg)
                else:
                    Xbar_sg = 0
                    Xvar_sg = 0
                
                # Use original dataset for treatment probabilities (like Stata)
                pi_2 = np.sum((subgroup == sg) & (D[:, 0] == t2)) / n
                pi_1 = np.sum((subgroup == sg) & (D[:, 0] == t1)) / n
                pi_z = np.sum(subgroup == sg) / n
                
                est = runreg(cur_X, cur_y, Xbar_sg, Xvar_sg, pi_2, pi_1, pi_z)
                
                stat = abs(est['ATE']) / (est['SE'] if studentized else 1)
                
                observed_coef[i, s, j] = est['ATE']
                observed_stat[i, s, j] = stat
    
    # Now bootstrap
    for b in range(B):
        # First resample the dataset
        boot_idx = idbootmat[:, b]
        Y_boot = Y[boot_idx]
        D_boot = D[boot_idx]
        DX_boot = DX[boot_idx]
        sub_boot = subgroup[boot_idx]
        
        # Process each outcome, subgroup, and comparison within this resampled dataset
        for i in range(num_outcomes):
            for s, sg in enumerate(subgroups):
                idx_sg = sub_boot == sg
                
                if np.sum(idx_sg) < 2:
                    continue
                
                # Calculate controls for FULL bootstrap subgroup
                if DX_boot.shape[1] > 1:
                    controls_boot_sg = DX_boot[idx_sg, 1:]
                    Xbar_b = np.mean(controls_boot_sg, axis=0)
                    Xvar_b = quadvariance(controls_boot_sg)
                else:
                    Xbar_b = 0
                    Xvar_b = 0
                
                # Subset the bootstrap sample for this outcome and subgroup
                Y_sg = Y_boot[idx_sg]
                D_sg = D_boot[idx_sg]
                DX_sg = DX_boot[idx_sg]
                
                for j in range(num_combos):
                    if select is not None and select[i, s, j] != 1:
                        continue
                    
                    t1 = combo[j, 0]
                    t2 = combo[j, 1]
                    
                    keep = np.isin(D_sg[:, 0], [t1, t2])
                    
                    if np.sum(keep) < 2:
                        continue
                    
                    treat = (D_sg[keep, 0] == t2).astype(int)
                    cur_X = DX_sg[keep].copy()
                    cur_X[:, 0] = treat
                    cur_y = Y_sg[keep, i]
                    
                    # Use original subgroup but bootstrap treatment (like Stata)
                    pi_2 = np.sum((subgroup == sg) & (D_boot[:, 0] == t2)) / n
                    pi_1 = np.sum((subgroup == sg) & (D_boot[:, 0] == t1)) / n
                    pi_z = np.sum(subgroup == sg) / n
                    
                    est_b = runreg(cur_X, cur_y, Xbar_b, Xvar_b, pi_2, pi_1, pi_z)
                    
                    # Fix test statistic to match Stata
                    stat_b = abs(est_b['ATE'] - observed_coef[i, s, j]) / (est_b['SE'] if studentized else 1)
                    
                    boot_stats[b, i, s, j] = stat_b
        
        if (b + 1) % 1000 == 0:
            print(f"Bootstrap progress: {b + 1}/{B}")
    
    return {
        'stat': observed_stat,
        'coef': observed_coef,
        'boot': boot_stats,
        'combo': combo
    }


def bootstrap_inference(
    Y: np.ndarray,
    treatment: np.ndarray,
    controls: Optional[np.ndarray] = None,
    subgroup: Optional[np.ndarray] = None,
    combo: str = "pairwise",
    n_bootstrap: int = 3000,
    studentized: bool = True
) -> Dict[str, Any]:
    """
    Perform bootstrap inference for multiple hypothesis testing.
    
    This is the main bootstrap function that replicates the R implementation.
    
    Parameters
    ----------
    Y : np.ndarray
        Outcome variables matrix (n_observations x n_outcomes)
    treatment : np.ndarray
        Treatment assignment vector
    controls : np.ndarray, optional
        Control variables
    subgroup : np.ndarray, optional
        Subgroup identifiers
    combo : str, default "pairwise"
        Comparison type ("pairwise" or "treatmentcontrol")
    n_bootstrap : int, default 3000
        Number of bootstrap replications
    studentized : bool, default True
        Whether to studentize test statistics
        
    Returns
    -------
    dict
        Bootstrap results including test statistics and confidence intervals
    """
    
    # Prepare data matrices
    n_obs = Y.shape[0]
    D = treatment.reshape(-1, 1)  # Treatment matrix
    
    # Build DX matrix (treatment + controls)
    if controls is not None:
        DX = np.column_stack([treatment, controls])
    else:
        DX = treatment.reshape(-1, 1)
    
    # Build comparison matrix
    unique_treatments = np.unique(treatment)
    if combo == "pairwise":
        # All pairwise comparisons
        combo_pairs = []
        for i, t1 in enumerate(unique_treatments):
            for t2 in unique_treatments[i+1:]:
                combo_pairs.append([t1, t2])
        combo_matrix = np.array(combo_pairs)
    else:  # treatmentcontrol
        # All treatments vs control (assuming 0 is control)
        control_val = 0 if 0 in unique_treatments else unique_treatments[0]
        combo_pairs = []
        for t in unique_treatments:
            if t != control_val:
                combo_pairs.append([control_val, t])
        combo_matrix = np.array(combo_pairs)
    
    # Generate bootstrap sample indices
    idbootmat = np.random.choice(n_obs, size=(n_obs, n_bootstrap), replace=True)
    
    # Run bootstrap
    results = bootstrap_runreg(
        Y=Y,
        D=D,
        DX=DX,
        combo=combo_matrix,
        B=n_bootstrap,
        subgroup=subgroup,
        studentized=studentized,
        idbootmat=idbootmat
    )
    
    # Add combo matrix to results
    results['combo'] = combo_matrix
    
    return results