"""
Core functionality for mhtexp2 package.

Main implementation of multiple hypothesis testing procedures.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any, Tuple
from .bootstrap import bootstrap_inference
from .corrections import apply_corrections


def mhtexp2(
    Y: Union[np.ndarray, pd.DataFrame],
    treatment: Union[np.ndarray, pd.Series],
    controls: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    subgroup: Optional[Union[np.ndarray, pd.Series]] = None,
    combo: str = "pairwise",
    bootstrap: int = 3000,
    studentized: bool = True,
    transitivity_check: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Multiple hypothesis testing for experimental economics.
    
    Parameters
    ----------
    Y : array-like or DataFrame
        Matrix of outcome variables (n_observations x n_outcomes)
    treatment : array-like or Series
        Treatment assignment vector
    controls : array-like or DataFrame, optional
        Control variables for regression adjustment
    subgroup : array-like or Series, optional
        Subgroup identifiers for stratified analysis
    combo : {"pairwise", "treatmentcontrol"}, default "pairwise"
        Type of comparisons to perform
    bootstrap : int, default 3000
        Number of bootstrap replications
    studentized : bool, default True
        Whether to studentize test statistics
    transitivity_check : bool, default False
        Whether to apply transitivity corrections
    **kwargs
        Additional parameters
        
    Returns
    -------
    dict
        Dictionary containing test results, p-values, and confidence intervals
        
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from mhtexp2 import mhtexp2
    >>> 
    >>> # Generate sample data
    >>> n = 100
    >>> Y = np.random.randn(n, 3)  # 3 outcome variables
    >>> treatment = np.random.choice([0, 1], n)
    >>> 
    >>> # Run analysis
    >>> results = mhtexp2(Y, treatment, bootstrap=1000)
    >>> print(results['pvalues'])
    """
    
    # Input validation
    Y = _validate_input_data(Y, "Y")
    treatment = _validate_input_data(treatment, "treatment")
    
    if controls is not None:
        controls = _validate_input_data(controls, "controls")
    
    if subgroup is not None:
        subgroup = _validate_input_data(subgroup, "subgroup")
    
    # Check dimensions
    n_obs, n_outcomes = Y.shape
    if len(treatment) != n_obs:
        raise ValueError("Length of treatment must match number of observations in Y")
    
    # Run bootstrap inference
    bootstrap_results = bootstrap_inference(
        Y=Y,
        treatment=treatment,
        controls=controls,
        subgroup=subgroup,
        combo=combo,
        n_bootstrap=bootstrap,
        studentized=studentized
    )
    
    # Extract results from bootstrap
    observed_stats = bootstrap_results['stat']
    observed_coef = bootstrap_results['coef']
    boot_stats = bootstrap_results['boot']
    combo_matrix = bootstrap_results['combo']
    
    # Calculate raw p-values from bootstrap distribution
    from .corrections import calculate_pvals, validate_statistics_with_external_data
    
    # Calculate 1-p values from bootstrap
    pvalues_raw = calculate_pvals(observed_stats, boot_stats)
    
    # Use the comprehensive correction method that handles all procedures
    correction_results = validate_statistics_with_external_data(
        observed_stats=observed_stats,
        bootstrap_stats=boot_stats,
        observed_coef=observed_coef,
        combo=combo_matrix,
        select=None,
        transitivitycheck=transitivity_check
    )
    
    results = {
        'test_statistics': observed_stats,
        'coefficients': observed_coef,
        'pvalues_raw': pvalues_raw,
        'pvalues_corrected': {
            'remark32': correction_results['alpha_sin'],
            'theorem31': correction_results['alpha_mul'],
            'remark38': correction_results['alpha_mulm']
        },
        'bootstrap_results': bootstrap_results,
        'correction_results': correction_results,
        'output_table': correction_results['output_df'],
        'summary': {
            'n_observations': n_obs,
            'n_outcomes': n_outcomes,
            'n_treatments': len(np.unique(treatment)),
            'bootstrap_replications': bootstrap,
            'studentized': studentized,
            'transitivity_check': transitivity_check
        }
    }
    
    return results


def _validate_input_data(data: Union[np.ndarray, pd.DataFrame, pd.Series], 
                        name: str) -> np.ndarray:
    """Validate and convert input data to numpy array."""
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.values
    elif not isinstance(data, np.ndarray):
        data = np.array(data)
    
    if data.size == 0:
        raise ValueError(f"{name} cannot be empty")
    
    return data