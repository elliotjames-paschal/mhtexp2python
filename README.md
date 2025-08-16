# mhtexp2python

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python implementation of multiple hypothesis testing procedures for experimental economics with exact replication of mhtexp2 functionality.

## Overview

`mhtexp2python` provides robust multiple hypothesis testing methods specifically designed for experimental economics research. This package replicates the functionality of the original Stata `mhtexp2` package by List, Shaikh, and Vayalinkal (2023), offering family-wise error rate control and various p-value correction procedures for analyzing experimental data with multiple outcomes.

## Installation

```bash
# Install from GitHub
pip install git+https://github.com/yourusername/mhtexp2python.git

# Or clone and install from source
git clone https://github.com/yourusername/mhtexp2python.git
cd mhtexp2python
pip install -e .
```

### Requirements

- Python >= 3.8
- numpy
- pandas  
- scipy
- statsmodels

## Basic Usage

```python
import numpy as np
import pandas as pd
from mhtexp2 import mhtexp2

# Generate sample experimental data
np.random.seed(42)
n_obs = 200

# Create outcome variables
Y = np.random.randn(n_obs, 2)  # 2 outcomes
# Add treatment effects
Y[100:, 0] += 0.5  # Treatment effect on outcome 1

# Treatment assignment (0=control, 1=treatment1, 2=treatment2)
treatment = np.concatenate([np.zeros(67), np.ones(67), np.full(66, 2)])

# Control variables
controls = np.random.randn(n_obs, 2)

# Subgroup indicator
subgroup = np.random.choice([1, 2], n_obs)

# Run multiple hypothesis testing
results = mhtexp2(
    Y=Y,
    treatment=treatment,
    controls=controls,
    subgroup=subgroup,
    combo="pairwise",
    bootstrap=3000,
    studentized=True,
    transitivity_check=True
)

# View results DataFrame
print(results)
```

## Parameters

- `Y`: Matrix/DataFrame of outcome variables (n_observations × n_outcomes)
- `treatment`: Treatment assignment vector
- `controls`: Optional control variables for regression adjustment
- `subgroup`: Optional subgroup identifiers for stratified analysis  
- `combo`: Comparison type - "pairwise" or "treatmentcontrol" (default: "pairwise")
- `bootstrap`: Number of bootstrap replications (default: 3000)
- `studentized`: Whether to studentize test statistics (default: True)
- `transitivity_check`: Apply transitivity corrections (default: False)

## Output

The function returns a pandas DataFrame with the following columns:

- `outcome`: Outcome variable index
- `subgroup`: Subgroup index  
- `comparison`: Treatment comparison index
- `t1`, `t2`: Treatment pair being compared
- `coefficient`: Treatment effect estimate
- `test_stat`: Test statistic
- `Remark3_2`: Single hypothesis adjusted p-values
- `Thm3_1`: Multiple hypothesis adjusted p-values (Theorem 3.1)
- `Remark3_8`: Transitivity-corrected p-values (Remark 3.8)
- `Bonf`: Bonferroni-corrected p-values
- `Holm`: Holm step-down corrected p-values

## Statistical Methods

This package implements the multiple hypothesis testing framework from List, Shaikh, and Vayalinkal (2023):

- **Remark 3.2**: Single hypothesis testing procedure
- **Theorem 3.1**: Multiple hypothesis testing with family-wise error rate control
- **Remark 3.8**: Transitivity-corrected multiple testing procedure
- **Bonferroni correction**: Conservative multiple testing adjustment
- **Holm procedure**: Step-down multiple testing method

All procedures use bootstrap-based inference to provide robust statistical inference without strong distributional assumptions.

## Validation

The package includes validation tools to compare results with the original Stata implementation:

```bash
cd tests/testthat
python bootstrap_test_with_stata.py
```

This requires Stata to be installed and generates comparison tables showing exact replication of Stata results.

## Package Structure

```
mhtexp2python/
├── mhtexp2/
│   ├── __init__.py          # Package initialization
│   ├── core.py              # Main mhtexp2() function
│   ├── bootstrap.py         # Bootstrap inference procedures
│   ├── corrections.py       # P-value correction methods
│   └── utils.py             # Helper functions
├── tests/testthat/          # Validation tests
├── examples/                # Usage examples
└── docs/                    # Documentation
```

## Citation

If you use this package in your research, please cite:

```bibtex
@article{list2023multiple,
  title={Multiple Testing with Covariate Adjustment in Experimental Economics},
  author={List, John A and Shaikh, Azeem M and Vayalinkal, Jaap P},
  journal={Experimental Economics},
  year={2023},
  publisher={Springer}
}
```

## References

- List, J. A., Shaikh, A. M., & Vayalinkal, J. P. (2023). Multiple Testing with Covariate Adjustment in Experimental Economics. *Experimental Economics*.
- Original Stata implementation: `mhtexp2`
- R implementation: [mhtexp2r](https://github.com/elliotjames-paschal/mhtexp2r)

## License

This project is licensed under the MIT License.

## Support

For questions, bug reports, or feature requests, please open an issue on the [GitHub repository](https://github.com/yourusername/mhtexp2python/issues).