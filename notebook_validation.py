# Cell 1: Setup and imports
import sys
import os
from pathlib import Path

# Add the project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Add testthat directory to path for validation script
testthat_dir = project_root / 'tests' / 'testthat'
sys.path.insert(0, str(testthat_dir))

print(f"Project root: {project_root}")
print(f"Testthat dir: {testthat_dir}")
print("Setup complete!")

# Cell 2: Import validation functions
from validate_python_package import validate_mhtexp2_python
print("Validation script imported successfully!")

# Cell 3: Run validation
result = validate_mhtexp2_python(
    n_obs=500, 
    bootstrap=1000,  # Smaller bootstrap for testing
    seed=12345,
    verbose=True
)

# Cell 4: View results
print("\nFirst few rows of comparison:")
print(result['comparison'].head(10))

# Cell 5: Summary statistics
comparison = result['comparison']
print("\nSummary of differences:")
print(f"Max absolute coefficient difference: {abs(comparison['diff_coef']).max():.6f}")
print(f"Max absolute Remark 3.2 difference: {abs(comparison['diff_Remark3_2']).max():.6f}")
print(f"Max absolute Theorem 3.1 difference: {abs(comparison['diff_Thm3_1']).max():.6f}")
print(f"Max absolute Remark 3.8 difference: {abs(comparison['diff_Remark3_8']).max():.6f}")

# Cell 6: Show full comparison (optional)
# Uncomment to see full results
# result['comparison']