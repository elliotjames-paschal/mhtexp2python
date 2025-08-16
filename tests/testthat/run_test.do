* Simple test script to run export_bootstrap.do
* This generates sample data and runs the export

clear all
set more off

* Files will be saved in current working directory
* Ensure you run this from the tests/testthat directory

* Generate test data matching R validate_package.R approach
set obs 1000  
* No seed set - truly random data each run

* Create treatment variable (0=control, 1=treatment1, 2=treatment2)
gen treatment = floor(runiform() * 3)

* Create subgroup variable (1 or 2)
gen subgroup = floor(runiform() * 2) + 1

* Create control variables
gen x1 = rnormal(0, 1)
gen x2 = rnormal(0, 1)

* Create outcome variables with treatment effects
* y1: treatment effects of 0, 0.05, 0.1 for treatments 0, 1, 2
gen y1 = rnormal(0, 1) + 0.1 * cond(treatment == 1, 0.5, cond(treatment == 2, 1, 0))

* y2: treatment effects of 0, 0.03, 0.08 for treatments 0, 1, 2  
gen y2 = rnormal(0, 1) + 0.1 * cond(treatment == 1, 0.3, cond(treatment == 2, 0.8, 0))

* Show data summary
tab treatment
tab subgroup
tab treatment subgroup
summarize y1 y2 x1 x2

* Delete any existing CSV files first
capture erase "abregact.csv"
capture erase "abregboot.csv" 
capture erase "pact.csv"
capture erase "pboot.csv"
capture erase "stata_results.csv"

* Load the export_bootstrap program
do "export_bootstrap.do"

* Run the command with controls and subgroups - this will create CSV files in current directory
mhtexp2 y1 y2, treatment(treatment) controls(x1 x2) subgroup(subgroup) combo("pairwise") bootstrap(5000)

* List created files
display "Files created in: " c(pwd)
! ls -la *.csv

display "Done! CSV files ready for Python validation test."