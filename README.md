## Preparation.
- Make sure Intel's oneAPI base toolkit and HPC toolkit are installed. 
- Open Makefile and correct the installation path in the INCLUDE variable. 
- Execute ``` $> bash make.sh ``` to compile all the approaches.
- Execute ```bash download_matrix.sh``` to download a test matrix from suitesparse.

## Execution.
Execute the approaches by ``` $> cmd matrix1_name [matrix2_name]```.
For example, ``` $> ./estimate_accu cant ``` shows the predicting accuracy of multiplying the matrix *cant* by itself.


