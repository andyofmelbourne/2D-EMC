# 2D-EMC

## Branches
- simple: uses single core python on minimal working examples with hard coded parameters. 
- main: simple + mpi, input and output parameters and structured code. 
- optimise: optimised with opencl, code becomes unreadable.
- optimise-fast: further optimised and with many comprimises to make the code fast, even if the results are dodgy.

## Notation
- K[d, i] photon counts for frame d at pixel i
- W[t, r, i] estimated photon counts for class t, rotation angle r and pixel i
- w[d] fluence estimates
- B[l, i] estimated photon counts for background class l
- b[d, l] estimated background weights
- T[d, t, r, i] = w[d] W[t, r, i] + b[d, l] B[l, i] estimated model for frame
- P[t, d, r] = sum_i 

## 
