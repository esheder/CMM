# CMM

This project is all about finding the correct method to obtain color set diffusion coefficients with a migration based method.
This is an academic project, and it is thus poorly maintained, badly written and is in no way stable or makes any sense.
We might clean this up at some point, if we want to, and then we will actually add things to make this an actual
well maintained open-source code project. Just not now.

## Includes:
1. Two Region Boundary Source Problem
   - Initial mathematical writeup.
   - Python code to try to test the ideas
2. Finite sphere model
   - Initial mathematical writeup
  
## Conclusions thus far:
### Methodology
It seems that the color set problem would require solving the simple homogenized diffusion problem in a similar manner
to the one used by the transport solver. Basically, we want to iteratively look for a diffusion coefficient set that
preserves the quantity of interest.
### Dehomogenization
It is so far unclear whether we want to require preservation of quantities for the dehomogenized flux or for the diffusion
flux. There are arguements to be had for both cases, really. It seems simpler to try preservation of the diffusion flux
first.
