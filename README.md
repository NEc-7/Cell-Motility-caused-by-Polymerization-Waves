# Cell-Motility-caused-by-Polymerization-Waves

## Compatibility
The code is meant to be used as a whole as most of the source files depend on one another. A cuda version of at least 5.0 with a compatible GPU is recommended to run the code properly. The code can be compiled by using 

    nvcc -O3 -arch=sm_60 -lcudart -lcublas -lcufft -lm ./*.cu -o ExecutableName
where -arch=sm_x0 should give the appropriate architecture of the used gpu.

## Implementation
* Cell objects defined in Cells.cu /-.cuh contain protein densities, actin polarity fields and a phasefield, together with  arrays to keep track of the center of mass and streams to write output files for individual quantities. The densities are included here in order to minimize the amount of memory accesses. Introducing a "densities" object and merging that with the Dynamics.* files would make it a bit easier to modify the code, but also more costly to compute the results because of an additional reference to said object.
* Dynamics.cu /-.cuh contain functions for calculating Euler steps following the dynamic equations with variable step size. These are then chained into the explicit midpoint rule in main.
* FourierHelpers.cu /-.cuh contain a set of complex coefficients for transforming the quantities of a single cell into fourier space with an FFT. Using only one for all cells saves space. This sets up the calculation of spatial derivatives in fourier space.
* Constants.cuh contains all of the parameters in the form of macros, again to save time when calling parameters on the GPU. They can be implemented on the level of cells in order to modify them during runtime, but this will make the computation again more costly as memory accesses are the most expensive operations on the GPU.

The main loop creates a new cell object with the parameters described in Constants.cuh, allocates all of the memory and initializes the simulation. Then, the explicit midpoint rule is performed until certain thresholds in simulation time are passed. There, the fields are saved as outputs to enable visualizing and analyzing them. 

The step size in each step gets determined by an adaptive step size control that performs two steps with half the size. Then, it compares the largest deviation in the resulting actin density, which contains the largest and most volatile values (order 10e3), with the result obtained with a single full step. If a pre-defined error threshold of 10e-8 is met, the step is saved and the step size increased by 1%. Otherwise, the step is repeated with half the step size.

Once the simulation has finished, the final values for the densities are saved and the memory is freed.

Please note that the folder in which the output is saved has to be created beforehand in the current iteration. By default, it is /Output/Daten/ZZDD where ZZDD is the name of the Iteration defined in Constants.cuh.
