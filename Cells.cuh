#ifndef CELLS_GPU
#define CELLS_GPU 1

//needs <cuda.h>, <time.h>, <math.h>, <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cufft.h>
#include "./Constants.cuh"
#include <stdio.h>
#include <unistd.h>
#include <math.h>

class FourierHelpers;

//SingleCell is a Host class. Never define a SingleCell variable on the GPU!
class SingleCell 
{
	public:
	//Host Data
	double *Fila, *NukA, *NukI, *PolX, *PolY, *Phas;
	double *SchwerpunktX, *SchwerpunktY;
	double XStart, YStart;

	//GPU Data
	double *GPU_Fila, *GPU_NukA, *GPU_NukI, *GPU_PolX, *GPU_PolY, *GPU_Phas, *GPU_Area;
	double *GPU_FilaGrdX, *GPU_FilaGrdY, *GPU_PolaDivg, *GPU_PolXDiff, *GPU_PolYDiff, *GPU_NukADiff, *GPU_NukIDiff, *GPU_PhasGrdX, *GPU_PhasGrdY, *GPU_PhasGrdAbs, *GPU_PhasGrdAbsC, *GPU_PhasDiff, *GPU_FilaDiff, *GPU_Curv;

	//Output Files
	FILE *Trajektorie, *CenterActin;

	//Functions
	SingleCell();											//Constructor, does nothing!! Initializer allocates GPU&Host memory!!
	~SingleCell();											//Destructor, does nothing!! So GPU stuff can be defined
	__host__ int Initializer(double startx, double starty, int TrajNum); 	//Allocates GPU&Host memory & Creates a cell with slightly randomized starting values and a radius of StartRadius (constants.cuh) at (startx, starty)
	__host__ void FreeMemory(); 									//Frees GPU&Host memory 
	__host__ void FilaToHost();									//Copies data from GPU to Host
	__host__ void PhasToHost();									// "
	__host__ void NuksToHost();									// "
	__host__ void AllToHost();									// "
	__host__ void SaveCenterOfMass(int TimePoint, double Time);					//Calculates CoM for trajectory
	__host__ int  ZellteilungsCheck(int TimePoint);							//NEEDS SAVECENTEROFMASS FIRST! Checks if the cell has divided (rudimentary)

	__host__ void EulerStep(int SourceOffset, int RelativeTarget, double StepSize);
	__host__ void TimeDerivative(int SourceOffset, int RelativeTarget, SingleCell *GPU_Cells, int CellNum);
	__host__ void TimeDerivativeStep(int SourceOffset, int RelativeTarget, int RelativeOrigin, double StepSize, SingleCell *GPU_Cells, int CellNum);
	__host__ void TotalValues(cublasHandle_t ToDevice, int Offset);					//Calculates the total value of Phasefield, NukA and NukI
	__host__ void SpectralDerivatives(FourierHelpers *FH, int Offset);				//Calculates all needed derivatives
	__host__ double NumericStepError(cublasHandle_t ToHost, double *GPU_Array);			//Calculates the relative error made during the step size control step
	__host__ void Update();										//Updates the values with the ones calculated in the step size control step
	__host__ void StepFilaUpdate();									//Updates the values with the ones calculated in the step size control step

	__host__ double FindMaxLastDeviation(cublasHandle_t ToHost);					//Gets largest d/dt T from the last step.

	__host__ void SaveFila(char path[128]);
	__host__ void SavePhas(char path[128]);

	__host__ void EndSave(char path[128], int Chooser);				//print the entire array. 0=Fila,1=NukA,2=NukI,3=PolX,4=PolY,5=Phas, everything else = nothing
		
};

__global__ void CopyKernel(double *In, double *Out);
__global__ void Vektorsubtraktion(double *Minuend, double *Subtrahend, double *Ziel);

#endif
