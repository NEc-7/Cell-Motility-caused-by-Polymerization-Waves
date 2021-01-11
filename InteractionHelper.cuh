#ifndef INTACTHELPER__H
#define INTACTHELPER__H 1

class SingleCell;

class InterActionHelper{
	public :
	double *GPU_DeltHelper, *GPU_DegradHelper;

	InterActionHelper();
	~InterActionHelper();	
	__host__ void Update(int StartOffset, SingleCell* CellArray);
};

__global__ void UpdateKernel(int StartOffset, SingleCell* CellArray, double *DeltHelper, double *DegradHelper);


#endif
