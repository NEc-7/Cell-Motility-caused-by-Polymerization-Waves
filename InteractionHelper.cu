#include "./InteractionHelper.cuh"
#include "./Cells.cuh"
#include "./Constants.cuh"
#include <cuda.h>
#include <cuda_runtime.h>

InterActionHelper::InterActionHelper(void){
	cudaMalloc((void**) &GPU_DeltHelper,    CELLCOUNT * ByteZahl_Konzentrationen_double); 
	cudaMalloc((void**) &GPU_DegradHelper,  CELLCOUNT * ByteZahl_Konzentrationen_double); 
	printf("Created InteractionHelper!\n");
}

InterActionHelper::~InterActionHelper(void){
	cudaFree(GPU_DeltHelper); 
	cudaFree(GPU_DegradHelper); 
	printf("Destroyed InteractionHelper!\n");
}

void InterActionHelper::Update(int StartOffset, SingleCell* CellArray){
	UpdateKernel<<<GPZX,GPZY,0>>>(StartOffset * RASTERGROESSE, CellArray, GPU_DeltHelper, GPU_DegradHelper);
}



__global__ void UpdateKernel(int StartOffset, SingleCell* CellArray, double *DeltHelper, double *DegradHelper){
	int i, j, ID = threadIdx.x + blockIdx.x * blockDim.x;
	double Helper, DeltaProd[CELLCOUNT], OverlapProd[CELLCOUNT];
	for(i=0;i<CELLCOUNT;i++){
		DeltaProd[i]   = 1;
		OverlapProd[i] = 0;
	}

	for(i=0;i<CELLCOUNT;i++){
		Helper = CellArray[i].GPU_Phas[ID + StartOffset];
		for(j=0;j<CELLCOUNT;j++){
			if(i != j){
				DeltaProd[j] *= (1 - Helper);
				OverlapProd[j] += Helper * Helper;
			}
		}
	}
	for(j=0;j<CELLCOUNT;j++){
		DeltHelper[ID + j * RASTERGROESSE]   = DeltaProd[j];
		DegradHelper[ID + j * RASTERGROESSE] = OverlapProd[j];
	}
};
