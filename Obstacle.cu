#include "./Obstacle.cuh"
#include "./Constants.cuh"
#include "./FourierHelper.cuh"
#include <cufft.h>
#include <stdio.h>
#include <math.h>

#define OBSMOD (OBSTACLEDIVISOR / ZOOMFACTOR)

Obstacles::Obstacles(FourierHelpers *FH){
	cudaMallocHost((void **) &Obstacle,	 	ByteZahl_Konzentrationen_double);
	cudaMallocHost((void **) &ObsDiff,	 	ByteZahl_Konzentrationen_double);

	cudaMalloc((void**) &GPU_Obstacle,     		ByteZahl_Konzentrationen_double); 
	cudaMalloc((void**) &GPU_ObsDiff,		ByteZahl_Konzentrationen_double);

	//Initialize Obstacle
	int i,j,k;
	switch(OBSTACLETYPE){
		case 0 :
			for(i=0;i<GPZX;i++){
				for(j=0;j<GPZY;j++){
					Obstacle[i * GPZX + j] = 1;																		//Nothing
				}
			}		
			break;
		case 1 :
			for(i=0;i<GPZX;i++){
				for(j=0;j<GPZY;j++){
					Obstacle[i * GPZX + j] = 0.5  * (1 - tanh( (sqrt((i-128)*(i-128)+(j-128)*(j-128)) - 110)/OBSMOD ));										//Circle
				}
			}
			break;
		case 2 :
			for(i=0;i<GPZX;i++){
				for(j=0;j<GPZY;j++){
					Obstacle[i * GPZX + j] = 0.25 * (tanh((i - 10.0) / OBSMOD) - tanh((i - (GPZX - 10.0)) / OBSMOD)) * (tanh((j - 10.0) / OBSMOD) - tanh((j - (GPZY - 10.0)) / OBSMOD));			//Square
				}
			}
			break;
		case 3 :
			for(i=0;i<GPZX;i++){
				for(j=0;j<GPZY;j++){
					Obstacle[i * GPZX + j] = 0.5  * (1 - tanh( (sqrt(abs((i-128)*(i-128)-(j-128)*(j-128))) - 80)/OBSMOD ));									//"Star"
				}
			}
			break;

		case 4 :
		{
			double x0=0, x1 = GPZX/2 * (1-1/sqrt(3)), x2 = GPZX/2 * (1-1/sqrt(4*3)), x3 = GPZX/2 * (1+1/sqrt(4*3)), x4 = GPZX/2 * (1+1/sqrt(3)), x5 = GPZX; 
			double y0 = 0, y1 = GPZY/4, y2 = GPZY/2, y3 = 3*GPZY/4, y4 = GPZY;

			LineDrawer20(Obstacle, x0, y1, x1, y1);
			LineDrawer20(Obstacle, x0, y3, x1, y3);
			LineDrawer20(Obstacle, x2, y0, x3, y0);
			LineDrawer20(Obstacle, x2, y2, x3, y2);
			LineDrawer20(Obstacle, x2, y4, x3, y4);
			LineDrawer20(Obstacle, x4, y1, x5, y1);
			LineDrawer20(Obstacle, x4, y3, x5, y3);
			LineDrawer20(Obstacle, x1, y1, x2, y0);
			LineDrawer20(Obstacle, x1, y1, x2, y2);
			LineDrawer20(Obstacle, x1, y3, x2, y2);
			LineDrawer20(Obstacle, x1, y3, x2, y4);
			LineDrawer20(Obstacle, x3, y0, x4, y1);
			LineDrawer20(Obstacle, x3, y2, x4, y1);
			LineDrawer20(Obstacle, x3, y2, x4, y3);
			LineDrawer20(Obstacle, x3, y4, x4, y3);
			break;
		}
		case 5 :																							//"Channel"
			for(i=0;i<GPZX;i++){
				for(j=0;j<GPZY;j++){
					Obstacle[i * GPZX + j] = 0.5 * (tanh((i - 32.0) / OBSMOD) - tanh((i - (GPZX - 32.0)) / OBSMOD));			
				}
			}
			break;
		case 6 :
		{
			for(i=0;i<GPZX;i++){
				for(j=0;j<GPZX;j++){
					Obstacle[i*GPZX + j] = 0.0;
				}
			}
			int RowLen = ceil(sqrt(CELLCOUNT/2.0));
			int Increment = floor(GPZX / RowLen) - 10;
			double Rad = 40.0/0.95;//((double)GPZX/RowLen - 17.5)/2.0;
			int PX,PY;
			for(j=0;j<GPZX;j++){
				for(k=0;k<GPZY;k++){
					Obstacle[j * GPZX + k] = 0.0;
				}
			}
			for(int i=0;i<CELLCOUNT/2;i++){
				PX = ceil(Rad) + 10 + (i%RowLen) * Increment;
				PY = ceil(Rad) + 10 + (i/RowLen) * Increment;
				for(j=0;j<GPZX;j++){
					for(k=0;k<GPZY;k++){
						Obstacle[j * GPZX + k] = fmax(Obstacle[j * GPZX + k], 0.5 * (1 - tanh((sqrt((j - PX)*(j - PX) + (k - PY)*(k - PY))-0.95*min(Rad, Startradius*2.5))/OBSMOD)));
					}
				}
			} 	
			break;
		}
	}

	cudaMemcpy(GPU_Obstacle,      Obstacle,    ByteZahl_Konzentrationen_double, cudaMemcpyHostToDevice);

	//Calculate Derivative
	cufftExecD2Z(FH->FFTplanReellzuKomplex, GPU_Obstacle,     (cufftDoubleComplex*) FH->GPU_Koeff_Phase);
	Diffusionsterm<<<GPZX,(GPZX+2),0>>>(FH->GPU_Koeff_Phase, FH->GPU_Koeff_Phase);
	cufftExecZ2D(FH->FFTplanKomplexzuReell, (cufftDoubleComplex*) FH->GPU_Koeff_Phase,    GPU_ObsDiff);
	cudaMemcpy(ObsDiff,    GPU_ObsDiff,      ByteZahl_Konzentrationen_double,  cudaMemcpyDeviceToHost);
	printf("Obstacle has been created!\n");
}


Obstacles::~Obstacles(void){
	cudaFreeHost(Obstacle);
	cudaFreeHost(ObsDiff);

	cudaFree(GPU_Obstacle); 
	cudaFree(GPU_ObsDiff);
	printf("Obstacle has been removed!\n");
}






//Determine the distance of a point s to a spline x..y
double DistanceFunction(double s1, double s2, double x1, double x2, double y1, double y2){

double xs1 = x1 - s1, xs2 = x2 - s2, ys1 = y1 - s1, ys2 = y2 - s2, yx1 = y1 - x1, yx2 = y2 - x2;
double lyx = sqrt(yx1 * yx1 + yx2 * yx2);
double a1 = - (xs1 * yx1 + xs2 * yx2) / lyx, a2 = (ys1 * yx1 + ys2 * yx2) / lyx;

	if((a1*a2) >= 0){
		xs1 = xs1 + a1 * yx1 / lyx;
		xs2 = xs2 + a1 * yx2 / lyx;
	}else{
		if(a2 < 0){
			xs1 = ys1;
			xs2 = ys2;
		}
	}

	return sqrt(xs1*xs1 + xs2*xs2);
}

//Draws a line from point p to point q and adds it to the phasefield stored in Phas
void LineDrawer20(double *Phas, double p1, double p2, double q1, double q2){
int i,j;
double Value;
double width = 20.0;

	for(i=0;i<GPZX;i++){
		for(j=0;j<GPZY;j++){
			Value = 0.5 * (1 - tanh( (DistanceFunction((double) j, (double) i, p1, p2, q1, q2) - width)/OBSMOD ));
			if(Value > Phas[i * GPZX + j]){
				Phas[i * GPZX + j] = Value;
			}
		}
	}
}
