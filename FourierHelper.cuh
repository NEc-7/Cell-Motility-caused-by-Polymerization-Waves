#ifndef FOURIERHELP__H
#define FOURIERHELP__H 1
#include <cufft.h>

class FourierHelpers
{
	//GPU Helper fields
	public:
	double *GPU_Koeff_Fila, *GPU_Koeff_Phase, *GPU_Koeff_NukAkt, *GPU_Koeff_NukInakt, *GPU_Koeff_PolaX, *GPU_Koeff_PolaX2, *GPU_Koeff_PolaY;
	double *GPU_Koeff_Fila2,*GPU_Koeff_Fila3, *GPU_Koeff_Phase2, *GPU_Koeff_Phase3;
 	cufftHandle FFTplanReellzuKomplex, FFTplanKomplexzuReell;

	//Functions
	FourierHelpers();		//Constructor, allocates GPU memory
	~FourierHelpers();		//Destructor, frees GPU memory

};

void Hilfsgroessen_Erzeugen(double *GPU_FilaA, double *GPU_NukA, double *GPU_NukI, double *GPU_PolX, double *GPU_PolY, double *GPU_Phas,
			    double *GPU_FilaAKoeff, double *GPU_FilaAKoeff2, double *GPU_FilaAKoeff3, double *GPU_NukAKoeff, double *GPU_NukIKoeff, double *GPU_PolXKoeff, double *GPU_PolXKoeff2, 
			    double *GPU_PolYKoeff, double *GPU_PhasKoeff, double *GPU_PhasKoeff2, double *GPU_PhasKoeff3, 
			    double *GPU_FilaAGradX, double *GPU_FilaAGradY, double *GPU_PolDiv, double *GPU_PhasAGradX, double *GPU_PhasAGradY, 
			    double *GPU_FilaADiff, double *GPU_NukADiffA, double *GPU_NukIDiffA, double *GPU_PolXDiff, double *GPU_PolYDiff, double *GPU_PhasDiffA, 
			    cufftHandle Hin, cufftHandle Zurueck);

__global__ void Diffusionsterm(double *Konz_Koeff,double *Ziel_Koeff);
__global__ void Gradiententerme(double *KonzKoeffX, double *KonzKoeffY);
__global__ void Divergenzterm(double *KonzKoeffX, double *KonzKoeffY);


#endif
