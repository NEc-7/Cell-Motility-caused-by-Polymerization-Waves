#include "./FourierHelper.cuh"
#include "./Constants.cuh"
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

FourierHelpers::FourierHelpers(void){
	cudaMalloc((void**) &GPU_Koeff_Fila, 		2*ByteZahl_Koeff_Frequenz_double);  /* 2*, da sich die Koeffizienten in Real- und Imaginärteil aufspalten */
	cudaMalloc((void**) &GPU_Koeff_Fila2, 		2*ByteZahl_Koeff_Frequenz_double);	 //Mehrere, da für eine Größe u.U. verschiedene Ableitungen berechnet werden müssen
	cudaMalloc((void**) &GPU_Koeff_Fila3, 		2*ByteZahl_Koeff_Frequenz_double);
	cudaMalloc((void**) &GPU_Koeff_NukAkt, 		2*ByteZahl_Koeff_Frequenz_double);
	cudaMalloc((void**) &GPU_Koeff_NukInakt,        2*ByteZahl_Koeff_Frequenz_double);
	cudaMalloc((void**) &GPU_Koeff_PolaX, 		2*ByteZahl_Koeff_Frequenz_double);
	cudaMalloc((void**) &GPU_Koeff_PolaX2, 		2*ByteZahl_Koeff_Frequenz_double);
	cudaMalloc((void**) &GPU_Koeff_PolaY, 		2*ByteZahl_Koeff_Frequenz_double);
	cudaMalloc((void**) &GPU_Koeff_Phase, 		2*ByteZahl_Koeff_Frequenz_double);
	cudaMalloc((void**) &GPU_Koeff_Phase2, 		2*ByteZahl_Koeff_Frequenz_double);
	cudaMalloc((void**) &GPU_Koeff_Phase3, 		2*ByteZahl_Koeff_Frequenz_double);

	cufftPlan2d(&FFTplanKomplexzuReell, GITTERPUNKTZAHLX, GITTERPUNKTZAHLY, CUFFT_Z2D);
	cufftPlan2d(&FFTplanReellzuKomplex, GITTERPUNKTZAHLX, GITTERPUNKTZAHLY, CUFFT_D2Z);
	printf("FourierHelper created!\n");
}


FourierHelpers::~FourierHelpers(void){
	cudaFree(GPU_Koeff_Fila);
	cudaFree(GPU_Koeff_Fila2);
	cudaFree(GPU_Koeff_Fila3);
	cudaFree(GPU_Koeff_NukAkt);
	cudaFree(GPU_Koeff_NukInakt);
	cudaFree(GPU_Koeff_PolaX);
	cudaFree(GPU_Koeff_PolaX2);
	cudaFree(GPU_Koeff_PolaY);
	cudaFree(GPU_Koeff_Phase);
	cudaFree(GPU_Koeff_Phase2);
	cudaFree(GPU_Koeff_Phase3);

	cufftDestroy(FFTplanReellzuKomplex);
	cufftDestroy(FFTplanKomplexzuReell);
	printf("FourierHelper removed!\n");
}

void Hilfsgroessen_Erzeugen(double *GPU_FilaA, double *GPU_NukA, double *GPU_NukI, double *GPU_PolX, double *GPU_PolY, double *GPU_Phas,
			    double *GPU_FilaAKoeff, double *GPU_FilaAKoeff2, double *GPU_FilaAKoeff3, double *GPU_NukAKoeff, double *GPU_NukIKoeff, double *GPU_PolXKoeff, double *GPU_PolXKoeff2, 
			    double *GPU_PolYKoeff, double *GPU_PhasKoeff, double *GPU_PhasKoeff2, double *GPU_PhasKoeff3, 
			    double *GPU_FilaAGradX, double *GPU_FilaAGradY, double *GPU_PolDiv, double *GPU_PhasAGradX, double *GPU_PhasAGradY, 
			    double *GPU_FilaADiff, double *GPU_NukADiffA, double *GPU_NukIDiffA, double *GPU_PolXDiff, double *GPU_PolYDiff, double *GPU_PhasDiffA, 
			    cufftHandle Hin, cufftHandle Zurueck){

			 
			//HinTrafo A
			cufftExecD2Z(Hin, GPU_FilaA,    (cufftDoubleComplex*) GPU_FilaAKoeff);
			#if PHASEFIELDFLAG > 0
				cufftExecD2Z(Hin, GPU_Phas,     (cufftDoubleComplex*) GPU_PhasKoeff);
			#endif
			cufftExecD2Z(Hin, GPU_NukA,     (cufftDoubleComplex*) GPU_NukAKoeff);
			cufftExecD2Z(Hin, GPU_NukI,     (cufftDoubleComplex*) GPU_NukIKoeff);
			cufftExecD2Z(Hin, GPU_PolX,     (cufftDoubleComplex*) GPU_PolXKoeff);
			cufftExecD2Z(Hin, GPU_PolY,     (cufftDoubleComplex*) GPU_PolYKoeff);

			//DiffTerme und Gradienten/Divergenzen berechnen	
//			Diffusionsterm<<<GPZX,(GPZY+2),0>>>(GPU_FilaAKoeff,GPU_FilaAKoeff2);
//			Diffusionsterm<<<GPZX,(GPZY+2),0>>>(GPU_PolXKoeff,GPU_PolXKoeff2);

			Diffusionsterm<<<GPZX,(GPZY+2),0>>>(GPU_NukAKoeff,GPU_NukAKoeff);
			Diffusionsterm<<<GPZX,(GPZY+2),0>>>(GPU_NukIKoeff,GPU_NukIKoeff);

			Gradiententerme<<<GPZX,(GPZY+2)>>1,0>>>(GPU_FilaAKoeff,GPU_FilaAKoeff3);		//Koeff 3 ist zum Aufruf noch leer, dient aber als zusätzlicher Speicher für die y-Ableitung
														//!!andere Lösung für dx px und dy py erforderlich (wenn man sie getrennt benötigt)!! Geht nur für T und Psi.
			#if PHASEFIELDFLAG > 0
				Diffusionsterm<<<GPZX,(GPZY+2),0>>>(GPU_PhasKoeff,GPU_PhasKoeff2);
				Gradiententerme<<<GPZX,(GPZY+2)>>1,0>>>(GPU_PhasKoeff,GPU_PhasKoeff3);
			#endif

			//RückTrafo ALLER so berechneten Hilfsgrößen
			cufftExecZ2D(Zurueck, (cufftDoubleComplex*) GPU_FilaAKoeff,    GPU_FilaAGradX );
//			cufftExecZ2D(Zurueck, (cufftDoubleComplex*) GPU_PolXKoeff2,    GPU_PolXDiff  );
//			cufftExecZ2D(Zurueck, (cufftDoubleComplex*) GPU_FilaAKoeff2,   GPU_FilaADiff  );
			cufftExecZ2D(Zurueck, (cufftDoubleComplex*) GPU_NukAKoeff,     GPU_NukADiffA  );
			cufftExecZ2D(Zurueck, (cufftDoubleComplex*) GPU_NukIKoeff,     GPU_NukIDiffA  );
			cufftExecZ2D(Zurueck, (cufftDoubleComplex*) GPU_FilaAKoeff3,   GPU_FilaAGradY );
			#if PHASEFIELDFLAG > 0
				cufftExecZ2D(Zurueck, (cufftDoubleComplex*) GPU_PhasKoeff,     GPU_PhasAGradX );
				cufftExecZ2D(Zurueck, (cufftDoubleComplex*) GPU_PhasKoeff3,    GPU_PhasAGradY );
				cufftExecZ2D(Zurueck, (cufftDoubleComplex*) GPU_PhasKoeff2,    GPU_PhasDiffA  );
			#endif

			Divergenzterm<<<GPZX,(GPZY+2)>>1,0>>>(GPU_PolXKoeff,GPU_PolYKoeff);
			cufftExecZ2D(Zurueck, (cufftDoubleComplex*) GPU_PolXKoeff,     GPU_PolDiv     );
//			Diffusionsterm<<<GPZX,(GPZY+2),0>>>(GPU_PolYKoeff,GPU_PolYKoeff);
//			cufftExecZ2D(Zurueck, (cufftDoubleComplex*) GPU_PolYKoeff,     GPU_PolYDiff   );



}


//Diffusionsterm berechnen
__global__ void Diffusionsterm(double *Konz_Koeff,double *Ziel_Koeff){
	int ID = threadIdx.x + blockIdx.x * blockDim.x;
	int i = threadIdx.x>>1, j = blockIdx.x;
	if(j>(gridDim.x>>1)){j-=gridDim.x;}
	double Faktor = 2.0 * PI / VAR_SysL;
	Faktor *= Faktor * (i * i + j * j) / RASTERGROESSE;
	Ziel_Koeff[ID] = - Faktor * Konz_Koeff[ID];
};

//Gradienten berechnen, es werden nur GPZX/2 + 1 Threads gestartet, um Real- und Imaginärteile gut verrechnen zu können; beide Felder erhalten aktualisierte Werte
//2 Eingaben, da diese aus einem Feld erzeugt werden; Dies garantiert einen Abschluss des Kopiervorgangs vor der Berechnung der Gradienten 
//Beide Eingabefelder sind identisch, es werden 2 Ausgabefelder benötigt
__global__ void Gradiententerme(double *KonzKoeffX, double *KonzKoeffY){
	int IDX = threadIdx.x, IDY = blockIdx.x;
	int ID = (IDX<<1) + IDY * (GPZX + 2);	//GPZX, da eine Reihe GPZX + 2 Einträge hat
	double AltRe = KonzKoeffX[ID], AltIm = KonzKoeffX[ID + 1];
	if(IDY>(gridDim.x>>1)){IDY-=gridDim.x;}
	double Faktor = 2.0 * PI / (VAR_SysL * RASTERGROESSE), Faktorx = Faktor * IDX, Faktory = Faktor * IDY;

	KonzKoeffX[ID] = - Faktorx * AltIm;
	KonzKoeffY[ID] = - Faktory * AltIm;
	KonzKoeffX[ID + 1] = AltRe * Faktorx;
	KonzKoeffY[ID + 1] = AltRe * Faktory;


};

//Divergenz berechnen, auch hier werden nur GPZX/2 + 1 Threads gestartet; es genügt nur ein Ausgabefeld (hier das erste Argument)
//Die beiden Eingabefelder sind verschieden (X und Y-Komponente der Polarisation)
__global__ void Divergenzterm(double *KonzKoeffX, double *KonzKoeffY){
	int IDX = threadIdx.x, IDY = blockIdx.x;
	int ID = (IDX<<1) + IDY * (GPZX + 2);	//GPZX, da eine Reihe GPZX + 2 Einträge hat

	if(IDY>(gridDim.x>>1)){IDY-=gridDim.x;}
	double Faktor = 2.0 * PI / (VAR_SysL * RASTERGROESSE),  Faktorx = Faktor * IDX, Faktory = Faktor * IDY;

	double AltXRe = KonzKoeffX[ID];

	//Die Ableitungen in x- und y-Richtung werden aufaddiert, weswegen hier jeweils 2 Summanden auftreten
	KonzKoeffX[ID] = - (KonzKoeffX[ID + 1] * Faktorx + KonzKoeffY[ID + 1] * Faktory);
	KonzKoeffX[ID + 1] = (AltXRe * Faktorx + KonzKoeffY[ID] * Faktory);
};
