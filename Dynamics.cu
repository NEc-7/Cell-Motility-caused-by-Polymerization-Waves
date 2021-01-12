#include "./Dynamics.cuh"
#include "./Cells.cuh"
#include "./Constants.cuh"
#include <cuda.h>
#include <cuda_runtime.h>

//If this flag is set, normalize the phasefield gradient vector by 1+its norm. If not, don't normalize (/(1+0))
#if MEMBGRADIENTNORMALIZER > 0
	#define PHAS_SQRT (4 + sqrt(PhGradX * PhGradX + PhGradY * PhGradY))
#else
	#define PHAS_SQRT 0
#endif

//Offset gives the increment for saving results. +1 means 1 full (GPZX x GPZY, or 256x256) array of double values. Values for the individual steps of the midpoint rule: k1->1, k2->3, k3->4, k4->5
__global__ void ZeitDiff(int Startvalue, int Offset, double *Fila, double *NukA, double *NukI, double *PolX, double *PolY, double *Phas, 
			 double *Area, double *FilaGradX, double *FilaGradY, double *PolaDiv, double *PhasGradX, double *PhasGradY, 
		  	 double *FilaDiff, double *NukADiff, double *NukIDiff, double *PolXDiff, double *PolYDiff, double *PhasDiff, double *Curv,
			 SingleCell *Cells, int CellNum){
	int ID = threadIdx.x + blockIdx.x * blockDim.x;
	int IDZiel = ID + (Offset << 16) + Startvalue;
	double PhDiff = (PHASEFIELDFLAG ? PhasDiff[ID] : 0), PhGradX = (PHASEFIELDFLAG ? PhasGradX[ID] : 0), PhGradY = (PHASEFIELDFLAG ? PhasGradY[ID] : 0);

	ID = ID + Startvalue;
	double FilaAlt = Fila[ID], NukAAlt = NukA[ID], NukIAlt = NukI[ID], PhasAlt = (PHASEFIELDFLAG ? Phas[ID] : 1), PolXAlt = PolX[ID], PolYAlt = PolY[ID];
	double NukSummand = PhasAlt * (VAR_wd * (FilaAlt/* + VAR_Acti * (PolXAlt * PolXAlt + PolYAlt * PolYAlt)*/) * NukAAlt - NukIAlt * (1 + VAR_w0 * NukAAlt * NukAAlt));
	ID = ID - Startvalue;

	Fila[IDZiel]  =  /*VAR_DiAkt    * (PhasAlt * FilaDiff[ID] - FilaAlt * PhDiff)*/	  	  + PhasAlt * (VAR_alpha * NukAAlt - VAR_va * PolaDiv[ID] - VAR_kd * FilaAlt) - FilaAlt;
	#if NUCLEATORDEGRADFLAG > 0
		NukA[IDZiel]  = (VAR_DiNa *   (PhasAlt * NukADiff[ID] - NukAAlt * PhDiff)	  - NukSummand + (PhasAlt - 1) * NukAAlt);
		NukI[IDZiel]  = (	      (PhasAlt * NukIDiff[ID] - NukIAlt * PhDiff)	  + NukSummand + (PhasAlt - 1) * NukIAlt + 15 * PhasAlt * (TotalNuk - *(Area+1) - *(Area+2)));
	#else
		NukA[IDZiel]  = (VAR_DiNa *   (PhasAlt * NukADiff[ID] - NukAAlt * PhDiff)	  - NukSummand);
		NukI[IDZiel]  = (	      (PhasAlt * NukIDiff[ID] - NukIAlt * PhDiff)	  + NukSummand);
	#endif
	PolX[IDZiel]  =  /*VAR_DiAkt    * (PhasAlt * PolXDiff[ID] - PolXAlt * PhDiff)*/ 	  - PhasAlt * (VAR_kd * PolXAlt + VAR_va * FilaGradX[ID]) - PolXAlt;
	PolY[IDZiel]  =  /*VAR_DiAkt    * (PhasAlt * PolYDiff[ID] - PolYAlt * PhDiff)*/	  	  - PhasAlt * (VAR_kd * PolYAlt + VAR_va * FilaGradY[ID]) - PolYAlt;
	#if PHASEFIELDFLAG > 0
		Phas[IDZiel]  = (VAR_DiPh * PhDiff	  	  + VAR_kappa * PhasAlt * (1 - PhasAlt) * (PhasAlt - (0.5 + VAR_epsilon * (*Area - VAR_MeanVol) * CONST_Unit_Area))
													  - VAR_beta * (PolXAlt * PhGradX + PolYAlt * PhGradY) / (1 + PHAS_SQRT);
	#endif
};





//SpeicherOffset gibt den Abstand des Endwertspeichers in Rastergrößen an. StartwertOffset gibt den negativen Abstand des Startwerts in Rastergrößen an.
__global__ void ZeitDiffMitEuler(int Startvalue, int SpeicherOffset, int StartwertOffset, double *Fila, double *NukA, double *NukI, double *PolX, double *PolY, double *Phas, 
				 double *Area, double *FilaGradX, double *FilaGradY, double *PolaDiv, double *PhasGradX, double *PhasGradY, 
				 double *FilaDiff, double *NukADiff, double *NukIDiff, double *PolXDiff, double *PolYDiff, double *PhasDiff, double *Curv,
				 SingleCell *Cells, int CellNum , double Schrittweite){
	int ID = threadIdx.x + blockIdx.x * blockDim.x;
	int IDZiel = (SpeicherOffset << 16) + ID + Startvalue;
	int IDAlt = ID - (StartwertOffset << 16) + Startvalue;
	double PhDiff = (PHASEFIELDFLAG ? PhasDiff[ID] : 0), PhGradX = (PHASEFIELDFLAG ? PhasGradX[ID] : 0), PhGradY = (PHASEFIELDFLAG ? PhasGradY[ID] : 0);

	ID = ID + Startvalue;
	double FilaAlt = Fila[ID], NukAAlt = NukA[ID], NukIAlt = NukI[ID], PhasAlt = (PHASEFIELDFLAG ? Phas[ID] : 1), PolXAlt = PolX[ID], PolYAlt = PolY[ID];
	double NukSummand = PhasAlt * (VAR_wd * (FilaAlt/* + VAR_Acti * (PolXAlt * PolXAlt + PolYAlt * PolYAlt)*/) * NukAAlt - NukIAlt * (1 + VAR_w0 * NukAAlt * NukAAlt));
	ID = ID - Startvalue;

	Fila[IDZiel]  = Fila[IDAlt] + Schrittweite * (/*VAR_DiAkt    * (PhasAlt * FilaDiff[ID] - FilaAlt * PhDiff)*/  	    + PhasAlt * (VAR_alpha * NukAAlt - VAR_va * PolaDiv[ID] - VAR_kd * FilaAlt) - FilaAlt);
	#if NUCLEATORDEGRADFLAG > 0
		NukA[IDZiel]  = NukA[IDAlt] + Schrittweite * (VAR_DiNa * (PhasAlt * NukADiff[ID] - NukAAlt * PhDiff)   - NukSummand + (PhasAlt - 1) * NukAAlt);
		NukI[IDZiel]  = NukI[IDAlt] + Schrittweite * (	         (PhasAlt * NukIDiff[ID] - NukIAlt * PhDiff)   + NukSummand + (PhasAlt - 1) * NukIAlt + 15 * PhasAlt * (TotalNuk - *(Area+1) - *(Area+2)));
	#else
		NukA[IDZiel]  = NukA[IDAlt] + Schrittweite * (VAR_DiNa * (PhasAlt * NukADiff[ID] - NukAAlt * PhDiff)   - NukSummand);
		NukI[IDZiel]  = NukI[IDAlt] + Schrittweite * (	         (PhasAlt * NukIDiff[ID] - NukIAlt * PhDiff)   + NukSummand);
	#endif
	PolX[IDZiel]  = PolX[IDAlt] + Schrittweite * (/*VAR_DiAkt    * (PhasAlt * PolXDiff[ID] - PolXAlt * PhDiff)*/  	    - PhasAlt * (VAR_kd * PolXAlt + VAR_va * FilaGradX[ID]) - PolXAlt);
	PolY[IDZiel]  = PolY[IDAlt] + Schrittweite * (/*VAR_DiAkt    * (PhasAlt * PolYDiff[ID] - PolYAlt * PhDiff)*/  	    - PhasAlt * (VAR_kd * PolYAlt + VAR_va * FilaGradY[ID]) - PolYAlt);
	#if PHASEFIELDFLAG > 0
		Phas[IDZiel]  = Phas[IDAlt] + Schrittweite * (VAR_DiPh * PhDiff	   	    + VAR_kappa * PhasAlt * (1 - PhasAlt) * (PhasAlt - (0.5 + VAR_epsilon * (*Area - VAR_MeanVol) * CONST_Unit_Area))
																    - VAR_beta * (PolXAlt * PhGradX + PolYAlt * PhGradY) / (1 + PHAS_SQRT));
	#endif
};

__global__ void Euler(int Offset, double Schrittweite, double *GPU_Fila, double *GPU_NukA, double *GPU_NukI, double *GPU_PolX, double *GPU_PolY, double *GPU_Phas){
	int ID = threadIdx.x + blockIdx.x * blockDim.x;
	int IDZiel = (Offset << 16) + ID, IDSchritt = (1 << 16) + ID;
	// 1<<17 entspricht "2*RASTERGROESSE", also 2*256*256 = 2^17 und gibt den Offset zum Speichern an. In der Variable "Offset" ist der Speicherbereich des aktuellen f(c) für c_j+1 = c_j + dt * f(c_j) gespeichert
	GPU_Fila[IDZiel] = GPU_Fila[ID] + Schrittweite * (GPU_Fila[IDSchritt]); 	
	GPU_NukA[IDZiel] = GPU_NukA[ID] + Schrittweite * (GPU_NukA[IDSchritt]); 	
	GPU_NukI[IDZiel] = GPU_NukI[ID] + Schrittweite * (GPU_NukI[IDSchritt]); 	
	GPU_PolX[IDZiel] = GPU_PolX[ID] + Schrittweite * (GPU_PolX[IDSchritt]); 	
	GPU_PolY[IDZiel] = GPU_PolY[ID] + Schrittweite * (GPU_PolY[IDSchritt]); 
	#if PHASEFIELDFLAG > 0
		GPU_Phas[IDZiel] = GPU_Phas[ID] + Schrittweite * (GPU_Phas[IDSchritt]); 
	#endif
};

