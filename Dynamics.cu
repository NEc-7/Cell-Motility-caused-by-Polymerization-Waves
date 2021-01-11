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

//If this flag is set, use the curvature term to modify the nucleation dynamics. If not, don't.
#if MEMBTENSFLAG > 0
	#define POLYM_EXP (exp(-VAR_MembTensMod * Curv[ID] * (*(Area+3) * CONST_Unit_Area - VAR_MeanCirc) / VAR_MeanCirc))
#else
	#define POLYM_EXP 1.0
#endif

//If this flag is set, dynamics scale with the obstacle fields. If not, they don't.
#if OBSDYNFLAG > 0
	#define OBS_DYN ObsA
#else
	#define OBS_DYN 1.0
#endif

//Offset gibt den Abstand zum Speichern von f(c) an. Offset-Werte: k1->1, k->3, k3->4, k4->5
__global__ void ZeitDiffIAH(int Startvalue, int Offset, double *Fila, double *NukA, double *NukI, double *PolX, double *PolY, double *Phas, 
			    double *Obs, double *Area, double *FilaGradX, double *FilaGradY, double *PolaDiv, double *PhasGradX, double *PhasGradY, 
		  	    double *FilaDiff, double *NukADiff, double *NukIDiff, double *PolXDiff, double *PolYDiff, double *PhasDiff, double *Curv,
			    double *ObsDiff, double *GPU_DegradHelper, double *GPU_DeltHelper){
	int ID = threadIdx.x + blockIdx.x * blockDim.x;
	int IDZiel = ID + (Offset << 16) + Startvalue;
	double ObsA = Obs[ID], ObsD = ObsDiff[ID];
	double PhDiff = (PHASEFIELDFLAG ? PhasDiff[ID] : 0), PhGradX = (PHASEFIELDFLAG ? PhasGradX[ID] : 0), PhGradY = (PHASEFIELDFLAG ? PhasGradY[ID] : 0);

	ID = ID + Startvalue;
	double FilaAlt = Fila[ID], NukAAlt = NukA[ID], NukIAlt = NukI[ID], PhasAlt = (PHASEFIELDFLAG ? Phas[ID] : 1), PolXAlt = PolX[ID], PolYAlt = PolY[ID];
	double NukSummand = PhasAlt * OBS_DYN * (VAR_wd * (FilaAlt/* + VAR_Acti * (PolXAlt * PolXAlt + PolYAlt * PolYAlt)*/) * NukAAlt - NukIAlt * (1 + VAR_w0 * NukAAlt * NukAAlt));
	ID = ID - Startvalue;

	Fila[IDZiel]  =  /*VAR_DiAkt    * (PhasAlt * FilaDiff[ID] - FilaAlt * PhDiff)*/	 	  + PhasAlt * OBS_DYN * (VAR_alpha * NukAAlt - VAR_va * POLYM_EXP * PolaDiv[ID] - VAR_kd * FilaAlt) - FilaAlt;
	#if NUCLEATORDEGRADFLAG > 0
		NukA[IDZiel]  = (VAR_DiNa *   (PhasAlt * NukADiff[ID] - NukAAlt * PhDiff)	  - NukSummand + (PhasAlt - 1) * NukAAlt);
		NukI[IDZiel]  = (	      (PhasAlt * NukIDiff[ID] - NukIAlt * PhDiff)	  + NukSummand + (PhasAlt - 1) * NukIAlt + 15 * PhasAlt * (TotalNuk - *(Area+1) - *(Area+2)));
	#else
		NukA[IDZiel]  = (VAR_DiNa *   (PhasAlt * NukADiff[ID] - NukAAlt * PhDiff)	  - NukSummand);
		NukI[IDZiel]  = (	      (PhasAlt * NukIDiff[ID] - NukIAlt * PhDiff)	  + NukSummand);
	#endif
	PolX[IDZiel]  =  /*VAR_DiAkt    * (PhasAlt * PolXDiff[ID] - PolXAlt * PhDiff)*/ 	  - PhasAlt * OBS_DYN * (VAR_kd * PolXAlt + VAR_va * POLYM_EXP * FilaGradX[ID]) - PolXAlt;
	PolY[IDZiel]  =  /*VAR_DiAkt    * (PhasAlt * PolYDiff[ID] - PolYAlt * PhDiff)*/	  	  - PhasAlt * OBS_DYN * (VAR_kd * PolYAlt + VAR_va * POLYM_EXP * FilaGradY[ID]) - PolYAlt;
	#if PHASEFIELDFLAG > 0
		Phas[IDZiel]  = (VAR_DiPh * (ObsA * PhDiff 	       - PhasAlt * ObsD)	  	  + VAR_kappa * ObsA * PhasAlt * (1 - PhasAlt) * (PhasAlt - (0.8 - 0.3 * ObsA * GPU_DeltHelper[ID] + VAR_epsilon * (*Area - VAR_MeanVol) * CONST_Unit_Area )) 
													  - VAR_beta * ObsA * ObsA * (PolXAlt * PhGradX + PolYAlt * PhGradY) / (1 + PHAS_SQRT) - PHASDEGRAD * PhasAlt * GPU_DegradHelper[ID]);
	#endif
};





//SpeicherOffset gibt den Abstand des Endwertspeichers in Rastergrößen an. StartwertOffset gibt den negativen Abstand des Startwerts in Rastergrößen an.
__global__ void ZeitDiffMitEulerIAH(int Startvalue, int SpeicherOffset, int StartwertOffset, double *Fila, double *NukA, double *NukI, double *PolX, double *PolY, double *Phas, 
				    double *Obs, double *Area, double *FilaGradX, double *FilaGradY, double *PolaDiv, double *PhasGradX, double *PhasGradY, 
				    double *FilaDiff, double *NukADiff, double *NukIDiff, double *PolXDiff, double *PolYDiff, double *PhasDiff, double *Curv,
				    double *ObsDiff, double *GPU_DegradHelper, double *GPU_DeltHelper, double Schrittweite){
	int ID = threadIdx.x + blockIdx.x * blockDim.x;
	int IDZiel = (SpeicherOffset << 16) + ID + Startvalue;
	int IDAlt = ID - (StartwertOffset << 16) + Startvalue;
	double ObsA = Obs[ID], ObsD = ObsDiff[ID];
	double PhDiff = (PHASEFIELDFLAG ? PhasDiff[ID] : 0), PhGradX = (PHASEFIELDFLAG ? PhasGradX[ID] : 0), PhGradY = (PHASEFIELDFLAG ? PhasGradY[ID] : 0);

	ID = ID + Startvalue;
	double FilaAlt = Fila[ID], NukAAlt = NukA[ID], NukIAlt = NukI[ID], PhasAlt = (PHASEFIELDFLAG ? Phas[ID] : 1), PolXAlt = PolX[ID], PolYAlt = PolY[ID];
	double NukSummand = PhasAlt * OBS_DYN * (VAR_wd * (FilaAlt/* + VAR_Acti * (PolXAlt * PolXAlt + PolYAlt * PolYAlt)*/) * NukAAlt - NukIAlt * (1 + VAR_w0 * NukAAlt * NukAAlt));
	ID = ID - Startvalue;
	
	Fila[IDZiel]  = Fila[IDAlt] + Schrittweite * (/*VAR_DiAkt    * (PhasAlt * FilaDiff[ID] - FilaAlt * PhDiff)*/  	    + PhasAlt * OBS_DYN * (VAR_alpha * NukAAlt - VAR_va * POLYM_EXP * PolaDiv[ID] - VAR_kd * FilaAlt) - FilaAlt);
	#if NUCLEATORDEGRADFLAG > 0
		NukA[IDZiel]  = NukA[IDAlt] + Schrittweite * (VAR_DiNa * (PhasAlt * NukADiff[ID] - NukAAlt * PhDiff)   - NukSummand + (PhasAlt - 1) * NukAAlt);
		NukI[IDZiel]  = NukI[IDAlt] + Schrittweite * (	         (PhasAlt * NukIDiff[ID] - NukIAlt * PhDiff)   + NukSummand + (PhasAlt - 1) * NukIAlt + 15*PhasAlt * (TotalNuk - *(Area+1) - *(Area+2)));
	#else
		NukA[IDZiel]  = NukA[IDAlt] + Schrittweite * (VAR_DiNa * (PhasAlt * NukADiff[ID] - NukAAlt * PhDiff)   - NukSummand);
		NukI[IDZiel]  = NukI[IDAlt] + Schrittweite * (	         (PhasAlt * NukIDiff[ID] - NukIAlt * PhDiff)   + NukSummand);
	#endif
	PolX[IDZiel]  = PolX[IDAlt] + Schrittweite * (/*VAR_DiAkt    * (PhasAlt * PolXDiff[ID] - PolXAlt * PhDiff)*/  	    - PhasAlt * OBS_DYN * (VAR_kd * PolXAlt + VAR_va * POLYM_EXP * FilaGradX[ID]) - PolXAlt);
	PolY[IDZiel]  = PolY[IDAlt] + Schrittweite * (/*VAR_DiAkt    * (PhasAlt * PolYDiff[ID] - PolYAlt * PhDiff)*/  	    - PhasAlt * OBS_DYN * (VAR_kd * PolYAlt + VAR_va * POLYM_EXP * FilaGradY[ID]) - PolYAlt);
	#if PHASEFIELDFLAG > 0
		Phas[IDZiel]  = Phas[IDAlt] + Schrittweite * (VAR_DiPh * (ObsA * PhDiff 	    - PhasAlt * ObsD)	   	    + VAR_kappa * ObsA * PhasAlt * (1 - PhasAlt) * (PhasAlt - (0.8 - 0.3 * ObsA * GPU_DeltHelper[ID] 
																    + VAR_epsilon * (*Area - VAR_MeanVol) * CONST_Unit_Area))
																    - VAR_beta * ObsA * ObsA * (PolXAlt * PhGradX + PolYAlt * PhGradY) / (1 + PHAS_SQRT) - PHASDEGRAD * PhasAlt * GPU_DegradHelper[ID]);

	#endif
};

__global__ void ZeitDiff(int Startvalue, int Offset, double *Fila, double *NukA, double *NukI, double *PolX, double *PolY, double *Phas, 
			 double *Obs, double *Area, double *FilaGradX, double *FilaGradY, double *PolaDiv, double *PhasGradX, double *PhasGradY, 
		  	 double *FilaDiff, double *NukADiff, double *NukIDiff, double *PolXDiff, double *PolYDiff, double *PhasDiff, double *Curv,
			 double *ObsDiff, SingleCell *Cells, int CellNum){
	int i, ID = threadIdx.x + blockIdx.x * blockDim.x;
	int IDZiel = ID + (Offset << 16) + Startvalue;
	double ObsA = Obs[ID], ObsD = ObsDiff[ID];
	double PhDiff = (PHASEFIELDFLAG ? PhasDiff[ID] : 0), PhGradX = (PHASEFIELDFLAG ? PhasGradX[ID] : 0), PhGradY = (PHASEFIELDFLAG ? PhasGradY[ID] : 0);
	double DeltaProd = 1, OverlapProd = 0, Helper;

	ID = ID + Startvalue;
	for(i=0;i<CELLCOUNT;i++){
		if(i != CellNum){
			Helper = Cells[i].GPU_Phas[ID];
			DeltaProd *= (1 - Helper);
			OverlapProd += Helper * Helper;
		}
	}

	double FilaAlt = Fila[ID], NukAAlt = NukA[ID], NukIAlt = NukI[ID], PhasAlt = (PHASEFIELDFLAG ? Phas[ID] : 1), PolXAlt = PolX[ID], PolYAlt = PolY[ID];
	double NukSummand = PhasAlt * OBS_DYN * (VAR_wd * (FilaAlt/* + VAR_Acti * (PolXAlt * PolXAlt + PolYAlt * PolYAlt)*/) * NukAAlt - NukIAlt * (1 + VAR_w0 * NukAAlt * NukAAlt));
	ID = ID - Startvalue;

	Fila[IDZiel]  =  /*VAR_DiAkt    * (PhasAlt * FilaDiff[ID] - FilaAlt * PhDiff)*/	  	  + PhasAlt * OBS_DYN * (VAR_alpha * NukAAlt - VAR_va * POLYM_EXP * PolaDiv[ID] - VAR_kd * FilaAlt/* + VAR_Cortex * PhasSqrt/(1/4 + PhasSqrt)*/) - FilaAlt;
	#if NUCLEATORDEGRADFLAG > 0
		NukA[IDZiel]  = (VAR_DiNa *   (PhasAlt * NukADiff[ID] - NukAAlt * PhDiff)	  - NukSummand + (PhasAlt - 1) * NukAAlt);
		NukI[IDZiel]  = (	      (PhasAlt * NukIDiff[ID] - NukIAlt * PhDiff)	  + NukSummand + (PhasAlt - 1) * NukIAlt + 15 * PhasAlt * (TotalNuk - *(Area+1) - *(Area+2)));
	#else
		NukA[IDZiel]  = (VAR_DiNa *   (PhasAlt * NukADiff[ID] - NukAAlt * PhDiff)	  - NukSummand);
		NukI[IDZiel]  = (	      (PhasAlt * NukIDiff[ID] - NukIAlt * PhDiff)	  + NukSummand);
	#endif
	PolX[IDZiel]  =  /*VAR_DiAkt    * (PhasAlt * PolXDiff[ID] - PolXAlt * PhDiff)*/ 	  - PhasAlt * OBS_DYN * (VAR_kd * PolXAlt + VAR_va * POLYM_EXP * FilaGradX[ID]) - PolXAlt;
	PolY[IDZiel]  =  /*VAR_DiAkt    * (PhasAlt * PolYDiff[ID] - PolYAlt * PhDiff)*/	  	  - PhasAlt * OBS_DYN * (VAR_kd * PolYAlt + VAR_va * POLYM_EXP * FilaGradY[ID]) - PolYAlt;
	#if PHASEFIELDFLAG > 0
		Phas[IDZiel]  = (VAR_DiPh * (ObsA * PhDiff 	       - PhasAlt * ObsD)	  	  + VAR_kappa * ObsA * PhasAlt * (1 - PhasAlt) * (PhasAlt - (0.8 - 0.3 * ObsA * DeltaProd + VAR_epsilon * (*Area - VAR_MeanVol) * CONST_Unit_Area))
													  - VAR_beta * ObsA * ObsA * (PolXAlt * PhGradX + PolYAlt * PhGradY) / (1 + PHAS_SQRT) - PHASDEGRAD * PhasAlt * OverlapProd);
	#endif
};





//SpeicherOffset gibt den Abstand des Endwertspeichers in Rastergrößen an. StartwertOffset gibt den negativen Abstand des Startwerts in Rastergrößen an.
__global__ void ZeitDiffMitEuler(int Startvalue, int SpeicherOffset, int StartwertOffset, double *Fila, double *NukA, double *NukI, double *PolX, double *PolY, double *Phas, 
				 double *Obs, double *Area, double *FilaGradX, double *FilaGradY, double *PolaDiv, double *PhasGradX, double *PhasGradY, 
				 double *FilaDiff, double *NukADiff, double *NukIDiff, double *PolXDiff, double *PolYDiff, double *PhasDiff, double *Curv,
				 double *ObsDiff, SingleCell *Cells, int CellNum , double Schrittweite){
	int i, ID = threadIdx.x + blockIdx.x * blockDim.x;
	int IDZiel = (SpeicherOffset << 16) + ID + Startvalue;
	int IDAlt = ID - (StartwertOffset << 16) + Startvalue;
	double ObsA = Obs[ID], ObsD = ObsDiff[ID];
	double PhDiff = (PHASEFIELDFLAG ? PhasDiff[ID] : 0), PhGradX = (PHASEFIELDFLAG ? PhasGradX[ID] : 0), PhGradY = (PHASEFIELDFLAG ? PhasGradY[ID] : 0);
	double DeltaProd = 1, OverlapProd = 0, Helper;

	ID = ID + Startvalue;
	for(i=0;i<CELLCOUNT;i++){
		if(i != CellNum){
			Helper = Cells[i].GPU_Phas[ID];
			DeltaProd *= (1 - Helper);
			OverlapProd += Helper * Helper;
		}
	}
	double FilaAlt = Fila[ID], NukAAlt = NukA[ID], NukIAlt = NukI[ID], PhasAlt = (PHASEFIELDFLAG ? Phas[ID] : 1), PolXAlt = PolX[ID], PolYAlt = PolY[ID];
	double NukSummand = PhasAlt * OBS_DYN * (VAR_wd * (FilaAlt/* + VAR_Acti * (PolXAlt * PolXAlt + PolYAlt * PolYAlt)*/) * NukAAlt - NukIAlt * (1 + VAR_w0 * NukAAlt * NukAAlt));
	ID = ID - Startvalue;

	Fila[IDZiel]  = Fila[IDAlt] + Schrittweite * (/*VAR_DiAkt    * (PhasAlt * FilaDiff[ID] - FilaAlt * PhDiff)*/  	    + PhasAlt * OBS_DYN * (VAR_alpha * NukAAlt - VAR_va * POLYM_EXP * PolaDiv[ID] - VAR_kd * FilaAlt/* + VAR_Cortex * PhasSqrt/(1/4 + PhasSqrt)*/) - FilaAlt);
	#if NUCLEATORDEGRADFLAG > 0
		NukA[IDZiel]  = NukA[IDAlt] + Schrittweite * (VAR_DiNa * (PhasAlt * NukADiff[ID] - NukAAlt * PhDiff)   - NukSummand + (PhasAlt - 1) * NukAAlt);
		NukI[IDZiel]  = NukI[IDAlt] + Schrittweite * (	         (PhasAlt * NukIDiff[ID] - NukIAlt * PhDiff)   + NukSummand + (PhasAlt - 1) * NukIAlt + 15 * PhasAlt * (TotalNuk - *(Area+1) - *(Area+2)));
	#else
		NukA[IDZiel]  = NukA[IDAlt] + Schrittweite * (VAR_DiNa * (PhasAlt * NukADiff[ID] - NukAAlt * PhDiff)   - NukSummand);
		NukI[IDZiel]  = NukI[IDAlt] + Schrittweite * (	         (PhasAlt * NukIDiff[ID] - NukIAlt * PhDiff)   + NukSummand);
	#endif
	PolX[IDZiel]  = PolX[IDAlt] + Schrittweite * (/*VAR_DiAkt    * (PhasAlt * PolXDiff[ID] - PolXAlt * PhDiff)*/  	    - PhasAlt * OBS_DYN * (VAR_kd * PolXAlt + VAR_va * POLYM_EXP * FilaGradX[ID]) - PolXAlt);
	PolY[IDZiel]  = PolY[IDAlt] + Schrittweite * (/*VAR_DiAkt    * (PhasAlt * PolYDiff[ID] - PolYAlt * PhDiff)*/  	    - PhasAlt * OBS_DYN * (VAR_kd * PolYAlt + VAR_va * POLYM_EXP * FilaGradY[ID]) - PolYAlt);
	#if PHASEFIELDFLAG > 0
		Phas[IDZiel]  = Phas[IDAlt] + Schrittweite * (VAR_DiPh * (ObsA * PhDiff 	    - PhasAlt * ObsD)	   	    + VAR_kappa * ObsA * PhasAlt * (1 - PhasAlt) * (PhasAlt - (0.8 - 0.3 * ObsA * DeltaProd 
																    + VAR_epsilon * (*Area - VAR_MeanVol) * CONST_Unit_Area))
																    - VAR_beta * ObsA * ObsA * (PolXAlt * PhGradX + PolYAlt * PhGradY) / (1 + PHAS_SQRT) - PHASDEGRAD * PhasAlt * OverlapProd);
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

