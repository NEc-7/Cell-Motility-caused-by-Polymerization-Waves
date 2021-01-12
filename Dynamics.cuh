#ifndef DYNAMICS__H
#define DYNAMICS__H 1

class SingleCell;

__global__ void ZeitDiff(int Startvalue, int Offset, double *Fila, double *NukA, double *NukI, double *PolX, double *PolY, double *Phas, 
			 double *Area, double *FilaGradX, double *FilaGradY, double *PolaDiv, double *PhasGradX, double *PhasGradY, 
		  	 double *FilaDiff, double *NukADiff, double *NukIDiff, double *PolXDiff, double *PolYDiff, double *PhasDiff, double *Curv,
			 SingleCell *Cells, int CellNum);

__global__ void ZeitDiffMitEuler(int Startvalue, int SpeicherOffset, int StartwertOffset, double *Fila, double *NukA, double *NukI, double *PolX, double *PolY, double *Phas, 
				 double *Area, double *FilaGradX, double *FilaGradY, double *PolaDiv, double *PhasGradX, double *PhasGradY, 
				 double *FilaDiff, double *NukADiff, double *NukIDiff, double *PolXDiff, double *PolYDiff, double *PhasDiff, double *Curv,
				 SingleCell *Cells, int CellNum , double Schrittweite);

__global__ void Euler(int Offset, double Schrittweite, double *GPU_FilaA, double *GPU_NukA, double *GPU_NukI, double *GPU_PolX, double *GPU_PolY, double *GPU_Phas);

#endif
