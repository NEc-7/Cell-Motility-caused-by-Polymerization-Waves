#include "./Cells.cuh"
#include "./Constants.cuh"
#include "./FourierHelper.cuh" //NEEDS CONSTANTS
#include "./Dynamics.cuh" //NEEDS CONSTANTS, FOURIERHELPER

#define LOADERPREFIX "Lin"
#define LOADERSUFFIX "I-1-4"

int SingleCell::Initializer(double startx, double starty, int TrajNum){
	int i,j,k;
	double Summe = 0;
	char buffer[128];

	XStart = (startx/(double)GPZX - 0.5) * VAR_SysL; 
	YStart = (starty/(double)GPZY - 0.5) * VAR_SysL;
	cudaMallocHost((void **) &Fila, 	ByteZahl_Konzentrationen_double);
	cudaMallocHost((void **) &NukA,		ByteZahl_Konzentrationen_double);
	cudaMallocHost((void **) &NukI,		ByteZahl_Konzentrationen_double);
	cudaMallocHost((void **) &PolX,		ByteZahl_Konzentrationen_double);
	cudaMallocHost((void **) &PolY,		ByteZahl_Konzentrationen_double);
	cudaMallocHost((void **) &Phas,		ByteZahl_Konzentrationen_double);

	cudaMallocHost((void **) &SchwerpunktX,	sizeof(double) * ((int)(EndZeit/ZEITINTERVALL) + 1));
	cudaMallocHost((void **) &SchwerpunktY,	sizeof(double) * ((int)(EndZeit/ZEITINTERVALL) + 1));
	//printf("%d\n", ((int)(EndZeit/ZEITINTERVALL) + 1)); fflush(stdout);

	cudaMalloc((void**) &GPU_Fila, 	     	5*ByteZahl_Konzentrationen_double); 	/* 5* wegen Pseudospek, expl MPR und zwei erforderlichen Schritten für den SWS-Vergleichswert (mit delta t/2) */
	cudaMalloc((void**) &GPU_NukA, 	  	5*ByteZahl_Konzentrationen_double);
	cudaMalloc((void**) &GPU_NukI, 	     	5*ByteZahl_Konzentrationen_double);
	cudaMalloc((void**) &GPU_PolX, 		5*ByteZahl_Konzentrationen_double);
	cudaMalloc((void**) &GPU_PolY, 	    	5*ByteZahl_Konzentrationen_double);
	cudaMalloc((void**) &GPU_Phas,     	5*ByteZahl_Konzentrationen_double);
     	cudaMalloc((void**) &GPU_Area,          5 * sizeof(double)); 			//1. PFeld_TotA, 2. NukA_TotA, 3. NukI_TotA, 4. Phas_GradientAbsoluteNorm, 5. NumberOfPhasGradientPoints

	cudaMalloc((void**) &GPU_FilaDiff, 	ByteZahl_Konzentrationen_double);
	cudaMalloc((void**) &GPU_FilaGrdX, 	ByteZahl_Konzentrationen_double);	//Felder zum Abspeichern der Ableitungen aus der Berechnung mit FFT
	cudaMalloc((void**) &GPU_FilaGrdY, 	ByteZahl_Konzentrationen_double);
	cudaMalloc((void**) &GPU_NukADiff, 	ByteZahl_Konzentrationen_double);
	cudaMalloc((void**) &GPU_NukIDiff, 	ByteZahl_Konzentrationen_double);
	cudaMalloc((void**) &GPU_PolaDivg, 	ByteZahl_Konzentrationen_double);
	cudaMalloc((void**) &GPU_PolXDiff, 	ByteZahl_Konzentrationen_double);
	cudaMalloc((void**) &GPU_PolYDiff, 	ByteZahl_Konzentrationen_double);
	cudaMalloc((void**) &GPU_PhasDiff, 	ByteZahl_Konzentrationen_double);
	cudaMalloc((void**) &GPU_PhasGrdX, 	ByteZahl_Konzentrationen_double);
	cudaMalloc((void**) &GPU_PhasGrdY, 	ByteZahl_Konzentrationen_double);
	cudaMalloc((void**) &GPU_PhasGrdAbs, 	ByteZahl_Konzentrationen_double);
	cudaMalloc((void**) &GPU_PhasGrdAbsC,	ByteZahl_Konzentrationen_double);
	cudaMalloc((void**) &GPU_Curv, 		ByteZahl_Konzentrationen_double);


	//Create savers
	sprintf(buffer,"./Output/Daten/" ZZDD "/SMPR" FFDD "Trajektorie%c" ZZDD ".txt",(char)(TrajNum+65));
	Trajektorie = fopen(buffer, "w");
	sprintf(buffer,"./Output/Daten/" ZZDD "/SMPR" FFDD "CenterAct%c" ZZDD ".txt",(char)(TrajNum+65));
	CenterActin = fopen(buffer, "w");
	

	#if STATELOADERFLAG == 0
		double Normalizer, Radius;
		for(i=0; i<GPZX; i++){
			for(j=0; j<GPZY; j++){
				Radius = sqrt( (i-startx)*(i-startx) + (j-starty)*(j-starty) );
				//Erzeugt Kreis mit Mittelpunkt (150,150) und Fläche 75*75 (vgl. get_Area, Grundvolumen)
				if( Radius <= 0.75*Startradius ){
		     		   	Phas[i * GPZX + j] = 1;
					Fila[i * GPZX + j] = 0.5 * (1 + 0.2 * (drand48() - 0.5));
					NukA[i * GPZX + j] = VAR_CNuk *  (1 + 0.2 * (drand48() - 0.5));
					NukI[i * GPZX + j] = 1.0 *  (1 + 0.2 * (drand48() - 0.5));
					PolX[i * GPZX + j] = 0.5 * (drand48() - 0.5); 
					PolY[i * GPZX + j] = 0.5 * (drand48() - 0.5);
					Summe	     += NukA[i * GPZX + j] + NukI[i * GPZX + j];
				}else if( Radius <= (1.3*Startradius) ){
					Phas   [i * GPZX + j] = 0.5 * (1 - tanh(25.0 * VAR_SysL / GPZX * (Radius - Startradius)));//20.0 * (1.15 - Radius / Startradius) / 3.0;
					Fila[i * GPZX + j]    = 0.5 * (1 + 0.2 * (drand48() - 0.5))       * Phas[i * GPZX + j];
					NukA[i * GPZX + j]    = VAR_CNuk *  (1 + 0.2 * (drand48() - 0.5)) * Phas[i * GPZX + j];
					NukI[i * GPZX + j]    = 1.0 *  (1 + 0.2 * (drand48() - 0.5))      * Phas[i * GPZX + j];
					PolX[i * GPZX + j]    = 0.5 * (drand48() - 0.5)                   * Phas[i * GPZX + j];
					PolY[i * GPZX + j]    = 0.5 * (drand48() - 0.5)		    * Phas[i * GPZX + j];
					Summe	     += NukA[i * GPZX + j] + NukI[i * GPZX + j];
				}else{
					Phas[i * GPZX + j] = 0;
					Fila[i * GPZX + j] = 0;
					NukA[i * GPZX + j] = 0;
					NukI[i * GPZX + j] = 0;
					PolX[i * GPZX + j] = 0;
					PolY[i * GPZX + j] = 0;
				}
			}
		}
		//Normiere die Gesamtmenge der erhaltenen Nukleatoren
		Normalizer = TotalNuk / Summe;
		//printf("%.1f\n",Summe[0]);

		for(i=0;i<RASTERGROESSE;i++){
			NukA[i]    *= Normalizer;
			NukI[i]    *= Normalizer;
		}
	#else
		FILE *FilaFile, *NukAFile, *NukIFile, *PhasFile, *PolXFile, *PolYFile;
		FilaFile = fopen("./InitState/" LOADERPREFIX "FilaA-" LOADERSUFFIX ".txt","r");
		NukAFile = fopen("./InitState/" LOADERPREFIX "NukAA-" LOADERSUFFIX ".txt","r");
		NukIFile = fopen("./InitState/" LOADERPREFIX "NukIA-" LOADERSUFFIX ".txt","r");
		PolXFile = fopen("./InitState/" LOADERPREFIX "PolXA-" LOADERSUFFIX ".txt","r");
		PolYFile = fopen("./InitState/" LOADERPREFIX "PolYA-" LOADERSUFFIX ".txt","r");
		PhasFile = fopen("./InitState/" LOADERPREFIX "PhasA-" LOADERSUFFIX ".txt","r");

		for(i=0; i<GPZX; i++){
			for(j=0; j<GPZY; j++){
				//load corresponding values
				if(	fscanf(FilaFile, "%lf", &Fila[i * GPZX + j]) * 
					fscanf(PhasFile, "%lf", &Phas[i * GPZX + j]) * 
					fscanf(NukAFile, "%lf", &NukA[i * GPZX + j]) *
					fscanf(NukIFile, "%lf", &NukI[i * GPZX + j]) *
					fscanf(PolXFile, "%lf", &PolX[i * GPZX + j]) *
					fscanf(PolYFile, "%lf", &PolY[i * GPZX + j]) == 0){
						printf("Reading Error!\n");
						fflush(stdout);
						return(-1);
				};
				Summe	     += NukA[i * GPZX + j] + NukI[i * GPZX + j];
			}
		}
		printf("%f\n",Summe);
	#endif

	cudaMemcpy(GPU_Fila,     Fila,   ByteZahl_Konzentrationen_double, cudaMemcpyHostToDevice);
	cudaMemcpy(GPU_NukA,     NukA,   ByteZahl_Konzentrationen_double, cudaMemcpyHostToDevice);
	cudaMemcpy(GPU_NukI,     NukI,   ByteZahl_Konzentrationen_double, cudaMemcpyHostToDevice);
	cudaMemcpy(GPU_PolX,     PolX,   ByteZahl_Konzentrationen_double, cudaMemcpyHostToDevice);
	cudaMemcpy(GPU_PolY,     PolY,   ByteZahl_Konzentrationen_double, cudaMemcpyHostToDevice);
	cudaMemcpy(GPU_Phas,     Phas,   ByteZahl_Konzentrationen_double, cudaMemcpyHostToDevice);

	//Get initial Center of Mass
        Summe = 0;
	SchwerpunktX[0]=0;
	SchwerpunktY[0]=0;
	
	for(k=0;k<GPZX;k++){
	        for(j=0;j<GPZY;j++){
		        Summe += Phas[j * GPZX + k];
	            	if((Phas[j * GPZX]>1e-5)&&(Phas[(j + 1) * GPZX - 1]>1e-5)){
				if(k<(GPZX >> 1)){
	                    		SchwerpunktX[0] += Phas[j * GPZX + k] * (double)(k + GPZX);
	                	}else{
	                    		SchwerpunktX[0] += Phas[j * GPZY + k] * (double)k;
	               	 	}     
	            	}else{
				SchwerpunktX[0] += Phas[j * GPZY + k] * (double)k;
	            	}
		            	if((Phas[k]>1e-5)&&(Phas[(GPZY - 1) * GPZX + k]>1e-5)){
	                	if(j<(GPZY >> 1)){
	                    		SchwerpunktY[0] += Phas[j * GPZY + k] * (double)(j + GPZY);
	                	}else{
				        SchwerpunktY[0] += Phas[j * GPZY + k] * (double)j;
	                	}
	            	}else{
				SchwerpunktY[0] += Phas[j * GPZY + k] * (double)j;
	            	}
	        }	
        }
		//printf("%f\n",Summe);fflush(stdout);
        SchwerpunktX[0] *= CONST_xSchrittweite / Summe;
        SchwerpunktY[0] *= CONST_xSchrittweite / Summe;
       	if(SchwerpunktX[0] > VAR_SysL){SchwerpunktX[0] -= VAR_SysL;}
       	if(SchwerpunktY[0] > VAR_SysL){SchwerpunktY[0] -= VAR_SysL;}

	#if STATELOADERFLAG == 0
		fprintf(Trajektorie,"%f\t%f\t%f\n", XStart, YStart, 0.0);
		fflush(Trajektorie);
	#else
		fprintf(Trajektorie,"%f\t%f\t%f\n", XStart, YStart, 0.0);
		fflush(Trajektorie);

	#endif

	printf("Cell created at point (%.2f,%.2f).\n",startx,starty);
	return(0);
}


SingleCell::SingleCell(void){	//does nothing so that SingleCell can safely be defined on the GPU
}


SingleCell::~SingleCell(void){
}

void SingleCell::FreeMemory(void){

	fclose(Trajektorie);
	fclose(CenterActin);

	cudaFreeHost(Fila);
	cudaFreeHost(NukA);
	cudaFreeHost(NukI);
	cudaFreeHost(PolX);
	cudaFreeHost(PolY);
	cudaFreeHost(Phas);

	cudaFreeHost(SchwerpunktX);
	cudaFreeHost(SchwerpunktY);

	cudaFree(GPU_Fila);
	cudaFree(GPU_Phas);
	cudaFree(GPU_NukA);
	cudaFree(GPU_NukI);
	cudaFree(GPU_PolX);
	cudaFree(GPU_PolY);
	cudaFree(GPU_Area);

	cudaFree(GPU_FilaDiff);
	cudaFree(GPU_FilaGrdX);
	cudaFree(GPU_FilaGrdY);
	cudaFree(GPU_NukADiff);
	cudaFree(GPU_NukIDiff);	
	cudaFree(GPU_PolaDivg);
	cudaFree(GPU_PolXDiff);
	cudaFree(GPU_PolYDiff);
	cudaFree(GPU_PhasDiff);
	cudaFree(GPU_PhasGrdX);
	cudaFree(GPU_PhasGrdY);
	cudaFree(GPU_PhasGrdAbs);
	cudaFree(GPU_PhasGrdAbsC);
	cudaFree(GPU_Curv);
	printf("Destroyed a cell!\n");
}

void SingleCell::TotalValues(cublasHandle_t ToDevice, int Offset){
    	cublasDasum(ToDevice, RASTERGROESSE, GPU_Phas + Offset * RASTERGROESSE, 1, GPU_Area);
	#if NUCLEATORDEGRADFLAG > 0
	  	cublasDasum(ToDevice, RASTERGROESSE, GPU_NukA + Offset * RASTERGROESSE, 1, GPU_Area + 1);
	   	cublasDasum(ToDevice, RASTERGROESSE, GPU_NukI + Offset * RASTERGROESSE, 1, GPU_Area + 2);
	#endif
}

void SingleCell::SpectralDerivatives(FourierHelpers *FH, int Offset){
	Hilfsgroessen_Erzeugen(GPU_Fila + Offset * RASTERGROESSE, GPU_NukA + Offset * RASTERGROESSE, GPU_NukI + Offset * RASTERGROESSE, GPU_PolX + Offset * RASTERGROESSE, GPU_PolY + Offset * RASTERGROESSE, GPU_Phas + Offset * RASTERGROESSE, 
			       FH->GPU_Koeff_Fila, FH->GPU_Koeff_Fila2, FH->GPU_Koeff_Fila3, FH->GPU_Koeff_NukAkt, FH->GPU_Koeff_NukInakt, FH->GPU_Koeff_PolaX, FH->GPU_Koeff_PolaX2, FH->GPU_Koeff_PolaY, FH->GPU_Koeff_Phase, FH->GPU_Koeff_Phase2, FH->GPU_Koeff_Phase3,
			       GPU_FilaGrdX, GPU_FilaGrdY, GPU_PolaDivg, GPU_PhasGrdX, GPU_PhasGrdY, GPU_FilaDiff, GPU_NukADiff, GPU_NukIDiff, GPU_PolXDiff, GPU_PolYDiff, GPU_PhasDiff,
			       FH->FFTplanReellzuKomplex, FH->FFTplanKomplexzuReell);
}


void SingleCell::FilaToHost(void){
	cudaMemcpy(Fila, GPU_Fila,       ByteZahl_Konzentrationen_double, cudaMemcpyDeviceToHost);
}

void SingleCell::PhasToHost(void){
	cudaMemcpy(Phas, GPU_Phas,       ByteZahl_Konzentrationen_double, cudaMemcpyDeviceToHost);
}

void SingleCell::NuksToHost(void){
	cudaMemcpy(NukA, GPU_NukA,       ByteZahl_Konzentrationen_double, cudaMemcpyDeviceToHost);
	cudaMemcpy(NukI, GPU_NukI,       ByteZahl_Konzentrationen_double, cudaMemcpyDeviceToHost);
}

void SingleCell::AllToHost(void){
	cudaMemcpy(Fila, GPU_Fila,       ByteZahl_Konzentrationen_double, cudaMemcpyDeviceToHost);
	cudaMemcpy(PolX, GPU_PolX,       ByteZahl_Konzentrationen_double, cudaMemcpyDeviceToHost);
	cudaMemcpy(PolY, GPU_PolY,       ByteZahl_Konzentrationen_double, cudaMemcpyDeviceToHost);
	cudaMemcpy(NukA, GPU_NukA,       ByteZahl_Konzentrationen_double, cudaMemcpyDeviceToHost);
	cudaMemcpy(NukI, GPU_NukI,       ByteZahl_Konzentrationen_double, cudaMemcpyDeviceToHost);
}

void SingleCell::Update(void){
	CopyKernel<<<GPZX,GPZY,0>>>(GPU_Fila + (RASTERGROESSE << 2), GPU_Fila);
	CopyKernel<<<GPZX,GPZY,0>>>(GPU_NukA + (RASTERGROESSE << 2), GPU_NukA);
	CopyKernel<<<GPZX,GPZY,0>>>(GPU_NukI + (RASTERGROESSE << 2), GPU_NukI);
	CopyKernel<<<GPZX,GPZY,0>>>(GPU_PolX + (RASTERGROESSE << 2), GPU_PolX);
	CopyKernel<<<GPZX,GPZY,0>>>(GPU_PolY + (RASTERGROESSE << 2), GPU_PolY);
	CopyKernel<<<GPZX,GPZY,0>>>(GPU_Phas + (RASTERGROESSE << 2), GPU_Phas);
}

void SingleCell::StepFilaUpdate(void){
	CopyKernel<<<GPZX,GPZY,0>>>(GPU_Fila + (RASTERGROESSE * 3), GPU_Fila + (RASTERGROESSE << 1));
}

//simple but faster than memcpy
__global__ void CopyKernel(double *In, double *Out){
	int ID = threadIdx.x + blockIdx.x * blockDim.x;
	Out[ID] = In[ID];
};

double SingleCell::NumericStepError(cublasHandle_t ToHost, double *GPU_Array){
	double Fehler[1];
	int Index[1];

	Vektorsubtraktion<<<GPZX,GPZY>>>(GPU_Fila + (RASTERGROESSE << 1), GPU_Fila + (RASTERGROESSE << 2), GPU_Array);
        cublasIdamax(ToHost, RASTERGROESSE, GPU_Array, 1, Index);
	cudaMemcpy(Fehler, (GPU_Array + Index[0]), sizeof(double), cudaMemcpyDeviceToHost);
	/*
	Vektorsubtraktion<<<GPZX,GPZY>>>(GPU_Phas + (RASTERGROESSE << 1), GPU_Phas + (RASTERGROESSE << 2), GPU_Array);
        cublasIdamax(ToHost, RASTERGROESSE, GPU_Array, 1, Index);
	cudaMemcpy(Fehler+1, (GPU_Array + Index[0]), sizeof(double), cudaMemcpyDeviceToHost);
	*/
	return(fabs(*Fehler)/*+0.001*fabs(*(Fehler+1))*/);
}

double SingleCell::FindMaxLastDeviation(cublasHandle_t ToHost){
	double Fehler[1];
	int Index[1];

	cublasIdamax(ToHost, RASTERGROESSE, GPU_Fila + RASTERGROESSE, 1, Index);
       	cudaMemcpy(Fehler, (GPU_Fila + RASTERGROESSE + Index[0]), sizeof(double), cudaMemcpyDeviceToHost);
	return(fabs(*Fehler));
}


//simple but faster than the more general cublas counterpart
__global__ void Vektorsubtraktion(double *Minuend, double *Subtrahend, double *Ziel){
	int ID = threadIdx.x + GPZX * blockIdx.x;
     Ziel[ID] = abs(Minuend[ID] - Subtrahend[ID]);
};


void SingleCell::SaveCenterOfMass(int TimePoint, double Time){
				   double Summe = 0, CenterCon[1];
				   SchwerpunktX[TimePoint+1]=0;
				   SchwerpunktY[TimePoint+1]=0;
				   //flags tell if the cell crosses the periodic boundary on the x- and/or y-axis. 0 = no
				   int FlagX = 0, FlagY = 0, j, k;

				   //determine flags
				   for(j=0;j<GPZX;j++){
				       if((Phas[j * GPZX] > 1e-5) && (Phas[(j + 1) * GPZX - 1]    > 1e-5)){FlagX = 1;}
				       if((Phas[j]        > 1e-5) && (Phas[(GPZY - 1) * GPZX + j] > 1e-5)){FlagY = 1;}
				   }

				   //calculate center of mass depending on the flags
				       for(k=0;k<GPZX;k++){
					       	for(j=0;j<GPZY;j++){
							Summe += Phas[j * GPZX + k];
				           		if(FlagX){
								if(k<(GPZX >> 1)){
				                   			SchwerpunktX[TimePoint+1] += Phas[j * GPZX + k] * (double)(k + GPZX);
				               			}else{
				                   			SchwerpunktX[TimePoint+1] += Phas[j * GPZY + k] * (double)k;
				              			}     
				           		}else{
							   	SchwerpunktX[TimePoint+1] += Phas[j * GPZY + k] * (double)k;
				           		}
				           		if(FlagY){
				               			if(j<(GPZY >> 1)){
				                   			SchwerpunktY[TimePoint+1] += Phas[j * GPZY + k] * (double)(j + GPZY);
				               			}else{
							       		SchwerpunktY[TimePoint+1] += Phas[j * GPZY + k] * (double)j;
				               			}
				           		}else{
							   	SchwerpunktY[TimePoint+1] += Phas[j * GPZY + k] * (double)j;
				           		}

					       }	
				       }



				   //normalize and update position accordingly if the cell migrated over the periodic boundary
				       SchwerpunktX[TimePoint+1] *= CONST_xSchrittweite / Summe;
				       SchwerpunktY[TimePoint+1] *= CONST_xSchrittweite / Summe;
				   if(SchwerpunktX[TimePoint+1] > VAR_SysL){SchwerpunktX[TimePoint+1] -= VAR_SysL;}
				   if(SchwerpunktY[TimePoint+1] > VAR_SysL){SchwerpunktY[TimePoint+1] -= VAR_SysL;}

				   if(SchwerpunktX[TimePoint+1]-SchwerpunktX[TimePoint] > (VAR_SysL/2.0)){
					   SchwerpunktX[0]+=VAR_SysL;
				   }else if(SchwerpunktX[TimePoint+1]-SchwerpunktX[TimePoint] < (-VAR_SysL/2.0)){
					   SchwerpunktX[0]-=VAR_SysL;
				   }
				   if(SchwerpunktY[TimePoint+1]-SchwerpunktY[TimePoint] > (VAR_SysL/2.0)){
					   SchwerpunktY[0]+=VAR_SysL;
				   }else if(SchwerpunktY[TimePoint+1]-SchwerpunktY[TimePoint] < (-VAR_SysL/2.0)){
					   SchwerpunktY[0]-=VAR_SysL;
				   }

		           	   fprintf(Trajektorie,"%f\t%f\t%f\n", XStart + SchwerpunktX[TimePoint+1] - SchwerpunktX[0], YStart + SchwerpunktY[TimePoint+1] - SchwerpunktY[0], Time);
		           	   fflush(Trajektorie);

				   cudaMemcpy(CenterCon, (GPU_Fila + (int)round(SchwerpunktY[TimePoint+1]/CONST_xSchrittweite) * GPZY + (int)round(SchwerpunktX[TimePoint+1]/CONST_xSchrittweite)), sizeof(double), cudaMemcpyDeviceToHost);

		           	   fprintf(CenterActin,"%f\t%f\n", CenterCon[0], Time);
		           	   fflush(CenterActin);
}

//currently not used, occasional check for cell "division". Not to be used in every time step.
int SingleCell::ZellteilungsCheck(int TimePoint){
		int XsPos = (int)round(SchwerpunktX[TimePoint+1]/VAR_SysL * GPZX) & (GPZX - 1), YsPos = (int)round(SchwerpunktY[TimePoint+1]/VAR_SysL * GPZY) & (GPZY - 1);
		int XsMin,XsMax,YsMin,YsMax, j,k;

		if(Phas[YsPos * GPZX + XsPos]<1e-4){
			XsMin = (XsPos - ((int)Startradius>>1)) & (GPZX - 1);   	// 15 roughly half the cell radius
			YsMin = (YsPos - ((int)Startradius>>1)) & (GPZY - 1);
			XsMax = (XsPos + ((int)Startradius>>1)) & (GPZX - 1);
			YsMax = (YsPos + ((int)Startradius>>1)) & (GPZY - 1);		
								//if all of those points are outside (Square of 32x32 with center = center of mass of the cell), the cell can be assumed to have split.

			
			if(XsMin < XsMax){
				for(k=XsMin;k<XsMax;k++){
					if(YsMin < YsMax){
						for(j=YsMin;j<YsMax;j++){
							if(Phas[j * GPZX + k] > 1e-3){
								return(0);
							}
						}
					}else{
						for(j=YsMin;j<GPZY;j++){
							if(Phas[j * GPZX + k] > 1e-3){
								return(0);
							}
						}
						for(j=0;j<YsMax;j++){
							if(Phas[j * GPZX + k] > 1e-3){
								return(0);
							}
						}
					}		
				}
			}else{
				for(k=XsMin;k<GPZX;k++){
					if(YsMin < YsMax){
						for(j=YsMin;j<YsMax;j++){
							if(Phas[j * GPZX + k] > 1e-3){
								return(0);
							}
						}
					}else{
						for(j=YsMin;j<GPZY;j++){
							if(Phas[j * GPZX + k] > 1e-3){
								return(0);
							}
						}
						for(j=0;j<YsMax;j++){
							if(Phas[j * GPZX + k] > 1e-3){
								return(0);
							}
						}
					}		
				}
				for(k=0;k<XsMax;k++){
					if(YsMin < YsMax){
						for(j=YsMin;j<YsMax;j++){
							if(Phas[j * GPZX + k] > 1e-3){
								return(0);
							}
						}
					}else{
						for(j=YsMin;j<GPZY;j++){
							if(Phas[j * GPZX + k] > 1e-3){
								return(0);
							}
						}
						for(j=0;j<YsMax;j++){
							if(Phas[j * GPZX + k] > 1e-3){
								return(0);
							}
						}
					}		
				}
			}
			return(1);
		}
		return(0);
}

void SingleCell::SaveFila(char path[128]){
	int i,j;
	FILE *source;
	source = fopen(path,"w");
	for(j=0;j<GPZX;j+=2){
		for(i=0;i<GPZY-2;i+=2){
			fprintf(source,"%f\t", Fila[j*GPZY+i]);
		}
		fprintf(source, "%f\n", Fila[j * GPZY + GPZY - 2]);
	}
	fclose(source);
}

void SingleCell::SavePhas(char path[128]){
	int i,j;

	FILE *source;
	source = fopen(path,"w");
	if(source == NULL){
		printf("Error when opening file %s!\n",path);fflush(stdout);
	}
	while(source == NULL){
		source = fopen(path, "w");
		printf("Trying again...\n"); fflush(stdout);
		usleep(1000000 * 5);
	}

	for(j=0;j<GPZX;j+=2){
		for(i=0;i<GPZY-2;i+=2){
		        fprintf(source,"%f\t",Phas[j*GPZY+i]);
		}
		fprintf(source,"%f\n", Phas[j * GPZY + GPZY - 2]);
	}
	fclose(source);
}

void SingleCell::EndSave(char path[128], int Chooser){
	int i,j;
	FILE *source;
	source = fopen(path,"w");
	for(j=0;j<GPZX;j++){
		for(i=0;i<GPZY-1;i++){
			switch(Chooser){
		        	case 0 : fprintf(source,"%f\t",Fila[j * GPZY + i]); break;
		        	case 1 : fprintf(source,"%f\t",NukA[j * GPZY + i]); break;
		        	case 2 : fprintf(source,"%f\t",NukI[j * GPZY + i]); break;
		        	case 3 : fprintf(source,"%f\t",PolX[j * GPZY + i]); break;
		        	case 4 : fprintf(source,"%f\t",PolY[j * GPZY + i]); break;
		        	case 5 : fprintf(source,"%f\t",Phas[j * GPZY + i]); break;
			}
		}
		switch(Chooser){
		        case 0 : fprintf(source,"%f\n", Fila[j * GPZY + GPZY - 1]); break;
		        case 1 : fprintf(source,"%f\n", NukA[j * GPZY + GPZY - 1]); break;
		        case 2 : fprintf(source,"%f\n", NukI[j * GPZY + GPZY - 1]); break;
		        case 3 : fprintf(source,"%f\n", PolX[j * GPZY + GPZY - 1]); break;
		        case 4 : fprintf(source,"%f\n", PolY[j * GPZY + GPZY - 1]); break;
		        case 5 : fprintf(source,"%f\n", Phas[j * GPZY + GPZY - 1]); break;
		}

	}
	fclose(source);
}

void SingleCell::EulerStep(int SourceOffset, int RelativeTarget, double StepSize){
	Euler<<<GPZX,GPZY,0>>>(RelativeTarget, StepSize, GPU_Fila + SourceOffset * RASTERGROESSE, GPU_NukA + SourceOffset * RASTERGROESSE, GPU_NukI + SourceOffset * RASTERGROESSE, 
			       GPU_PolX + SourceOffset * RASTERGROESSE, GPU_PolY + SourceOffset * RASTERGROESSE, GPU_Phas + SourceOffset * RASTERGROESSE);
}

void SingleCell::TimeDerivative(int SourceOffset, int RelativeTarget, SingleCell *GPU_Cells, int CellNum){
	ZeitDiff<<<GPZX,GPZY,0>>>(SourceOffset * RASTERGROESSE, RelativeTarget, GPU_Fila, GPU_NukA, GPU_NukI, 
			          GPU_PolX, GPU_PolY, GPU_Phas, GPU_Area,
				  GPU_FilaGrdX, GPU_FilaGrdY, GPU_PolaDivg, GPU_PhasGrdX, GPU_PhasGrdY, GPU_FilaDiff, GPU_NukADiff, GPU_NukIDiff, GPU_PolXDiff, GPU_PolYDiff, GPU_PhasDiff, GPU_Curv, GPU_Cells, CellNum);
}

void SingleCell::TimeDerivativeStep(int SourceOffset, int RelativeTarget, int RelativeOrigin, double StepSize, SingleCell *GPU_Cells, int CellNum){
	ZeitDiffMitEuler<<<GPZX,GPZY,0>>>(SourceOffset * RASTERGROESSE, RelativeTarget, RelativeOrigin, GPU_Fila, GPU_NukA, GPU_NukI, 
			          	  GPU_PolX, GPU_PolY, GPU_Phas, GPU_Area,
				  	  GPU_FilaGrdX, GPU_FilaGrdY, GPU_PolaDivg, GPU_PhasGrdX, GPU_PhasGrdY, GPU_FilaDiff, GPU_NukADiff, GPU_NukIDiff, GPU_PolXDiff, GPU_PolYDiff, GPU_PhasDiff, GPU_Curv, GPU_Cells, CellNum, StepSize);
}



