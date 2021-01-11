#include <cuda.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <time.h>
#include <stdio.h>
#include <sys/types.h>
#include <math.h>
#include <unistd.h>

#include "./Constants.cuh"
#include "./Cells.cuh"
#include "./FourierHelper.cuh"
#include "./Obstacle.cuh"

#if CELLCOUNT > 5		//choose the optimal function for calculating cell-cell interaction based on the cell count
	#include "./InteractionHelper.cuh"
	#define STEPUPDATER_CELLCOUNTSTART 	CellArray[i].TimeDerivativeStepIAH(
	#define UPDATER_CELLCOUNTSTART 		CellArray[i].TimeDerivativeIAH(
	#define UPDATER_CELLCOUNTEND 		IAH->GPU_DegradHelper + i * RASTERGROESSE, IAH->GPU_DeltHelper + i * RASTERGROESSE); 
#else
	#define STEPUPDATER_CELLCOUNTSTART 	CellArray[i].TimeDerivativeStep(
	#define UPDATER_CELLCOUNTSTART 		CellArray[i].TimeDerivative(
	#define UPDATER_CELLCOUNTEND 		GPU_CellArray, i);
#endif



int main(){
	#if PHASEFIELDFLAG > 0
		printf("Simulating %d cells in a system with a length of %f with a zoom of %f.\nObstacle type: ", CELLCOUNT, VAR_SysLen, ZOOMFACTOR);
		switch(OBSTACLETYPE){
			case 0:  printf("Nothing\n");break;
			case 1:  printf("Circle\n");break;
			case 2:  printf("Square\n");break;
			case 3:  printf("Star\n");break;
			case 4:  printf("Hexagonal Maze\n");break;
			case 5:  printf("X Channel\n");break;
			case 6:  printf("Wells\n");break;
			default: printf("Undefined\n");
		}
	#else
		printf("Simulating actin waves without a cell in a system with a length of %f with a zoom of %f.\n", CELLCOUNT, VAR_SysLen, ZOOMFACTOR);
	#endif

	#if NUCLEATORDEGRADFLAG > 0
		printf("Forcing constant nucleator levels with externel degradation.\n");
	#else
		printf("Not forcing constant nucleator levels.\n");
	#endif
	#if MEMBTENSFLAG > 0
		printf("Considering membrane curvature terms.\n");
	#else
		printf("Not considering membrane curvature terms.\n");
	#endif

	#if STATELOADERFLAG > 0
		printf("Loading initial states from file.\n");
	#else
		printf("Using randomized initial conditions.\n");
	#endif
	cudaDeviceReset();
	cudaSetDevice(GPUID);

	fflush(stdout);

	cublasHandle_t handle, handleResultToDevice, handleResultToHost;

	cublasCreate(&handle);
	cublasCreate(&handleResultToDevice);
	cublasCreate(&handleResultToHost);
	cublasSetPointerMode(handleResultToDevice, CUBLAS_POINTER_MODE_DEVICE);

	SingleCell CellArray[CELLCOUNT], *GPU_CellArray; //GPU array is needed for copying the GPU array pointers to the GPU. Not possible otherwise because CellArray is a host pointer to an array of host objects with host pointers to the device.
	FourierHelpers FH[1];
	Obstacles Obs[1]{FH};	
	#if CELLCOUNT > 5
		printf("Using Interaction helpers to minimize memory accesses.\n");
		InterActionHelper IAH[1];
	#else
		printf("Not using Interaction helpers.\n");
	#endif

	fflush(stdout);

	int i, j;
	long int Schrittzahl, seed = ((long int)time(NULL) + getpid() + (long int) clock());
	//seed = 1460718856; //constant value for testing
	srand48(seed);

	double Zeit, ZeitAlt, Startzeit, Schrittweite, MaxWeiteSp=1e-2, Fehlersumme=0;
	int    ZeitSchwelle = 0, Bildzaehler = 1;

	double *GPU_Arr, Area[4];						
	char buffer[128];
	FILE *Schrittweiten, *Speicher;
	FILE *Params, *Error;

	Schrittweiten = fopen("./Output/Daten/" ZZDD "/SMPR" FFDD "WeitenDelt" ZZDD ".txt","w");

	Params = fopen("./Output/Daten/" ZZDD "/SMPR" FFDD "ParameterWerteDelt" ZZDD ".txt","w");
	fprintf(Params,"Parameterwerte\nAlpha:\t%f\nBeta:\t%f\nepsilon:\t%f\nKappa:\t%f\nOmega:\t%f\nOmega_0:\t%f\nk:\t%f\nv_a:\t%f\n", VAR_alpha, VAR_beta, VAR_epsilon, VAR_kappa, VAR_wd, VAR_w0, VAR_kd, VAR_va);
	fprintf(Params,"Motoraktivitaet:\t%g\nInflux_Akt:\t%f\nD_Na:\t%f\nD_Phase:\t%f\nCNuk:\t%f\nSysL:\t%f\nStartradius:\t%f\nGitterpunkte:\t%dx%d\nFehlerschwelle SWS:\t%e\nSeed:\t%ld\nCellcount:\t%d\nZoom:\t%.2f\nCortexLevel:\t%f\nMembTensMod:\t%f\n", VAR_Acti, InfluxStrength, VAR_DiNa, VAR_DiPh, VAR_CNuk, VAR_SysL, Startradius, GPZX, GPZY, Fehlerschwelle, seed, CELLCOUNT, ZOOMFACTOR, VAR_Cortex, VAR_MembTensMod);
	fprintf(Params,"Phasefield:\t%d\nObstacletype:\t%d\nNucDegrad:\t%d\nCurvatureSim:\t%d\nAdaptiveSteps:\t%d\nNormalizePhasGrad:\t%d\nTimeSaveStep:\t%f\nConcSaveInt:\t%d\nInitState:\t%d\nObsProtScaling:\t%d\nObsDivisor:\t%d\nOverlapDegrad:\t%f\n", PHASEFIELDFLAG, OBSTACLETYPE, NUCLEATORDEGRADFLAG, MEMBTENSFLAG, STEPSIZEFLAG, MEMBGRADIENTNORMALIZER, ZEITINTERVALL, BILDINTERVALL,STATELOADERFLAG,OBSDYNFLAG,OBSTACLEDIVISOR,PHASDEGRAD);
	fprintf(Params,"TimeStep:\t%f\nEndTime:\t%f\nTimeOffset:\t%f\nDensitySaveInterval:\t%d\n",ZEITINTERVALL, EndZeit, ZEITOFFSET,BILDINTERVALL);
	fclose(Params);

	//Save obstacle field
	Speicher = fopen("./Output/Daten/" ZZDD "/SMPR" FFDD "ObstaclePhasefieldDelt" ZZDD ".txt","w");
	for(j=0;j<GPZX;j++){
		for(i=0;i<GPZY-1;i++){
			fprintf(Speicher,"%f\t", Obs->Obstacle[j*GPZY+i]);
		}
		fprintf(Speicher, "%f\n", Obs->Obstacle[j * GPZY + GPZY - 1]);
	}
	fclose(Speicher);


	fflush(stdout);
	int RetVal = 0;

	for(i=0;i<CELLCOUNT;i++){
		//Initialize cells
		//Needs to be done BEFORE copying pointers to the device because it allocates space.
		int ArrayLen, Increment, PosX, PosY;
		if(OBSTACLETYPE==0){
			//ArrayLen  = ceil(GPZX / (2 * Startradius + 30));
			//Increment = ceil(2 * Startradius + 20);
			ArrayLen  = ceil(sqrt(CELLCOUNT));
			Increment = floor(GPZX / ArrayLen) - 5;
			printf("%d\t%d\n",ArrayLen,Increment);
		}
		switch(OBSTACLETYPE){
			case 0 :
				PosX = (i % (ArrayLen));
				PosY = (i / (ArrayLen));
				#if PHASEFIELDFLAG > 0
					RetVal = min(RetVal, CellArray[i].Initializer(ceil(Startradius) + Increment * PosX, ceil(Startradius) + Increment * PosY, Obs->Obstacle, i));	
				#else
					RetVal = min(RetVal, CellArray[i].Initializer(round(GPZX/2),round(GPZY/2), Obs->Obstacle, i));
				#endif	
				break;
			case 1 : case 2 :
				RetVal = min(RetVal, CellArray[i].Initializer(128 + cos(2*PI*i/(double)CELLCOUNT)*(120-Startradius), 128 + sin(2*PI*i/(double)CELLCOUNT)*(120-Startradius), Obs->Obstacle, i));
				break;
			case 3 : case 5 :
				RetVal = min(RetVal, CellArray[i].Initializer(128 + cos(2*PI*i/(double)CELLCOUNT)*(120-Startradius-30), 128 + sin(2*PI*i/(double)CELLCOUNT)*(120-Startradius-30), Obs->Obstacle, i));	
				break;
			case 4 :		
				//Hexagonal maze is hardcoded up until 7 cells.
				switch(i){
					case 0 :
						RetVal = min(RetVal, CellArray[i].Initializer(GPZX/2 * (1-1/sqrt(3)), GPZY/4, Obs->Obstacle, i));	
						break;
					case 1 :
						RetVal = min(RetVal, CellArray[i].Initializer(GPZX/2 * (1+1/sqrt(3)), GPZY/4, Obs->Obstacle, i));	
						break;
					case 2 :
						RetVal = min(RetVal, CellArray[i].Initializer(GPZX/2 * (1-1/sqrt(3)), 3*GPZY/4, Obs->Obstacle, i));	
						break;
					case 3 :
						RetVal = min(RetVal, CellArray[i].Initializer(GPZX/2 * (1+1/sqrt(3)), 3*GPZY/4, Obs->Obstacle, i));	
						break;
					case 4 :
						RetVal = min(RetVal, CellArray[i].Initializer(GPZX/2, GPZY/2, Obs->Obstacle, i));	
						break;
					case 5 :
						RetVal = min(RetVal, CellArray[i].Initializer(GPZX/2 * (1-1/sqrt(4*3)), GPZY/2, Obs->Obstacle, i));	
						break;
					case 6 :
						RetVal = min(RetVal, CellArray[i].Initializer(GPZX/2 * (1+1/sqrt(4*3)), GPZY/2, Obs->Obstacle, i));	
						break;
				}
				break;
			case 6 :
			{
				int RowLen = ceil(sqrt(CELLCOUNT/2.0));
                        	int Increment = floor(GPZX / RowLen) - 10;
                        	double Rad = ((double)GPZX/RowLen - 17.5)/2.0;
                        	int PX,PY;
				PX = ceil(Rad) + 10 + ((i/2)%RowLen) * Increment;
				PY = ceil(Rad) + 10 + ((i/2)/RowLen) * Increment;
                        	
				RetVal = min(RetVal, CellArray[i].Initializer(PX, PY + (2 * (i%2) - 1) * floor(Rad/3), Obs->Obstacle, i));
			}
		}
	}

	//Error check for Initializer
	if(RetVal < 0)	return(0);

	//Initialize constants
	Schrittweite = 1e-6;
	Startzeit    = clock();
	Zeit         = 0;
	ZeitAlt      = 0;
	Schrittzahl  = 0;
	cudaMalloc((void**) &GPU_Arr, 		ByteZahl_Konzentrationen_double);
	cudaMalloc((void**) &GPU_CellArray, 	CELLCOUNT * sizeof(SingleCell));
	cudaMemcpy(GPU_CellArray,  CellArray,   CELLCOUNT * sizeof(SingleCell), cudaMemcpyHostToDevice); 	//Copy pointers to the cuda stuff to the device. Needs to be done AFTER INITIALIZING CellArray!
	fflush(stdout);

	while(Zeit < EndZeit){

#if STEPSIZEFLAG > 0
		// STEP SIZE CONTROL
		//Standard midpoint rule
		for(i=0;i<CELLCOUNT;i++){
			CellArray[i].SpectralDerivatives(FH, 0);
			CellArray[i].TotalValues(handleResultToDevice, 0);
			#if CELLCOUNT > 5 	
				IAH->Update(0, GPU_CellArray); 
			#endif
		}
		cudaDeviceSynchronize();
		for(i=0;i<CELLCOUNT;i++){
			UPDATER_CELLCOUNTSTART 0, 1, Obs, UPDATER_CELLCOUNTEND	//This weird construction optimizes the amount of memory accesses when calculating cell-cell interaction by choosing different functions depending on the cell count (see beginning of main.cu)
			CellArray[i].EulerStep(0, 2, Schrittweite/2);
		}
		cudaDeviceSynchronize();
		for(i=0;i<CELLCOUNT;i++){
			CellArray[i].SpectralDerivatives(FH, 2);
			CellArray[i].TotalValues(handleResultToDevice, 2);
			#if CELLCOUNT > 5 	
				IAH->Update(2, GPU_CellArray);
			#endif
		}
		cudaDeviceSynchronize();
		for(i=0;i<CELLCOUNT;i++){
			STEPUPDATER_CELLCOUNTSTART 2, 0, 2, Schrittweite, Obs, UPDATER_CELLCOUNTEND
		}

		//2 Midpoint rule steps with half step size. Halve it until the error threshold is met.
		do{
			for(i=0;i<CELLCOUNT;i++){
				CellArray[i].EulerStep(0, 3, Schrittweite/4);
				CellArray[i].SpectralDerivatives(FH, 3);
				CellArray[i].TotalValues(handleResultToDevice, 3);
				#if CELLCOUNT > 5 	
					IAH->Update(3, GPU_CellArray);
				#endif
			}
			cudaDeviceSynchronize();
			for(i=0;i<CELLCOUNT;i++){
				STEPUPDATER_CELLCOUNTSTART 3, 0, 3, Schrittweite/2, Obs, UPDATER_CELLCOUNTEND
				CellArray[i].SpectralDerivatives(FH, 3);
				CellArray[i].TotalValues(handleResultToDevice, 3);
				#if CELLCOUNT > 5 	
					IAH->Update(3, GPU_CellArray);
				#endif
			}
			cudaDeviceSynchronize();
			for(i=0;i<CELLCOUNT;i++){
				STEPUPDATER_CELLCOUNTSTART 3, 1, 0, Schrittweite/4, Obs, UPDATER_CELLCOUNTEND
				CellArray[i].SpectralDerivatives(FH, 4);
				CellArray[i].TotalValues(handleResultToDevice, 4);
				#if CELLCOUNT > 5 	
					IAH->Update(4, GPU_CellArray);
				#endif
			}
			cudaDeviceSynchronize();
			for(i=0;i<CELLCOUNT;i++){
				STEPUPDATER_CELLCOUNTSTART 4, 0, 1, Schrittweite/2, Obs, UPDATER_CELLCOUNTEND
			}

			cudaDeviceSynchronize();
			Fehlersumme = 0;

			for(i=0;i<CELLCOUNT;i++){
				Fehlersumme += CellArray[i].NumericStepError(handleResultToHost, GPU_Arr);
			}

			if((Fehlersumme > Fehlerschwelle)||(Schrittweite>MaxWeiteSp)){
				Schrittweite/=2.0;
				for(i=0;i<CELLCOUNT;i++){
					CellArray[i].StepFilaUpdate();
				}

			}
		}while((Fehlersumme > Fehlerschwelle)||(Schrittweite>MaxWeiteSp));

		//always check for divergence for trouble shooting
		if(Fehlersumme < 0 || isnan(Fehlersumme)){
			Error = fopen("./Output/Daten/" ZZDD "/Error" ZZDD "Error.txt","w");
			fprintf(Error, "Divergenz bei t=%3f nach %d Bildern.",Zeit,Bildzaehler);
			fclose(Error);
			break;
		}

		//Update all quantities
		for(i=0;i<CELLCOUNT;i++){
			CellArray[i].Update();
		}

		Zeit+=Schrittweite;
		Schrittweite*=1.01;		
		Schrittzahl++;

		//frequently save step size and check if a stationary state is reached.
		if(Schrittzahl%1000==0){
			fprintf(Schrittweiten,"%ld\t%g\t%g\n",Schrittzahl, Schrittweite, Fehlersumme); fflush(Schrittweiten);

			//"Fehlersumme" ist hier das Maximum, spart ein paar Speicherplätze. Gesucht wird das Maximum von d/dt T im letzten Simulationsschritt. Passiert nichts, wird die Simulation direkt unterbrochen.
			Fehlersumme = 0;				
			for(i=0;i<CELLCOUNT;i++){
					Fehlersumme = max(Fehlersumme, CellArray[i].FindMaxLastDeviation(handleResultToHost));
			}
			if(Fehlersumme < (VAR_CNuk * 0.2e-2)){
				printf("Stationary state.");
				Error = fopen("./Output/Daten/" ZZDD "/Error" ZZDD "Error.txt","w");
				fprintf(Error, "Stationärer Zustand bei t=%3f nach %d Bildern.",Zeit,Bildzaehler);
				fclose(Error);
				break;
			}

		}
#else
		//NO STEP SIZE CONTROL! STANDARD EULER STEP
		for(i=0;i<CELLCOUNT;i++){
			CellArray[i].SpectralDerivatives(FH, 0);
			CellArray[i].TotalValues(handleResultToDevice, 0);
			#if CELLCOUNT > 5 	
				IAH->Update(0, GPU_CellArray); 
			#endif
		}
		for(i=0;i<CELLCOUNT;i++){
			STEPUPDATER_CELLCOUNTSTART 0, 4, 0, Schrittweite, Obs, UPDATER_CELLCOUNTEND
			CellArray[i].Update();
		}
		Fehlersumme = 0;


		Zeit+=Schrittweite;		
		Schrittzahl++;

		if(Schrittzahl%1000==0){
			fprintf(Schrittweiten,"%ld\t%g\t%g\n",Schrittzahl, Schrittweite, Fehlersumme); fflush(Schrittweiten);
		}

#endif




		//sometimes print progress and check if the step size isn't diminishing
		if(Schrittzahl%100000==0){
			printf(ZZDD", %ld steps; Time: %g\n", Schrittzahl, Zeit);
			if((Zeit-ZeitAlt < 1e-4) && (Schrittzahl > 1e6)){
				printf("No progress anymore.");
				Error = fopen("./Output/Daten/" ZZDD "/Error" ZZDD "Error.txt","w");
				fprintf(Error, "Zu kleine Schrittweite bei t=%3f nach %d Bildern (-> Divergenz).",Zeit,Bildzaehler);
				fclose(Error);
				break;
			}
			ZeitAlt = Zeit;
		}

		//if time passes the save interval, calculate and save important stuff
		if(Zeit > (ZEITOFFSET + ZEITINTERVALL * ZeitSchwelle)){
			#if PHASEFIELDFLAG > 0
				for(i=0;i<CELLCOUNT;i++){
					CellArray[i].PhasToHost();
					CellArray[i].SaveCenterOfMass(ZeitSchwelle, Zeit);
				}
			#endif
			
			//Konzentrationen speicherncd
			if(((ZeitSchwelle % BILDINTERVALL)==0)){
				for(i=0;i<CELLCOUNT;i++){
					#if PHASEFIELDFLAG > 0
					     	sprintf(buffer,"./Output/Daten/" ZZDD "/SMPR" FFDD "Phasenfeld%c" ZZDD "B%d.txt",(char)(i+65), Bildzaehler);
						CellArray[i].SavePhas(buffer);
					#endif
					CellArray[i].FilaToHost();
					sprintf(buffer,"./Output/Daten/" ZZDD "/SMPR" FFDD "Filamente%c" ZZDD "B%d.txt",(char)(i+65), Bildzaehler);
					CellArray[i].SaveFila(buffer);
					//CellArray[i].NuksToHost();
					#if PHASEFIELDFLAG > 0
						#if NUCLEATORDEGRADFLAG > 0
							#if MEMBTENSFLAG > 0
								cudaMemcpy(Area,           CellArray[i].GPU_Area,       4*sizeof(double),                cudaMemcpyDeviceToHost);
					 			printf("\tCell %d: Area %.1f, CoM: (%.2f,%.2f), N_Tot: %.2f, Circ: %.2f\n", i, Area[0], CellArray[i].SchwerpunktX[ZeitSchwelle], CellArray[i].SchwerpunktY[ZeitSchwelle],Area[1]+Area[2],(*(Area+3) * CONST_Unit_Area - VAR_MeanCirc) / VAR_MeanCirc);
							#else
								cudaMemcpy(Area,           CellArray[i].GPU_Area,       3*sizeof(double),                cudaMemcpyDeviceToHost);
					 			printf("\tCell %d: Area %.1f, CoM: (%.2f,%.2f), N_Tot: %.2f\n", i, Area[0], CellArray[i].SchwerpunktX[ZeitSchwelle], CellArray[i].SchwerpunktY[ZeitSchwelle],Area[1]+Area[2]);
							#endif
						#else
							#if MEMBTENSFLAG > 0
								cudaMemcpy(Area,           CellArray[i].GPU_Area,       4*sizeof(double),                cudaMemcpyDeviceToHost);
					 			printf("\tCell %d: Area %.1f, CoM: (%.2f,%.2f), Circ: %.2f\n", i, Area[0], CellArray[i].SchwerpunktX[ZeitSchwelle], CellArray[i].SchwerpunktY[ZeitSchwelle],(*(Area+3) * CONST_Unit_Area - VAR_MeanCirc) / VAR_MeanCirc);

							#else
								cudaMemcpy(Area,           CellArray[i].GPU_Area,       sizeof(double),                cudaMemcpyDeviceToHost);
					 			printf("\tCell %d: Area %.1f, CoM: (%.2f,%.2f)\n", i, Area[0], CellArray[i].SchwerpunktX[ZeitSchwelle], CellArray[i].SchwerpunktY[ZeitSchwelle]);

							#endif
						#endif				
					#endif
				}
				Bildzaehler++;

			}
			//printf("%d\n",ZeitSchwelle); fflush(stdout);
			ZeitSchwelle++;
		}
	}

	printf("Time for %ld iterations: %.2f s\n", Schrittzahl, (clock() - Startzeit)/(double) CLOCKS_PER_SEC);
	printf("End-time (sim): %g with an err thresh of %g\n",Zeit, Fehlerschwelle);
	printf("Last error: %g\n", Fehlersumme);

	//Save Endstates
	for(i=0;i<CELLCOUNT;i++){
		printf("%f\n",CellArray[i].Fila[150*256+80]);
		CellArray[i].AllToHost();
		sprintf(buffer,"./Output/Daten/" ZZDD "/SMPR" FFDD "EndeFila%c" ZZDD ".txt",(char)(i+65));
		CellArray[i].EndSave(buffer, 0);	//print the entire array. 0=Fila,1=NukA,2=NukI,3=PolX,4=PolY,5=Phas, everything else = nothing
		sprintf(buffer,"./Output/Daten/" ZZDD "/SMPR" FFDD "EndeNukA%c" ZZDD ".txt",(char)(i+65));
		CellArray[i].EndSave(buffer, 1);
		sprintf(buffer,"./Output/Daten/" ZZDD "/SMPR" FFDD "EndeNukI%c" ZZDD ".txt",(char)(i+65));
		CellArray[i].EndSave(buffer, 2);
		sprintf(buffer,"./Output/Daten/" ZZDD "/SMPR" FFDD "EndePolX%c" ZZDD ".txt",(char)(i+65));
		CellArray[i].EndSave(buffer, 3);
		sprintf(buffer,"./Output/Daten/" ZZDD "/SMPR" FFDD "EndePolY%c" ZZDD ".txt",(char)(i+65));
		CellArray[i].EndSave(buffer, 4);
		sprintf(buffer,"./Output/Daten/" ZZDD "/SMPR" FFDD "EndePhas%c" ZZDD ".txt",(char)(i+65));
		CellArray[i].EndSave(buffer, 5);
		printf("%f\n",CellArray[i].Fila[150*256+80]);
	}

	//Cleanup
	for(i=0;i<CELLCOUNT;i++){
		CellArray[i].FreeMemory();
	}
        usleep(1);

	cudaFree(GPU_Arr);
	cudaFree(GPU_CellArray);
	cublasDestroy(handle);
	cublasDestroy(handleResultToDevice);
	cublasDestroy(handleResultToHost);
	fclose(Schrittweiten);

	return 0;
}
