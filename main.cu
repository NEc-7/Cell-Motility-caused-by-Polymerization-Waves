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

//Leftovers from a bigger program
#define STEPUPDATER_CELLCOUNTSTART 	CellArray[0].TimeDerivativeStep(
#define UPDATER_CELLCOUNTSTART 		CellArray[0].TimeDerivative(
#define UPDATER_CELLCOUNTEND 		GPU_CellArray, 0);


int main(){
	#if PHASEFIELDFLAG > 0
		printf("Simulating %d cell in a system with a length of %f with a zoom of %f.\nObstacle type: ", 1, VAR_SysLen, ZOOMFACTOR);
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
		printf("Simulating actin waves without a cell in a system with a length of %f with a zoom of %f.\n", VAR_SysLen, ZOOMFACTOR);
	#endif

	#if NUCLEATORDEGRADFLAG > 0
		printf("Forcing constant nucleator levels with externel degradation.\n");
	#else
		printf("Not forcing constant nucleator levels.\n");
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

	SingleCell CellArray[1], *GPU_CellArray; //GPU array is needed for copying the GPU array pointers to the GPU. Not possible otherwise because CellArray is a host pointer to an array of host objects with host pointers to the device.
	FourierHelpers FH[1];
	fflush(stdout);

	int i;
	long int Schrittzahl, seed = ((long int)time(NULL) + getpid() + (long int) clock());
	//seed = 1460718856; //constant value for testing
	srand48(seed);

	double Zeit, ZeitAlt, Startzeit, Schrittweite, MaxWeiteSp=1e-2, Fehlersumme=0;
	int    ZeitSchwelle = 0, Bildzaehler = 1;

	double *GPU_Arr, Area[4];						
	char buffer[128];
	FILE *Schrittweiten;
	FILE *Params, *Error;

	Schrittweiten = fopen("./Output/Daten/" ZZDD "/SMPR" FFDD "WeitenDelt" ZZDD ".txt","w");

	Params = fopen("./Output/Daten/" ZZDD "/SMPR" FFDD "ParameterWerteDelt" ZZDD ".txt","w");
	fprintf(Params,"Parameterwerte\nAlpha:\t%f\nBeta:\t%f\nepsilon:\t%f\nKappa:\t%f\nOmega:\t%f\nOmega_0:\t%f\nk:\t%f\nv_a:\t%f\n", VAR_alpha, VAR_beta, VAR_epsilon, VAR_kappa, VAR_wd, VAR_w0, VAR_kd, VAR_va);
	fprintf(Params,"D_Na:\t%f\nD_Phase:\t%f\nCNuk:\t%f\nSysL:\t%f\nStartradius:\t%f\nGitterpunkte:\t%dx%d\nFehlerschwelle SWS:\t%e\nSeed:\t%ld\nZoom:\t%.2f\n", VAR_DiNa, VAR_DiPh, VAR_CNuk, VAR_SysL, Startradius, GPZX, GPZY, Fehlerschwelle, seed, ZOOMFACTOR);
	fprintf(Params,"Phasefield:\t%d\nNucDegrad:\t%d\nAdaptiveSteps:\t%d\nNormalizePhasGrad:\t%d\nTimeSaveStep:\t%f\nConcSaveInt:\t%d\nInitState:\t%d\n", PHASEFIELDFLAG, NUCLEATORDEGRADFLAG, STEPSIZEFLAG, MEMBGRADIENTNORMALIZER, ZEITINTERVALL, BILDINTERVALL,STATELOADERFLAG);
	fprintf(Params,"TimeStep:\t%f\nEndTime:\t%f\nTimeOffset:\t%f\nDensitySaveInterval:\t%d\n",ZEITINTERVALL, EndZeit, ZEITOFFSET,BILDINTERVALL);
	fclose(Params);

	fflush(stdout);
	int RetVal = 0;



	//Initialize cells
	//Needs to be done BEFORE copying pointers to the device because it allocates space.
	RetVal = min(RetVal, CellArray[0].Initializer(round(GPZX/2),round(GPZY/2), 0));	
	
	//Error check for Initializer
	if(RetVal < 0)	return(0);

	//Initialize constants
	Schrittweite = 1e-6;
	Startzeit    = clock();
	Zeit         = 0;
	ZeitAlt      = 0;
	Schrittzahl  = 0;
	cudaMalloc((void**) &GPU_Arr, 		ByteZahl_Konzentrationen_double);
	cudaMalloc((void**) &GPU_CellArray, 	sizeof(SingleCell));
	cudaMemcpy(GPU_CellArray,  CellArray,   sizeof(SingleCell), cudaMemcpyHostToDevice); 	//Copy pointers to the cuda stuff to the device. Needs to be done AFTER INITIALIZING CellArray!
	fflush(stdout);

	while(Zeit < EndZeit){

#if STEPSIZEFLAG > 0
		// STEP SIZE CONTROL
		//Standard midpoint rule
		CellArray[0].SpectralDerivatives(FH, 0);
		CellArray[0].TotalValues(handleResultToDevice, 0);
		cudaDeviceSynchronize();

		UPDATER_CELLCOUNTSTART 0, 1, UPDATER_CELLCOUNTEND	
		CellArray[0].EulerStep(0, 2, Schrittweite/2);
		cudaDeviceSynchronize();

		CellArray[i].SpectralDerivatives(FH, 2);
		CellArray[i].TotalValues(handleResultToDevice, 2);
		cudaDeviceSynchronize();

		STEPUPDATER_CELLCOUNTSTART 2, 0, 2, Schrittweite, UPDATER_CELLCOUNTEND
		
		//2 Midpoint rule steps with half step size. Halve it until the error threshold is met.
		do{
			CellArray[0].EulerStep(0, 3, Schrittweite/4);
			CellArray[0].SpectralDerivatives(FH, 3);
			CellArray[0].TotalValues(handleResultToDevice, 3);
			cudaDeviceSynchronize();

			STEPUPDATER_CELLCOUNTSTART 3, 0, 3, Schrittweite/2, UPDATER_CELLCOUNTEND

			CellArray[0].SpectralDerivatives(FH, 3);
			CellArray[0].TotalValues(handleResultToDevice, 3);
			cudaDeviceSynchronize();

			STEPUPDATER_CELLCOUNTSTART 3, 1, 0, Schrittweite/4, UPDATER_CELLCOUNTEND

			CellArray[0].SpectralDerivatives(FH, 4);
			CellArray[0].TotalValues(handleResultToDevice, 4);
			cudaDeviceSynchronize();

			STEPUPDATER_CELLCOUNTSTART 4, 0, 1, Schrittweite/2, UPDATER_CELLCOUNTEND
	
			cudaDeviceSynchronize();

			Fehlersumme = CellArray[0].NumericStepError(handleResultToHost, GPU_Arr);
			
			if((Fehlersumme > Fehlerschwelle)||(Schrittweite>MaxWeiteSp)){
				Schrittweite/=2.0;
				CellArray[0].StepFilaUpdate();
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
		CellArray[0].Update();
		
		Zeit+=Schrittweite;
		Schrittweite*=1.01;		
		Schrittzahl++;

		//frequently save step size and check if a stationary state is reached.
		if(Schrittzahl%1000==0){
			fprintf(Schrittweiten,"%ld\t%g\t%g\n",Schrittzahl, Schrittweite, Fehlersumme); fflush(Schrittweiten);

			//"Fehlersumme" ist hier das Maximum, spart ein paar Speicherplätze. Gesucht wird das Maximum von d/dt T im letzten Simulationsschritt. Passiert nichts, wird die Simulation direkt unterbrochen.
			Fehlersumme = 0;				
			Fehlersumme = max(Fehlersumme, CellArray[0].FindMaxLastDeviation(handleResultToHost));
			
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
		CellArray[0].SpectralDerivatives(FH, 0);
		CellArray[0].TotalValues(handleResultToDevice, 0);

		STEPUPDATER_CELLCOUNTSTART 0, 4, 0, Schrittweite, UPDATER_CELLCOUNTEND
		CellArray[0].Update();

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
				CellArray[0].PhasToHost();
				CellArray[0].SaveCenterOfMass(ZeitSchwelle, Zeit);
				
			#endif
			
			//save concentrations
			if(((ZeitSchwelle % BILDINTERVALL)==0)){
				#if PHASEFIELDFLAG > 0
				     	sprintf(buffer,"./Output/Daten/" ZZDD "/SMPR" FFDD "PhasenfeldA" ZZDD "B%d.txt",Bildzaehler);
					CellArray[0].SavePhas(buffer);
				#endif
				CellArray[0].FilaToHost();
				sprintf(buffer,"./Output/Daten/" ZZDD "/SMPR" FFDD "FilamenteA" ZZDD "B%d.txt", Bildzaehler);
				CellArray[0].SaveFila(buffer);
				//CellArray[0].NuksToHost();
				#if PHASEFIELDFLAG > 0
					#if NUCLEATORDEGRADFLAG > 0
						cudaMemcpy(Area,           CellArray[0].GPU_Area,       3*sizeof(double),              cudaMemcpyDeviceToHost);
				 		printf("\tCell %d: Area %.1f, CoM: (%.2f,%.2f), N_Tot: %.2f\n", 0, Area[0], CellArray[0].SchwerpunktX[ZeitSchwelle], CellArray[0].SchwerpunktY[ZeitSchwelle],Area[1]+Area[2]);
					#else
						cudaMemcpy(Area,           CellArray[0].GPU_Area,       sizeof(double),                cudaMemcpyDeviceToHost);
				 		printf("\tCell %d: Area %.1f, CoM: (%.2f,%.2f)\n", 0, Area[0], CellArray[0].SchwerpunktX[ZeitSchwelle], CellArray[0].SchwerpunktY[ZeitSchwelle]);
					#endif				
				#endif

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
	printf("%f\n",CellArray[0].Fila[150*256+80]);
	CellArray[0].AllToHost();
	sprintf(buffer,"./Output/Daten/" ZZDD "/SMPR" FFDD "EndeFilaA" ZZDD ".txt");
	CellArray[0].EndSave(buffer, 0);	//print the entire array. 0=Fila,1=NukA,2=NukI,3=PolX,4=PolY,5=Phas, everything else = nothing
	sprintf(buffer,"./Output/Daten/" ZZDD "/SMPR" FFDD "EndeNukAA" ZZDD ".txt");
	CellArray[0].EndSave(buffer, 1);
	sprintf(buffer,"./Output/Daten/" ZZDD "/SMPR" FFDD "EndeNukIA" ZZDD ".txt");
	CellArray[0].EndSave(buffer, 2);
	sprintf(buffer,"./Output/Daten/" ZZDD "/SMPR" FFDD "EndePolXA" ZZDD ".txt");
	CellArray[0].EndSave(buffer, 3);
	sprintf(buffer,"./Output/Daten/" ZZDD "/SMPR" FFDD "EndePolYA" ZZDD ".txt");
	CellArray[0].EndSave(buffer, 4);
	sprintf(buffer,"./Output/Daten/" ZZDD "/SMPR" FFDD "EndePhasA" ZZDD ".txt");
	CellArray[0].EndSave(buffer, 5);
	printf("%f\n",CellArray[0].Fila[150*256+80]);
	

	//Cleanup
	CellArray[0].FreeMemory();
	
        usleep(1);

	cudaFree(GPU_Arr);
	cudaFree(GPU_CellArray);
	cublasDestroy(handle);
	cublasDestroy(handleResultToDevice);
	cublasDestroy(handleResultToHost);
	fclose(Schrittweiten);

	return 0;
}
