#ifndef CONSTANTS__H
#define CONSTANTS__H 1


#define PI 3.141592653589793

#define PHASEFIELDFLAG 0	//turns phasefield on (1) or off (0)
#define ZZDD "TOKEN"
#define STEPSIZEFLAG 1		//0 = standard euler step, 1 = step size controled mid point rule
#define ZOOMFACTOR 1.0		//Magnifies the system. In this version, it's identical to changing both the system length and the cell area and radius.

#define NUCLEATORDEGRADFLAG 0	//turns on external degredation of nucleating proteins (1 = on), with a balancing part.
#define MEMBGRADIENTNORMALIZER 0 //normalizes phasefield gradient vectors
#define STATELOADERFLAG 0 	 //0 = generate random initial condition, 1 = load initial values from file in State folder

#define GITTERPUNKTZAHLX 256
#define GITTERPUNKTZAHLY 256
#define GPZX 256
#define GPZY 256
#define RASTERGROESSE (GITTERPUNKTZAHLX*GITTERPUNKTZAHLY)

#define Startradius (PHASEFIELDFLAG ? 32.0 * ZOOMFACTOR : 512.0)

//SettingStart
#define EndZeit 150.0
#define ZEITINTERVALL 0.05	
#define BILDINTERVALL 10
#define ZEITOFFSET 0.0		//needs to be smaller than EndZeit. Nothing is saved before the simulation passes this time.
//SettingEnd


//ParamStart
#define VAR_DiAkt 0.0 //(changes have no effect)
#define VAR_DiNa 0.04
#define VAR_DiPh 0.005

#define VAR_va 0.46
#define VAR_wd 0.43
#define VAR_w0 0.006
#define VAR_kd 175.0

#define VAR_alpha 588.0
#define VAR_beta 0.006
#define VAR_kappa 118.0
#define VAR_epsilon 8.0

#define VAR_SysLen 1.3
#define VAR_CNuk 700.0	
//ParamEnd

#define VAR_SysL (VAR_SysLen / ZOOMFACTOR)

#define CONST_Unit_Area ((VAR_SysL * VAR_SysL)/((double)GPZX * GPZY))
#define VAR_MeanVol (PHASEFIELDFLAG ? (PI * Startradius * Startradius) : (GPZX * GPZY))	
#define VAR_MeanCirc (2 * PI * Startradius * VAR_SysL / GPZX)

#define Fehlerschwelle 1e-8	//Threshold of the adaptive step size control. Maximum value allowed for the largest deviation in the actin concentration.
#define FFDD "G8"

#define GPUID 0			//GPUs automatically receive an ID of 0 in most cases. If multiple GPUs are used and not managed by another piece of software, this needs to be changed

/* a real-to-complex FFT just creates n/2 + 1 complex coefficients, or (n + 2) values (real and imaginary parts)*/
#define KOEFFZAHL_IM_FREQUENZRAUM ((GITTERPUNKTZAHLX/2+1)*GITTERPUNKTZAHLY)

#define CONST_xSchrittweite (VAR_SysL / (double) GPZX)
#define TotalNuk (VAR_MeanVol * VAR_CNuk)

#define ByteZahl_Koeff_Frequenz_double  (KOEFFZAHL_IM_FREQUENZRAUM * sizeof(double))
#define ByteZahl_Konzentrationen_double (RASTERGROESSE * sizeof(double))

#endif

