#ifndef CONSTANTS__H
#define CONSTANTS__H 1


#define PI 3.141592653589793

#define GPU_Iterationen 3000
#define Anzahl_der_Ausgabewerte 200

#define PHASEFIELDFLAG 0	//turns phasefield on (1) or off (0)
//!!!!!!!! ONLY MAX 7 CELLS ALLOWED IF CHOOSING OBSTACLEFLAG 4 !!!!!!!!
#define ZZDD "TOKEN"
#define OBSTACLETYPE (PHASEFIELDFLAG ? 0 : 0)		//... : 0 = none, 1 = circle, 2 = square, 3 = "star", 4 = hexagonal maze, 5 = channel, 6 = Wells. Always 0 without a phasefield.
#define STEPSIZEFLAG 1		//0 = standard euler step, 1 = step size controled mid point rule
#define ZOOMFACTOR 1.0		//magnifies systems with a solid border, such as circle and square. Unphysical behaviour for star and hexagonal obstacles.
#define CELLCOUNT 1		//Number of Cells.

#define OBSTACLEDIVISOR 4	//Tells how steep the obstacle phasefield is. 4 was standard and could be far too wide. 2 was standard for channel and wells.
#define PHASDEGRAD	8.0	//constant for phasefield degradation due to overlaps. Used to be 15.0 for most stuff not taking place in wells.

#define NUCLEATORDEGRADFLAG 0	//turns on external degredation of nucleating proteins (1 = on), with a balancing part.
#define MEMBTENSFLAG 0		//turns on membrane tension / curvature terms (1 = on)
#define MEMBGRADIENTNORMALIZER 0 //normalizes phasefield gradient vectors
#define STATELOADERFLAG 0 	 //0 = generate random initial condition, 1 = load initial values from file in State folder
#define OBSDYNFLAG 0		 //0 = default, 1 = protein dynamics also scale with obstacle phase field

#define GITTERPUNKTZAHLX 256
#define GITTERPUNKTZAHLY 256
#define GPZX 256
#define GPZY 256
#define RASTERGROESSE (GITTERPUNKTZAHLX*GITTERPUNKTZAHLY)

#define Startradius (PHASEFIELDFLAG ? 45.0 * ZOOMFACTOR : 512.0)

//SettingStart
#define EndZeit 40.0
#define ZEITINTERVALL 0.05	
#define BILDINTERVALL 5
#define ZEITOFFSET 0.0		//needs to be smaller than EndZeit. Nothing is saved before the simulation passes this time.
//SettingEnd


//ParamStart
#define VAR_DiAkt 0.0 //(changes have no effect)
#define VAR_DiNa 0.120
#define VAR_DiPh 0.014

#define VAR_va 15.0
#define VAR_wd 0.43
#define VAR_w0 0.015
#define VAR_kd 175.0

#define VAR_alpha 588.0
#define VAR_beta 0.006
#define VAR_kappa 118.0
#define VAR_epsilon 8.0

#define VAR_SysLen 1.0
#define VAR_CNuk 700.0	
#define VAR_Acti 0.0
#define VAR_Cortex 0.0
#define VAR_MembTensMod 0.0
//ParamEnd

#define VAR_SysL (VAR_SysLen / ZOOMFACTOR)

#define CONST_Unit_Area ((VAR_SysL * VAR_SysL)/((double)GPZX * GPZY))
#define VAR_MeanVol (PHASEFIELDFLAG ? (PI * Startradius * Startradius) : (GPZX * GPZY))	
#define VAR_MeanCirc (2 * PI * Startradius * VAR_SysL / GPZX)

#define InfluxStrength 0.0

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

