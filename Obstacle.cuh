#ifndef OBSTACLE__H
#define OBSTACLE__H 1

//NEEDS CONSTANTS AND DYNAMICS FIRST
class FourierHelpers;

class Obstacles{
	public:
	//Host&GPU Obstacle
	double *GPU_Obstacle, *GPU_ObsDiff, *Obstacle, *ObsDiff;

	//Functions
	Obstacles(FourierHelpers *FH);		//Constructor, allocates GPU memory, defines Obstacle and calculates derivatives
	~Obstacles();				//Destructor, frees GPU memory
};

double DistanceFunction(double s1, double s2, double x1, double x2, double y1, double y2);
void LineDrawer20(double *Phas, double p1, double p2, double q1, double q2);


#endif
