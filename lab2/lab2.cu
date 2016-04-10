#include "lab2.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <vector>
#include <fstream>
#include <iostream>
#include <numeric>
#include "../utils/SyncedMemory.h"
static const unsigned W = 640;
static const unsigned H = 480;
static const unsigned NFRAME = 240;

using namespace std;

//////////////initial////


struct Lab2VideoGenerator::Impl {
	int t = 0;
	vector<int> p;
};

__device__ double fade(double t) { 
	return t * t * t * (t * (t * 6 - 15) + 10);
}

__device__ double lerp(double t, double a, double b) { 
	return a + t * (b - a); 
}

__device__ double grad(int hash, double x, double y, double z) {
	int h = hash & 15;
	// Convert lower 4 bits of hash inot 12 gradient directions
	double u = h < 8 ? x : y,
		   v = h < 4 ? y : h == 12 || h == 14 ? x : z;
	return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

__global__ void noise(double z,const int* p ,uint8_t* yuv) {
	//printf("in\n");
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	double x = (double)(idx%W)/W;
	double y = (double)(idx/W)/H;
	// Find the unit cube that contains the point
	int X = (int) floor(x) & 255;
	int Y = (int) floor(y) & 255;
	int Z = (int) floor(z) & 255;

	// Find relative x, y,z of point in cube
	x -= floor(x);
	y -= floor(y);
	z -= floor(z);

	// Compute fade curves for each of x, y, z
	double u = fade(x);
	double v = fade(y);
	double w = fade(z);

	// Hash coordinates of the 8 cube corners
	int A = p[X] + Y;
	int AA = p[A] + Z;
	int AB = p[A + 1] + Z;
	int B = p[X + 1] + Y;
	int BA = p[B] + Z;
	int BB = p[B + 1] + Z;

	// Add blended results from 8 corners of cube
	double res = lerp(w, lerp(v, lerp(u, grad(p[AA], x, y, z), grad(p[BA], x-1, y, z)), lerp(u, grad(p[AB], x, y-1, z), grad(p[BB], x-1, y-1, z))),	lerp(v, lerp(u, grad(p[AA+1], x, y, z-1), grad(p[BA+1], x-1, y, z-1)), lerp(u, grad(p[AB+1], x, y-1, z-1),	grad(p[BB+1], x-1, y-1, z-1))));
	res = (res + 1.0)/2.0;
	//origin
	//yuv[idx] = floor((res + 1.0)/2.0*255);
	//yuv[idx] = (res + 1.0)/2.0;
	//typical noise
	//wood like
	double n = (60+30*z) * res;
	n = n - floor(n);

	yuv[idx] = uint8_t(floor(150 * n));
			
}


Lab2VideoGenerator::Lab2VideoGenerator(): impl(new Impl) {
	unsigned int seed = 237;
	
	impl->p.resize(256);

	// Fill p with values from 0 to 255
	std::iota(impl->p.begin(), impl->p.end(), 0);

	// Initialize a random engine with seed
	std::default_random_engine engine(seed);

	// Suffle  using the above random engine
	std::shuffle(impl->p.begin(), impl->p.end(), engine);

	// Duplicate the permutation vector
	impl->p.insert(impl->p.end(), impl->p.begin(), impl->p.end());
	
	printf("output test\n");

	printf("Initialize finish\n" );
	
}

Lab2VideoGenerator::~Lab2VideoGenerator() {

}

void Lab2VideoGenerator::get_info(Lab2VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 24;
	info.fps_d = 1;
};


void Lab2VideoGenerator::Generate(uint8_t *yuv) {
	//cudaMemset(yuv, (impl->t)*255/NFRAME, W*H);
	
	
	//test
	
	
		
	

	int n = impl->p.size();
	//printf("length%d\n", sizeof(data));
	int *p_gpu;
	cudaMalloc(&p_gpu, sizeof(int)*n);
	SyncedMemory<int> p_sync(impl->p.data(), p_gpu, n);
	p_sync.get_cpu_wo();
	//const int* temp = impl->s->get_cpu_ro();
	if(impl->t==0){
		for(int i=0;i<256;i++){			
			printf("%d ", p_sync.get_cpu_ro()[i]);
		}
	}
	//printf("output test\n");
	
	noise<<<W*H/64,64>>>((double)impl->t/NFRAME,p_sync.get_gpu_ro(),yuv);
	cudaMemset(yuv+W*H, 91, W*H/4);
	cudaMemset(yuv+W*H+W*H/4,167 , W*H/4);
	//printf("generate frame: %d\n", impl->t);

	++(impl->t);
}
