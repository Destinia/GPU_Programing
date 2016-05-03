#include "lab3.h"
#include <cstdio>
#include <cmath>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht and xt < wt and mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
			output[curb*3+0] = target[curt*3+0];
			output[curb*3+1] = target[curt*3+1];
			output[curb*3+2] = target[curt*3+2];
		}
	}
}

__global__ void Upsample(
	float *target,
	float *up,
	const int wt, const int ht,
	const int wt_o,const int ht_o
	)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	//if(yt>=ht||xt>=wt)return;
	const int curt = wt*yt+xt;
	const int curu = 2*wt_o*yt+2*xt;
	if(curt>=wt*ht)return;

	int t1,t2,t3,t4;

	if(2*xt<wt_o && 2*yt<ht_o) {
		t1 = curu;
		t2 = curu+1;
		t3 = curu+wt_o;
		t4 = curu+wt_o+1;

	}
	else if(2*xt>=wt_o && 2*yt>=ht_o) {
		t1 = t2 = t3 = t4 = curu;
		//printf("error up %d\n",curt);

	}
	else if(2*xt>=wt_o) {
		t1 = curu;
		t2 = curu;
		t3 = curu+wt_o;
		t4 = curu+wt_o;
		//printf("error up %d\n",curt);


	}
	else if(2*yt>=ht_o) {
		t1 = curu;
		t2 = curu+1;
		t3 = curu;
		t4 = curu+1;
		//printf("error up %d\n",curt);

	}
	else{
		printf("error up %d\n",curt);
	}

	up[(3*t1)+0] = target[(3*curt)+0];up[(3*t2)+0] = target[(3*curt)+0];up[(3*t3)+0] = target[(3*curt)+0];up[(3*t4)+0] = target[(3*curt)+0];
	up[(3*t1)+1] = target[(3*curt)+1];up[(3*t2)+1] = target[(3*curt)+1];up[(3*t3)+1] = target[(3*curt)+1];up[(3*t4)+1] = target[(3*curt)+1];
	up[(3*t1)+2] = target[(3*curt)+2];up[(3*t2)+2] = target[(3*curt)+2];up[(3*t3)+2] = target[(3*curt)+2];up[(3*t4)+2] = target[(3*curt)+2];
	

}


__global__ void Downsample(
	const float *target,
	float *down,
	const int wt, const int ht,
	const int wt_o,const int ht_o
	)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	//if(yt>=ht||xt>=wt)return;
	const int curd = wt*yt+xt;
	const int curt = 2*wt_o*yt+2*xt;
	if(curd>=wt*ht)return;

	int t1,t2,t3,t4;

	if(2*xt<wt_o && 2*yt<ht_o) {
		t1 = curt;
		t2 = curt+1;
		t3 = curt+wt_o;
		t4 = curt+wt_o+1;

	}
	else if(2*xt>=wt_o && 2*yt>=ht_o) {
		t1 = t2 = t3 = t4 = curt;
		//printf("error down1 %d\n",curd);

	}
	else if(2*xt>=wt_o) {
		t1 = curt;
		t2 = curt;
		t3 = curt+wt_o;
		t4 = curt+wt_o;
		//printf("error down2 %d\n",curd);
	}
	else if(2*yt>=ht_o) {
		t1 = curt;
		t2 = curt+1;
		t3 = curt;
		t4 = curt+1;
		//printf("error down3 %d\n",curd);
	}
	else{
		printf("error down4 %d\n",curd);
	}

	down[(3*curd)+0] = (target[(3*t1)+0] + target[(3*t2)+0] + target[(3*t3)+0] + target[(3*t4)+0])/4;
	down[(3*curd)+1] = (target[(3*t1)+1] + target[(3*t2)+1] + target[(3*t3)+1] + target[(3*t4)+1])/4;
	down[(3*curd)+2] = (target[(3*t1)+2] + target[(3*t2)+2] + target[(3*t3)+2] + target[(3*t4)+2])/4;

}

__global__ void Downsample_mask(
	const float *target,
	float *down,
	const int wt, const int ht,
	const int wt_o,const int ht_o
	)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curd = wt*yt+xt;
	const int curt = 2*wt_o*yt+2*xt;

	int t1,t2,t3,t4;

	if(2*xt<wt_o && 2*yt<ht_o) {
		t1 = curt;
		t2 = curt+1;
		t3 = curt+wt_o;
		t4 = curt+wt_o+1;

	}
	else if(2*xt>=wt_o && 2*yt>=ht_o) {
		t1 = t2 = t3 = t4 = curt;
	}
	else if(2*xt>=wt_o) {
		t1 = curt;
		t2 = curt;
		t3 = curt+wt_o;
		t4 = curt+wt_o;

	}
	else if(2*yt>=ht_o) {
		t1 = curt;
		t2 = curt+1;
		t3 = curt;
		t4 = curt+1;

	}
	else{
		printf("error\n");
	}

	down[curd] = target[t1] + target[t2] + target[t3] + target[t4];
	
	if(down[curd]>=255.0*2){
		down[curd]=255.0;
	}
	else{
		down[curd]=0.0;
	}
}

__global__ void merge(const float *target,float *output,const int wt,const int ht){
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curd = wt*yt+xt;
	if(abs(target[3*curd+0]+target[3*curd+1]+target[3*curd+2]-output[3*curd+0]+output[3*curd+1]+output[3*curd+2])<100.0){
	output[3*curd+0] = target[3*curd+0];
	output[3*curd+1] = target[3*curd+1];
	output[3*curd+2] = target[3*curd+2];
	//output[3*curd+0] = (output[3*curd+0]+target[3*curd+0])/2;
	//output[3*curd+1] = (output[3*curd+1]+target[3*curd+1])/2;
	//output[3*curd+2] = (output[3*curd+2]+target[3*curd+2])/2;
	}
	//else {
	//}
}

__global__	void CalculateFixed(
	const float *background,
	const float *target,
	const float *mask,
	float *fixed,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	if(yt>=ht||xt>=wt)return;
	const int curt = wt*yt+xt;
	const int Tn = (yt-1<0)? curt:curt-wt, Ts = (ht<=yt+1)? curt:curt+wt, Tw = (xt-1<0)? curt:curt-1, Te = (wt<=xt+1)? curt:curt+1;

	fixed[curt*3+0] = 4*target[curt*3+0] - target[(Tn)*3+0] - target[(Ts)*3+0] - target[(Tw)*3+0] - target[(Te)*3+0];
	fixed[curt*3+1] = 4*target[curt*3+1] - target[(Tn)*3+1] - target[(Ts)*3+1] - target[(Tw)*3+1] - target[(Te)*3+1];
	fixed[curt*3+2] = 4*target[curt*3+2] - target[(Tn)*3+2] - target[(Ts)*3+2] - target[(Tw)*3+2] - target[(Te)*3+2];

	if(yt-1<0) {
		const int yb = (oy+yt-1<0)? oy+yt:oy+yt-1, xb = ox+xt;
		const int curb = wb*yb+xb;

		fixed[curt*3+0] += background[curb*3+0];
		fixed[curt*3+1] += background[curb*3+1];
		fixed[curt*3+2] += background[curb*3+2];
	}
	else if (mask[curt-wt] < 127.0f) {
		const int yb = oy+yt-1, xb = ox+xt;
		const int curb = wb*yb+xb;

		fixed[curt*3+0] += background[curb*3+0];
		fixed[curt*3+1] += background[curb*3+1];
		fixed[curt*3+2] += background[curb*3+2];
	}
	if(ht<=yt+1) {
		const int yb = (hb<=oy+yt+1)? oy+yt:oy+yt+1, xb = ox+xt;
		const int curb = wb*yb+xb;

		fixed[curt*3+0] += background[curb*3+0];
		fixed[curt*3+1] += background[curb*3+1];
		fixed[curt*3+2] += background[curb*3+2];
	}
	else if (mask[curt+wt] < 127.0f) {
		const int yb = oy+yt+1, xb = ox+xt;
		const int curb = wb*yb+xb;

		fixed[curt*3+0] += background[curb*3+0];
		fixed[curt*3+1] += background[curb*3+1];
		fixed[curt*3+2] += background[curb*3+2];
	}

	if(xt-1<0) {
		const int yb = oy+yt, xb = (ox+xt-1<0)? ox+xt:ox+xt-1;
		const int curb = wb*yb+xb;

		fixed[curt*3+0] += background[curb*3+0];
		fixed[curt*3+1] += background[curb*3+1];
		fixed[curt*3+2] += background[curb*3+2];

	}
	else if (mask[curt-1] < 127.0f) {
		const int yb = oy+yt, xb = ox+xt-1;
		const int curb = wb*yb+xb;
		fixed[curt*3+0] += background[curb*3+0];
		fixed[curt*3+1] += background[curb*3+1];
		fixed[curt*3+2] += background[curb*3+2];
	}
	if(wt<=xt+1) {
		const int yb = oy+yt, xb = (wb<=ox+xt+1)? ox+xt:ox+xt+1;
		const int curb = wb*yb+xb;

		fixed[curt*3+0] += background[curb*3+0];
		fixed[curt*3+1] += background[curb*3+1];
		fixed[curt*3+2] += background[curb*3+2];
	}
	else if (mask[curt+1] < 127.0f) {
		const int yb = oy+yt, xb = ox+xt+1;
		const int curb = wb*yb+xb;
		fixed[curt*3+0] += background[curb*3+0];
		fixed[curt*3+1] += background[curb*3+1];
		fixed[curt*3+2] += background[curb*3+2];
	}
}

__global__ void PoissonImageCloningIteration(
	float *fixed,
	const float *mask,
	float *buf1,
	float *buf2,
	const int wt,const int ht
	)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	if(yt>=ht||xt>=wt)return;
	const int curt = wt*yt+xt;
	buf2[curt*3+0] = fixed[curt*3+0];//+buf1[(curt-wt)*3+0]+buf1[(curt+wt)*3+0]+buf1[(curt-1)*3+0]+buf1[(curt+1)*3+0];
	buf2[curt*3+1] = fixed[curt*3+1];//+buf1[(curt-wt)*3+1]+buf1[(curt+wt)*3+1]+buf1[(curt-1)*3+1]+buf1[(curt+1)*3+1];
	buf2[curt*3+2] = fixed[curt*3+2];//+buf1[(curt-wt)*3+2]+buf1[(curt+wt)*3+2]+buf1[(curt-1)*3+2]+buf1[(curt+1)*3+2];

	if(yt-1>=0) {
		if (mask[curt-wt] > 127.0f) {
			const int cur = curt-wt; 
			buf2[curt*3+0] += (buf1[cur*3+0]);
			buf2[curt*3+1] += (buf1[cur*3+1]);
			buf2[curt*3+2] += (buf1[cur*3+2]);
		}
	}

	if(ht>yt+1) {	
		if (mask[curt+wt] > 127.0f) {
			const int cur = curt+wt;
			buf2[curt*3+0] += (buf1[cur*3+0]);
			buf2[curt*3+1] += (buf1[cur*3+1]);
			buf2[curt*3+2] += (buf1[cur*3+2]);
		}
	}

    if(xt-1>=0) {   
		if (mask[curt-1] > 127.0f) {
			const int cur = curt-1;
			buf2[curt*3+0] += (buf1[cur*3+0]);
			buf2[curt*3+1] += (buf1[cur*3+1]);
			buf2[curt*3+2] += (buf1[cur*3+2]);
		}
	}

	if(wt>xt+1) {
		if (mask[curt+1] > 127.0f) {
			const int cur = curt+1;
			buf2[curt*3+0] += (buf1[cur*3+0]);
			buf2[curt*3+1] += (buf1[cur*3+1]);
			buf2[curt*3+2] += (buf1[cur*3+2]);
		}
	}
	
	
	buf2[curt*3+0] /=4;
	buf2[curt*3+1] /=4;
	buf2[curt*3+2] /=4;

}

void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	//cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	//SimpleClone<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
	//	background, target, mask, output,
	//	wb, hb, wt, ht, oy, ox
	//);
	// set up
	float *fixed, *buf1, *buf2;
	cudaMalloc(&fixed, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf1, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf2, 3*wt*ht*sizeof(float));

	//downsample

	//set downsample memory
	
	const int wt_d = CeilDiv(wt,2);
	const int ht_d = CeilDiv(ht,2);
	const int wb_d = CeilDiv(wb,2);
	const int hb_d = CeilDiv(hb,2);
	const int ox_d = CeilDiv(ox,2);
	const int oy_d = CeilDiv(oy,2);

	float *fixed_d,*buf1_d,*buf2_d,*mask_d, *bkg_d;
	cudaMalloc(&fixed_d, 3*wt_d*ht_d*sizeof(float));
	cudaMalloc(&buf1_d, 3*wt_d*ht_d*sizeof(float));
	cudaMalloc(&buf2_d, 3*wt_d*ht_d*sizeof(float));
	cudaMalloc(&mask_d, wt_d*ht_d*sizeof(float));
	cudaMalloc(&bkg_d, 3*wb_d*hb_d*sizeof(float));
	//cudaMalloc(&bkg_u,3*wb*hb*sizeof(float));

	const int wt_4 = CeilDiv(wt_d,2);const int wt_8 = CeilDiv(wt_4,2);
	const int ht_4 = CeilDiv(ht_d,2);const int ht_8 = CeilDiv(ht_4,2);
	const int wb_4 = CeilDiv(wb_d,2);const int wb_8 = CeilDiv(wb_4,2);
	const int hb_4 = CeilDiv(hb_d,2);const int hb_8 = CeilDiv(hb_4,2);
	const int ox_4 = CeilDiv(ox_d,2);const int ox_8 = CeilDiv(ox_4,2);
	const int oy_4 = CeilDiv(oy_d,2);const int oy_8 = CeilDiv(oy_4,2);

	float *fixed_4,*buf1_4,*buf2_4,*mask_4, *bkg_4;
	float *fixed_8,*buf1_8,*buf2_8,*mask_8, *bkg_8;

	cudaMalloc(&fixed_4, 3*wt_4*ht_4*sizeof(float));
	cudaMalloc(&buf1_4, 3*wt_4*ht_4*sizeof(float));
	cudaMalloc(&buf2_4, 3*wt_4*ht_4*sizeof(float));
	cudaMalloc(&mask_4, wt_4*ht_4*sizeof(float));
	cudaMalloc(&bkg_4, 3*wb_4*hb_4*sizeof(float));
	cudaMalloc(&fixed_8, 3*wt_8*ht_8*sizeof(float));
	cudaMalloc(&buf1_8, 3*wt_8*ht_8*sizeof(float));
	cudaMalloc(&buf2_8, 3*wt_8*ht_8*sizeof(float));
	cudaMalloc(&mask_8, wt_8*ht_8*sizeof(float));
	cudaMalloc(&bkg_8, 3*wb_8*hb_8*sizeof(float));
	
	/*
	fixed_4 = fixed_d + wt_d*ht_d; fixed_8 = fixed_4 + wt_4*ht_4;
	buf1_4  = buf1_d + wt_d*ht_d;  buf1_8  = buf1_4 + wt_4*ht_4;
	buf2_4  = buf2_d + wt_d*ht_d;  buf2_8  = buf2_4 + wt_4*ht_4;
	mask_4  = mask_d + wt_d*ht_d;  mask_8  = mask_4 + wt_4*ht_4;
	bkg_4  = bkg_d + wb_d*hb_d;    bkg_8  = bkg_4 + wb_d*hb_4;
	*/




	
	// initialize the iteration
	dim3 gdim(CeilDiv(wt,32), CeilDiv(ht,16)), bdim(32,16);

	dim3 gdim_d(CeilDiv(wt_d,32), CeilDiv(ht_d,16));
	dim3 gdim_bd(CeilDiv(wb_d,32),CeilDiv(hb_d,16));
	dim3 gdim_4(CeilDiv(wt_4,32), CeilDiv(ht_4,16));
	dim3 gdim_b4(CeilDiv(wb_4,32),CeilDiv(hb_4,16));
	dim3 gdim_8(CeilDiv(wt_8,32), CeilDiv(ht_8,16));
	dim3 gdim_b8(CeilDiv(wb_8,32),CeilDiv(hb_8,16));

	CalculateFixed<<<gdim, bdim>>>(
	background, target, mask, fixed,
	wb, hb, wt, ht, oy, ox
	);

	//X2
	//cudaMemcpy(buf1, target, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice);
	Downsample<<<gdim_bd, bdim>>>(background,bkg_d,wb_d,hb_d,wb,hb);
	Downsample<<<gdim_d, bdim>>>(target,buf1_d,wt_d,ht_d,wt,ht);
	Downsample_mask<<<gdim_d, bdim>>>(mask,mask_d,wt_d,ht_d,wt,ht);

	CalculateFixed<<<gdim_d, bdim>>>(
	bkg_d, buf1_d, mask_d, fixed_d,
	wb_d, hb_d, wt_d, ht_d, oy_d, ox_d
	);

	//X4
	
	Downsample<<<gdim_b4, bdim>>>(bkg_d,bkg_4,wb_4,hb_4,wb_d,hb_d);
	Downsample<<<gdim_4, bdim>>>(buf1_d,buf1_4,wt_4,ht_4,wt_d,ht_d);
	Downsample_mask<<<gdim_4, bdim>>>(mask_d,mask_4,wt_4,ht_4,wt_d,ht_d);
	
	CalculateFixed<<<gdim_d, bdim>>>(
	bkg_4, buf1_4, mask_4, fixed_4,
	wb_4, hb_4, wt_4, ht_4, oy_4, ox_4
	);
	
	//X8
	Downsample<<<gdim_b8, bdim>>>(bkg_4,bkg_8,wb_8,hb_8,wb_4,hb_4);
	Downsample<<<gdim_8, bdim>>>(buf1_4,buf1_8,wt_8,ht_8,wt_4,ht_4);
	Downsample_mask<<<gdim_8, bdim>>>(mask_4,mask_8,wt_8,ht_8,wt_4,ht_4);
	
	CalculateFixed<<<gdim_d, bdim>>>(
	bkg_8, buf1_8, mask_8, fixed_8,
	wb_8, hb_8, wt_8, ht_8, oy_8, ox_8
	);
	

	// downsample iterate

	for(int i = 0; i < 200; ++i ){
		PoissonImageCloningIteration<<<gdim_8, bdim>>>(
		fixed_8, mask_8, buf1_8, buf2_8, wt_8, ht_8
		);
		PoissonImageCloningIteration<<<gdim_4, bdim>>>(
		fixed_8, mask_8, buf2_8, buf1_8, wt_8, ht_8
		);
	}
	
	Upsample<<<gdim_8, bdim>>>(buf1_8,buf1_4,wt_8,ht_8,wt_4,ht_4);

	for(int i = 0; i < 20; ++i ){
		PoissonImageCloningIteration<<<gdim_4, bdim>>>(
		fixed_4, mask_4, buf1_4, buf2_4, wt_4, ht_4
		);
		PoissonImageCloningIteration<<<gdim_4, bdim>>>(
		fixed_4, mask_4, buf2_4, buf1_4, wt_4, ht_4
		);
	}
	
	Upsample<<<gdim_4, bdim>>>(buf1_4,buf1_d,wt_4,ht_4,wt_d,ht_d);

	
	for(int i = 0; i < 20; ++i ){
		PoissonImageCloningIteration<<<gdim_d, bdim>>>(
		fixed_d, mask_d, buf1_d, buf2_d, wt_d, ht_d
		);
		PoissonImageCloningIteration<<<gdim_d, bdim>>>(
		fixed_d, mask_d, buf2_d, buf1_d, wt_d, ht_d
		);
	}
	
	Upsample<<<gdim_d, bdim>>>(buf1_d,buf1,wt_d,ht_d,wt,ht);
	//merge<<<gdim, bdim>>>(target,buf1,wt,ht);
	//Upsample<<<gdim_bd, bdim>>>(bkg_d,bkg_u,wb_d,hb_d,wb,hb);

	
	
	// iterate
	
	for (int i = 0; i < 20; ++i) {
		PoissonImageCloningIteration<<<gdim, bdim>>>(
		fixed, mask, buf1, buf2, wt, ht
		);
		PoissonImageCloningIteration<<<gdim, bdim>>>(
		fixed, mask, buf2, buf1, wt, ht
		);
	}
	
	

	
	// copy the image back
	//Downsample<<<gdim_d, bdim>>>(fixed,buf1_d,wt_d,ht_d,wt,ht);
	//Upsample<<<gdim_d, bdim>>>(buf1_d,buf1,wt_d,ht_d,wt,ht);
	cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	
	SimpleClone<<<gdim, bdim>>>(
	background, buf1, mask, output,
	wb, hb, wt, ht, oy, ox
	);
	
	
	// clean up
	cudaFree(fixed);
	cudaFree(buf1);
	cudaFree(buf2);
	cudaFree(buf1_d);
	cudaFree(buf2_d);
	cudaFree(fixed_d);
	cudaFree(bkg_d);
	cudaFree(mask_d);
	cudaFree(buf1_4);
	cudaFree(buf2_4);
	cudaFree(fixed_4);
	cudaFree(bkg_4);
	cudaFree(mask_4);
	cudaFree(buf1_8);
	cudaFree(buf2_8);
	cudaFree(fixed_8);
	cudaFree(bkg_8);
	cudaFree(mask_8);
	//cudaFree(bkg_u);
}
