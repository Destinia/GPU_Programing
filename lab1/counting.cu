#include "counting.h"
#include <fstream>
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include <thrust/copy.h>

using namespace std;
__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

struct is_one
  {
    __host__ __device__
    bool operator()(const int x)
    {
      return (x != 1);
    }
  };

 struct to_sym
 {
 	__host__ __device__
 	int operator()(const int x)
 	{
 		return -1;
 	}
 };
 struct is_not_sym
 {
 	__host__ __device__
 	bool operator()(const int x)
 	{
 		return (x != -1);
 	}
 };
 

__global__ void check_all(const char *text, int *pos, int text_size, int *segtree,int leaf_start){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(text[idx]=='\n'||idx>=text_size){
		segtree[idx+leaf_start] = 0;
	}
	else{
		segtree[idx+leaf_start] = 1;
	}

}

__global__ void buildTree(int *segtree,int leaf_start){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(segtree[(idx+leaf_start)*2+1] != 0 && segtree[(idx+leaf_start)*2+2] != 0){
		segtree[idx+leaf_start] = segtree[(idx+leaf_start)*2+1] + segtree[(idx+leaf_start)*2+2];
	}
	else{
		segtree[idx+leaf_start] = 0;
	}
}

__global__ void findlength(int *segtree,int *pos,int leaf_start,int leaf_end,int text_size){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx>=text_size)return;
	int pivot = idx + leaf_start;
	int next;
	pos[idx] = 0;
	//bottom-up
	while(pivot != 0){
		//right child
		if(pivot%2 == 0){
			if(segtree[pivot]==0){
				break;
			}
			else{
				next = (pivot-1)/2;
				if(segtree[next]==0){
					pos[idx] += segtree[pivot];
					next = next*2+1;
				}
			}
		}
		//left child
		else{
			if(segtree[pivot]==0){
				break;
			}
			else{
				next = (pivot-1)/2 -1;
				pos[idx] += segtree[pivot];
			}

		}
	pivot = next;
	}
	//top-down
	while(pivot<=leaf_end){
		if(segtree[pivot] == 0){
			//find right child
			next = (pivot+1)*2;
		}
		else{
			pos[idx] += segtree[pivot];
			next = pivot*2;
		}
		pivot = next;
	}
}

__global__ void switch_char(char *ref,int *pos, char *out, int text_size){
	int idx = (blockIdx.x * blockDim.x + threadIdx.x)*2;
	if(pos[idx]==0||idx>=text_size){
		return;
	}
	if(pos[idx] % 2== 1){
		if(pos[idx+1]!=0){
			out[idx] = ref[idx+1];
			out[idx+1] = ref[idx] ;
		}
	}
	else {
		out[idx-1] = ref[idx];
		out[idx] = ref[idx-1];
	}
}

void CountPosition(const char *text, int *pos, int text_size)
{	
	int level = 0;
	int temp = text_size-1;
	int leaf_start=0;
	while(1){
		temp/=2;
		level++;
		if(temp==0)break;
	}
	int pivot = 1;
	for(int i=0;i<level;i++){
		leaf_start +=pivot;
		pivot*=2;
	}
	int length = leaf_start + pivot;
	int start  = leaf_start;
	//printf("\nlength %d\nlevel %d\npivot %d\nstart %d\n",length ,level,pivot,start);
	//create segtree memory
	int *seg_tree;
	cudaMalloc((void**)&seg_tree,length*sizeof(int));

	
	check_all<<<(pivot>>6),64>>>(text,pos,text_size,seg_tree,start);

	cudaDeviceSynchronize();
	for(int i=0;i<level;i++){
		pivot/=2;
		leaf_start -=pivot;
		if(pivot>=64)
			buildTree<<<(pivot>>6),64>>>(seg_tree,leaf_start);
		else
			buildTree<<<1,pivot>>>(seg_tree,leaf_start);
	}
	//for debug
	/*
	int check = 1;
	int count = 0;
	int *check_tree = new int[length];
	cudaMemcpy(check_tree,seg_tree,length*sizeof(int),cudaMemcpyDeviceToHost);
	fstream fp;
	fp.open("tree.txt",ios::out);
	for(int i=0;i<length;i++){
		if(count == check ){check*=2;count=0;fp<<endl;}
		fp<<check_tree[i]<<" ";
		count++;
	}
	fp.close();
	delete[] check_tree;
	*/
	////
	
	cudaDeviceSynchronize();
	
	findlength<<<(text_size/64+1),64>>>(seg_tree,pos,start,length,text_size);

	cudaFree(seg_tree);
}

int ExtractHead(const int *pos, int *head, int text_size)
{
	int *buffer;
	int nhead;
	cudaMalloc(&buffer, sizeof(int)*text_size*2); // this is enough
	thrust::device_ptr<const int> pos_d(pos);
	thrust::device_ptr<int> head_d(head), flag_d(buffer), cumsum_d(buffer+text_size);
	thrust::device_ptr<int> temp;
	thrust::sequence(flag_d,flag_d+text_size);
 

  	thrust::transform_if(flag_d, flag_d + text_size, pos_d, flag_d, to_sym(), is_one());
  	temp = thrust::copy_if(flag_d,flag_d+text_size,head_d,is_not_sym());
  	nhead = temp - head_d ;


	// TODO

	cudaFree(buffer);
	return nhead;
}

void Part3(char *text, int *pos, int *head, int text_size, int n_head)
{
	char *temp;
	cudaMalloc(&temp,sizeof(char)*text_size);
	cudaMemcpy(temp,text,text_size*sizeof(char),cudaMemcpyDeviceToDevice);
	switch_char<<<text_size/64+1,64>>>(temp,pos,text,text_size);

	//check
	/*char *check = new char[text_size];
	cudaMemcpy(check,text,text_size*sizeof(char),cudaMemcpyDeviceToHost);
	fstream fp;
	fp.open("switch.txt",ios::out);
	for(int i=0;i<text_size;i++){
		fp<<check[i]<<' ';
	}
	fp.close();*/

	cudaFree(temp);
}
