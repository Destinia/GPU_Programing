#include <cstdio>
#include <cstdlib>
#include "SyncedMemory.h"

#define CHECK {\
	auto e = cudaDeviceSynchronize();\
	if (e != cudaSuccess) {\
		printf("At " __FILE__ ":%d, %s\n", __LINE__, cudaGetErrorString(e));\
		abort();\
	}\
}

__global__ void SomeTransform(char *input_gpu, int fsize) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx < fsize && input_gpu[idx] != '\n') {
		//input_gpu[idx] = '!';
		//transform to capital
		if(input_gpu[idx] >= 'a' && input_gpu[idx] <= 'z'){
			input_gpu[idx] -= 32;
		}
		//swap all pairs in all words	
		if(input_gpu[idx] == ' ' || idx==0){
			char* cur;
			if(idx==0)
				cur = input_gpu;
			else
				cur = input_gpu+idx+1;
			while((*cur>'A'&&*cur<'Z')||(*cur>'a'&&*cur<'z')){
				if((*(cur+1)>'A'&&*(cur+1)<'Z')||(*(cur+1)>'a'&&*(cur+1)<'z')){
					char temp = *cur;
					*cur = *(cur+1);
					*(cur+1) = temp;
					cur+=2;
				}
				else
					break;
			}
		}
	}	
}

int main(int argc, char **argv)
{
	// init, and check
	if (argc != 2) {
		printf("Usage %s <input text file>\n", argv[0]);
		abort();
	}
	FILE *fp = fopen(argv[1], "r");
	if (!fp) {
		printf("Cannot open %s", argv[1]);
		abort();
	}
	// get file size
	fseek(fp, 0, SEEK_END);
	size_t fsize = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	// read files
	MemoryBuffer<char> text(fsize+1);
	auto text_smem = text.CreateSync(fsize);
	CHECK;
	fread(text_smem.get_cpu_wo(), 1, fsize, fp);
	text_smem.get_cpu_wo()[fsize] = '\0';
	fclose(fp);

	// TODO: do your transform here
	char *input_gpu = text_smem.get_gpu_rw();
	// An example: transform the first 64 characters to '!'
	// Don't transform over the tail
	// And don't transform the line breaks
	SomeTransform<<<fsize/64+1, 64>>>(input_gpu, fsize);

	puts(text_smem.get_cpu_ro());
	return 0;
}
