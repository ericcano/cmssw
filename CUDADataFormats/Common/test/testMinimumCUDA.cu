#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

int main(void) {
  int devices = 0;
  cudaError_t st = cudaGetDeviceCount(&devices);
  std::cout << "st= " << cudaGetErrorString(st) << " count=" << devices << std::endl;  
  return 0;
}