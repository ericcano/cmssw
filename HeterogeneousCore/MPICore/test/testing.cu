#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <vector>
#include <random>
#include <utility>
#include <mpi.h>
#include <unistd.h>
/////////////////CUDA/////////////
#include <cuda.h>

struct MPIData 
{
    int num_procs{0};
    int rank{0};
  
    std::pair<int, int> workSplit;
    std::vector<float> input1;          //declare vector 1.
    std::vector<float> input2;          //declare vector 2.
    std::vector<float> output;          //declare vector fulled only by root to get result from workers.
    std::vector<float> reference;       //declare vector to verify the ruslt form each process.
    std::vector<float> vectorWorkers1;  //declare vector 1 for workers only.
    std::vector<float> vectorWorkers2;  //declare vector 2 for workers only.
    std::vector<int> displacement;      //declare vector for selecting location of each element to be sent.
    std::vector<int> numberToSend;
  };

int root = 0; 
int precision = 4;           //default digits after decimal point.
int sizeVector = 5; //Vector size for adding.

const std::pair<float, float> nonBlockSend(MPIData& mpiInput);
void randomGenerator(std::vector<float>& vect);
const std::vector<int> numberDataSend(int numberOfProcess, std::pair<int, int> splitWorks);
std::pair<int, int> splitProcess(int works, int numberOfProcess);
const std::vector<int> displacmentData(int numberOfProcess,
                                       std::pair<int, int> splitWorks,
                                       const std::vector<int>& numberDataSend);
void checkingResultsPrintout(std::vector<float>& reference,
                             std::vector<float>& output,
                             std::pair<int, int> workSplit,
                             const std::vector<int>& displacement,
                             const std::vector<int>& numberDataSend);





//////////////////////////////////////////// C U D A  /////////////////////////////////////////
//called in the Host and excuted in the Device (GPU)
__global__ void addVectorsGpu(float *vect1, float *vect2, float *vect3, int size) //add two vectors and save the result into the third vector.
{
  //blockDim.x gives the number of threads in a block, in the x direction.
  //gridDim.x gives the number of blocks in a grid, in the x direction.
  //blockDim.x * gridDim.x gives the number of threads in a grid (in the x direction, in this case).
  int first = blockDim.x*blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = first; i < size; i+= stride)
  {
    vect3[i] = vect2[i] + vect1[i];
  }
}


int main (int argc, char* argv[])
{   
    
    MPIData mpiInputs;  //greate object from structur to pass into MPI functios.
    
  MPI_Init(&argc, &argv);                            //initialize communicator environment.
  mpiInputs.num_procs = MPI::COMM_WORLD.Get_size();  //get total size of processes.
  mpiInputs.rank = MPI::COMM_WORLD.Get_rank();       //get each process number.

  mpiInputs.input1.resize(sizeVector);  //initialize size.
  mpiInputs.input2.resize(sizeVector);
  mpiInputs.output.resize(sizeVector);
  mpiInputs.reference.resize(sizeVector);

  mpiInputs.workSplit = splitProcess(sizeVector, mpiInputs.num_procs);

  if (!mpiInputs.workSplit.first) {
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    return 0;
  }

  mpiInputs.numberToSend = numberDataSend(mpiInputs.num_procs, mpiInputs.workSplit);
  mpiInputs.displacement = displacmentData(mpiInputs.num_procs, mpiInputs.workSplit, mpiInputs.numberToSend);

  mpiInputs.vectorWorkers1.resize(
      mpiInputs.numberToSend[mpiInputs.rank]);  //Resizing each process with appropriate Receiving Data.
  mpiInputs.vectorWorkers2.resize(mpiInputs.numberToSend[mpiInputs.rank]);

  if (!mpiInputs.rank)  //Only for root
  {
    randomGenerator(mpiInputs.input1);  //generate random floating numbers from(0,1) Only in the root.
    randomGenerator(mpiInputs.input2);
    std::cout << "\n\tNumber of Processes " << mpiInputs.num_procs << std::endl;
    std::cout << "\tNumber of workSplit First " << mpiInputs.workSplit.first << std::endl;
    std::cout << "\tNumber of workSplit Second " << mpiInputs.workSplit.second << std::endl;
    for (int j = 0; j < sizeVector; j++) {
      mpiInputs.reference[j] = mpiInputs.input1[j] + mpiInputs.input2[j];  //Summing for verification.
    }
  }

  std::pair<float, float> results;
  results = nonBlockSend(mpiInputs);



  MPI::Finalize();


    return 0;
}
void randomGenerator(std::vector<float>& vect) 
{
    std::random_device rand;
    std::default_random_engine gener(rand());
    std::uniform_real_distribution<> dis(0., 1.);
    int size = vect.size();
    for (int i = 0; i < size; i++) {
      vect.at(i) = dis(gener);
    }
}
std::pair<int, int> splitProcess(int works, int numberOfProcess) {
  std::pair<int, int> Return{0, 0};
  if (numberOfProcess > 1 && numberOfProcess <= works) {
    Return.first = works / (numberOfProcess - 1);   //number of cycle for each process.
    Return.second = works % (numberOfProcess - 1);  //extra cycle for process.
  } else {
    std::cout << "\tError Either No worker are found OR Number Processes Larger than Length!!!\n";
  }

  return Return;
}
const std::vector<int> numberDataSend(int numberOfProcess, std::pair<int, int> splitWorks) {
  std::vector<int> dataSend(numberOfProcess, splitWorks.first);
  dataSend[0] = 0;
  for (int i = 1; i < splitWorks.second + 1; i++)  //neglect root
  {
    dataSend[i] += 1;  //extra work for each first processes.
  }
  return dataSend;
}
const std::vector<int> displacmentData(int numberOfProcess,
                                       std::pair<int, int> splitWorks,
                                       const std::vector<int>& numberDataSend) {
  std::vector<int> displacment(numberOfProcess, splitWorks.first);

  displacment[0] = 0;
  displacment[1] = 0;  //start Here.

  for (int i = 2; i < numberOfProcess; i++)  //neglect root
  {
    displacment[i] = numberDataSend[i - 1] + displacment[i - 1];  //extra work for each first processes.
  }
  return displacment;
}
void checkingResultsPrintout(std::vector<float>& reference,
                             std::vector<float>& output,
                             std::pair<int, int> workSplit,
                             const std::vector<int>& displacement,
                             const std::vector<int>& numberDataSend) {
  float percent{0.0};
  float totalError{0.0};
  int p{1};
  for (int j = 0; j < sizeVector; j++) {
    percent = ((reference[j] - output[j]) / reference[j]) * 100;
    totalError += percent;
  }
  
    std::cout << "\n-------------------------------------------------------\n";
    std::cout << "| RootSum | WorksSum | Error   | Error %  | Process # |";
    std::cout << "\n-------------------------------------------------------\n";
    std::cout.precision(precision);
    for (int j = 0; j < sizeVector; j++) {
      std::cout << "| " << reference[j] << "  | " << output[j] << "  |" << std::setw(9) << reference[j] - output[j]
                << " |" << std::setw(9) << percent << " |" << std::setw(9) << p << " |\n";

      if (j + 1 == displacement[p + 1]) {
        ++p;
      }
    }
    std::cout << "-------------------------------------------------------\n";
    std::cout << "-Total Error is " << totalError << std::endl;
    for (long unsigned int j = 1; j < displacement.size(); j++) {
      std::cout << "Process [" << j << "]"
                << " Worked On " << numberDataSend[j] << " Data\n";
    }
  
}
const std::pair<float, float> nonBlockSend(MPIData& mpiInput) 
{
    std::pair<float, float> returnValue;
    double startTimeRootSend = 0;
    double endTimeRootSend = 0;
    double startTimeRootRecv = 0;
    double endTimeRootRecv = 0;
  
    MPI_Request requestRootSend[2];
    MPI_Request requestRootRecv;
    MPI_Request requestWorkerSend;
    MPI_Request requestWorkerRecv[1];
  
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
  
    if (!mpiInput.rank)  //Only for root
    {
      // std::cout << "\n\t\tNon-Blocking Send and Receive " << std::endl;
      startTimeRootSend = MPI_Wtime();
      for (int i = 1; i < mpiInput.num_procs; i++) {
        MPI_Issend(&mpiInput.input1[mpiInput.displacement[i]],
                   mpiInput.numberToSend[i],
                   MPI_FLOAT,
                   i,
                   0,
                   MPI_COMM_WORLD,
                   &requestRootSend[0]);  //Tag is 0
        MPI_Issend(&mpiInput.input2[mpiInput.displacement[i]],
                   mpiInput.numberToSend[i],
                   MPI_FLOAT,
                   i,
                   0,
                   MPI_COMM_WORLD,
                   &requestRootSend[1]);
        MPI_Waitall(2, requestRootSend, MPI_STATUS_IGNORE);
      }
      endTimeRootSend = MPI_Wtime();
    }
  
    if (mpiInput.rank)  //Only for Workers
    {
      MPI_Irecv(&mpiInput.vectorWorkers1[0],
                mpiInput.numberToSend[mpiInput.rank],
                MPI_FLOAT,
                root,
                0,
                MPI_COMM_WORLD,
                &requestWorkerRecv[0]);
      MPI_Irecv(&mpiInput.vectorWorkers2[0],
                mpiInput.numberToSend[mpiInput.rank],
                MPI_FLOAT,
                root,
                0,
                MPI_COMM_WORLD,
                &requestWorkerRecv[1]);
  
      MPI_Waitall(2, requestWorkerRecv, MPI_STATUS_IGNORE);
      
      std::cout << "vectorWorkers1 : \n";
      for (long unsigned int i = 0; i < mpiInput.vectorWorkers1.size(); i++) {
        std::cout <<  mpiInput.vectorWorkers1[i] << std::endl;
      }
      std::cout << "vectorWorkers2 : \n";
      for (long unsigned int i = 0; i < mpiInput.vectorWorkers2.size(); i++) {
        std::cout <<  mpiInput.vectorWorkers2[i] << std::endl;
      }
      
      ///////////////////////////////////// C U D A //////////////////////////////////
      float *d_vect1, *d_vect2, *d_vect3Gpu; //create pointers for Device.

      cudaMalloc((void**)&d_vect1, mpiInput.vectorWorkers1.size()*sizeof(float));//allocate memory space for vector in the global memory of the Device.
      cudaMalloc((void**)&d_vect2, mpiInput.vectorWorkers2.size()*sizeof(float));
      cudaMalloc((void**)&d_vect3Gpu, mpiInput.vectorWorkers1.size()*sizeof(float));

      cudaMemcpy(d_vect1, mpiInput.vectorWorkers1.data(), mpiInput.vectorWorkers1.size()*sizeof(float), cudaMemcpyHostToDevice);//copy random vector from host to device.
      cudaMemcpy(d_vect2, mpiInput.vectorWorkers2.data(), mpiInput.vectorWorkers1.size()*sizeof(float), cudaMemcpyHostToDevice);

      int threads = 512; //arbitrary number.
      int blocks = (sizeVector + threads - 1) / threads; //get ceiling number of blocks.
      blocks = std::min(blocks, 8); // Number 8 is least number can be got from lowest Nevedia GPUs.
      
      addVectorsGpu<<<blocks,threads>>> (d_vect1, d_vect2, d_vect3Gpu, sizeVector); //call device function to add two vectors and save into vect3Gpu.
      
      	
      cudaError_t checkRecieving;
      checkRecieving = cudaMemcpy( mpiInput.vectorWorkers1.data(),d_vect3Gpu, mpiInput.vectorWorkers1.size()*sizeof(float), cudaMemcpyDeviceToHost);
      if ( checkRecieving != cudaSuccess )
     {
        printf("CUDA Error: %s\n", cudaGetErrorString(checkRecieving));       

        
     }
      ///////////////////////////////////// C U D A //////////////////////////////////
      
      MPI_Issend(&mpiInput.vectorWorkers1[0],
                 mpiInput.numberToSend[mpiInput.rank],
                 MPI_FLOAT,
                 root,
                 0,
                 MPI_COMM_WORLD,
                 &requestWorkerSend);  //Tag is 0
      MPI_Wait(&requestWorkerSend, MPI_STATUS_IGNORE);
    }
  
    if (!mpiInput.rank)  //Only for root
    {
      startTimeRootRecv = MPI_Wtime();
      for (int i = 1; i < mpiInput.num_procs; i++) {
        MPI_Irecv(&mpiInput.output[mpiInput.displacement[i]],
                  mpiInput.numberToSend[i],
                  MPI_FLOAT,
                  i,
                  0,
                  MPI_COMM_WORLD,
                  &requestRootRecv);
        MPI_Wait(&requestRootRecv, MPI_STATUS_IGNORE);
      }
      endTimeRootRecv = MPI_Wtime();
  
      checkingResultsPrintout(mpiInput.reference,
                              mpiInput.output,
                              mpiInput.workSplit,
                              mpiInput.displacement,
                              mpiInput.numberToSend);  //Only root print out the results.
      returnValue.first = (endTimeRootSend - startTimeRootSend) * 1000;
      returnValue.second = (endTimeRootRecv - startTimeRootRecv) * 1000;
    }
    return returnValue;
  }


