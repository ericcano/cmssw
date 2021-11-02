#include <mpi.h>
#include <iostream>
#include <unistd.h>
#include <limits.h>

int main(int argc, char** argv) {
  // Initialize the MPI environment
  MPI_Init(NULL, NULL);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Get the name of the processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  // Print off a hello world message
  char hostname[HOST_NAME_MAX + 1];
  auto rc = gethostname(hostname, sizeof(hostname));
  if (rc) {
    perror("Failed to get hostname: ");
    hostname[0] = '\0';
  }
  printf("Hello world from processor %s, rank %d out of %d processors on host %s\n",
         processor_name, world_rank, world_size, hostname);

  // Finalize the MPI environment.
  MPI_Finalize();
}