#include <stdio.h>
#include <mpi.h>

void probing_process(int *int_sum, float *float_sum) {
    MPI_Status status;

    // 1- Probe the incoming message
    MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

    // 2- Get the tag and the source
    int tag = status.MPI_TAG;
    int source = status.MPI_SOURCE;

    // Printing the message
    printf("Received a message from process %d with tag %d\n", source, tag);

    // 3- Add to int_sum or float_sum depending on the tag of the message
    if (tag == 0) {
        int message;
        MPI_Recv(&message, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
        *int_sum += message;
    } else if (tag == 1) {
        float message;
        MPI_Recv(&message, 1, MPI_FLOAT, source, tag, MPI_COMM_WORLD, &status);
        *float_sum += message;
    }
}
