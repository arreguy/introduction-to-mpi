#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

// Global variables to store the rank of the process and the size
// of the communicator
int rank, size;

// Number of points on one side. The total number of points
// will be p_count*p_count.
#define p_count 512

// Other global variables. We read them from the command line
// this will be handled by the script running on tech.io, don't
// mind this part.
// The cutoff variable indicates when we decide the series does not converge
// The other variable are just used to center the view and zoom level.
int cutoff;
double min_x, max_x, min_y, max_y, dx, dy;

// The modulus of a complex number
double modulus(double x, double y) {
  return sqrt(x * x + y * y);
}

// Multiplying a complex number by itself
void self_mul(double *x, double *y) {
  double ox = (*x) * (*x) - (*y) * (*y);
  double oy = (*x) * (*y) + (*y) * (*x);
  *x = ox;
  *y = oy;
}

// Computation of the number of iterations on a set of points
// The result is stored in mset.
void compute_mandelbrot(double *points, int npts, int *mset) {
  // For each point
  for (int i = 0; i < npts; ++i) {
    double px, py;
    px = points[i * 2];
    py = points[i * 2 + 1];

    int iteration = 0;
    double zx = 0;
    double zy = 0;

    // We iterate until cutoff or modulus > 2
    while (iteration < cutoff) {
      self_mul(&zx, &zy);
      zx += px;
      zy += py;
      double mod = modulus(zx, zy);

      if (mod > 2.0f)
        break;

      iteration++;
    }

    // We store the number of iterations, and we use
    // a special value (-1) if we don't converge
    if (iteration == cutoff)
      mset[i] = -1;
    else
      mset[i] = iteration;
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  // Getting rank and size
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Reading the parameters on the command line
  min_x = strtod(argv[1], NULL);
  max_x = strtod(argv[2], NULL);
  min_y = strtod(argv[3], NULL);
  max_y = strtod(argv[4], NULL);
  dx = max_x - min_x;
  dy = max_y - min_y;
  cutoff = atoi(argv[5]);

  // Initialisation of the points :
  // The process with rank 0 will hold all the points
  // The others will keep the variable points as a null pointer
  MPI_Barrier(MPI_COMM_WORLD);
  double *points = NULL;

  if (rank == 0) {
    points = (double *)malloc(p_count * p_count * 2 * sizeof(double));
    for (int yp = 0; yp < p_count; ++yp) {
      double py = min_y + dy * yp / p_count;
      for (int xp = 0; xp < p_count; ++xp) {
        double px = min_x + dx * xp / p_count;

        int lid = yp * p_count * 2 + xp * 2;
        points[lid] = px;
        points[lid + 1] = py;
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // The number of points we hold
  int npts_per_process = p_count * p_count / size;
  double *local_points = (double *)malloc(npts_per_process * 2 * sizeof(double));

  MPI_Scatter(points, npts_per_process * 2, MPI_DOUBLE,
              local_points, npts_per_process * 2, MPI_DOUBLE,
              0, MPI_COMM_WORLD);

  // Computing the mandelbrot set.
  // This function is already coded, and you don't have to worry about it
  int *mset = (int *)malloc(npts_per_process * sizeof(int));
  compute_mandelbrot(local_points, npts_per_process, mset);

  int *gathered_mset = NULL;
  if (rank == 0) {
    gathered_mset = (int *)malloc(p_count * p_count * sizeof(int));
  }

  MPI_Gather(mset, npts_per_process, MPI_INT,
             gathered_mset, npts_per_process, MPI_INT,
             0, MPI_COMM_WORLD);

  // Printing only one result that will be used to create the image
  if (rank == 0) {
    for (int yp = 0; yp < p_count; ++yp) {
      for (int xp = 0; xp < p_count; ++xp)
        printf("%d ", gathered_mset[yp * p_count + xp]);
      printf("\n");
    }
  }

  // Cleaning up the mess and exiting properly
  free(points);
  free(local_points);
  free(mset);
  free(gathered_mset);

  MPI_Finalize();
  return 0;
}
