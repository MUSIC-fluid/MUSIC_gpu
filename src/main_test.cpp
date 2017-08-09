// Original copyright 2011 @ Bjoern Schenke, Sangyong Jeon, and Charles Gale
// Massively cleaned up and improved by Chun Shen 2015-2016

#include <stdio.h>
#include <sys/stat.h>
#include <iostream>
#include <unistd.h>

#define GRID_SIZE_X 200
#define GRID_SIZE_Y 200
#define GRID_SIZE_ETA 64

struct Field {
     double *e_rk0;
     double *e_rk1;
     double *e_prev;
     double *rhob_rk0;
     double *rhob_rk1;
     double *rhob_prev;
     double **u_rk0;
     double **u_rk1;
     double **u_prev;
     double **dUsup;
     double **Wmunu_rk0;
     double **Wmunu_rk1;
     double **Wmunu_prev;
     double *pi_b_rk0;
     double *pi_b_rk1;
     double *pi_b_prev;
};

using namespace std;

// main program
int main(int argc, char *argv[]) {
    Field *hydro_fields;
    int n_cell = GRID_SIZE_ETA*(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1);
    hydro_fields->e_rk0 = new double[n_cell];
    hydro_fields->e_rk1 = new double[n_cell];
    hydro_fields->e_prev = new double[n_cell];
    hydro_fields->rhob_rk0 = new double[n_cell];
    hydro_fields->rhob_rk1 = new double[n_cell];
    hydro_fields->rhob_prev = new double[n_cell];
    hydro_fields->u_rk0 = new double* [n_cell];
    hydro_fields->u_rk1 = new double* [n_cell];
    hydro_fields->u_prev = new double* [n_cell];
    hydro_fields->dUsup = new double* [n_cell];
    hydro_fields->Wmunu_rk0 = new double* [n_cell];
    hydro_fields->Wmunu_rk1 = new double* [n_cell];
    hydro_fields->Wmunu_prev = new double* [n_cell];
    for (int i = 0; i < n_cell; i++) {
        hydro_fields->e_rk0[i] = drand48();
        hydro_fields->e_rk1[i] = drand48();
        hydro_fields->e_prev[i] = drand48();
        hydro_fields->rhob_rk0[i] = drand48();
        hydro_fields->rhob_rk1[i] = drand48();
        hydro_fields->rhob_prev[i] = drand48();

        hydro_fields->u_rk0[i] = new double[4];
        hydro_fields->u_rk1[i] = new double[4];
        hydro_fields->u_prev[i] = new double[4];
        for (int j = 0; j < 4; j++) {
            hydro_fields->u_rk0[i][j] = drand48();
            hydro_fields->u_rk1[i][j] = drand48();
            hydro_fields->u_prev[i][j] = drand48();
        }
        hydro_fields->dUsup[i] = new double[20];
        for (int j = 0; j < 20; j++) {
            hydro_fields->dUsup[i][j] = drand48();
        }
        hydro_fields->Wmunu_rk0[i] = new double[14];
        hydro_fields->Wmunu_rk1[i] = new double[14];
        hydro_fields->Wmunu_prev[i] = new double[14];
        for (int j = 0; j < 14; j++) {
            hydro_fields->Wmunu_rk0[i][j] = drand48();
            hydro_fields->Wmunu_rk1[i][j] = drand48();
            hydro_fields->Wmunu_prev[i][j] = drand48();
        }
    }
    cout << "pre data copy" << endl;
    #pragma acc data copyin (hydro_fields[0:1],\
                         hydro_fields->e_rk0[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA],\
                         hydro_fields->e_prev[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA],\
                         hydro_fields->rhob_rk0[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA],\
                         hydro_fields->e_rk1[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA],\
                         hydro_fields->rhob_rk1[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA],\
                         hydro_fields->rhob_prev[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA], \
                         hydro_fields->u_rk0[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA][0:4], \
                         hydro_fields->u_rk1[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA][0:4], \
                         hydro_fields->u_prev[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA][0:4], \
                         hydro_fields->dUsup[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA][0:20], \
                         hydro_fields->Wmunu_rk0[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA][0:14], \
                         hydro_fields->Wmunu_rk1[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA][0:14], \
                         hydro_fields->Wmunu_prev[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA][0:14], \
                         hydro_fields->pi_b_rk0[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA], \
                         hydro_fields->pi_b_rk1[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA], \
                         hydro_fields->pi_b_prev[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA])
    {
        sleep(100000);
    }
    cout << "post data copy" << endl;
    return(0);
}  /* main */

