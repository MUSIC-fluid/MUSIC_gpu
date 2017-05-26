// Copyright 2011 @ Bjoern Schenke, Sangyong Jeon, and Charles Gale
#ifndef SRC_RECONST_H_
#define SRC_RECONST_H_

#include <iostream>
#include "./util.h"
#include "./data.h"
#include "./grid.h"
#include "./eos.h"

class Reconst {
 private:
    EOS *eos;
    double eos_eps_max;
    InitData *DATA_ptr;

    int max_iter;
    double rel_err, abs_err;

    double LARGE;

    int echo_level;
    double v_critical;

 public:
    Reconst(EOS *eos, InitData *DATA_in);
    ~Reconst();
      
    int ReconstIt_shell(double *grid_array, double tau, double *uq,
                        double *grid_array_p);

    // reconst_type == 0
    int ReconstIt(double *grid_array, double tau, double *uq,
                  double *grid_array_p);
    double GuessEps(double T00, double K00, double cs2);
    
    void revert_grid(double *grid_array, double *grid_prev);

    // reconst_type == 1
    int ReconstIt_velocity_iteration(double *grid_array, double tau,
                                     double *uq, double *grid_array_p);
    double reconst_velocity_f(double v, double T00, double M, double J0);
    double reconst_u0_f(double u0, double T00, double K00, double M,
                        double J0);

    // reconst_type == 2
    int ReconstIt_velocity_Newton(double *grid_array, double tau,
                                  double *uq, double *grid_array_p);
    double reconst_velocity_f_Newton(double v, double T00, double M,
                                     double J0);
    double reconst_u0_f_Newton(double u0, double T00, double K00,
                               double M, double J0);
    double reconst_velocity_df(double v, double T00, double M, double J0);
    double reconst_u0_df(double u0, double T00, double K00, double M,
                         double J0);

    void regulate_grid(double *grid_array, double elocal);
};

#endif  // SRC_RECONST_H_
