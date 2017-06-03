// Copyright 2012 Bjoern Schenke, Sangyong Jeon, and Charles Gale
#ifndef SRC_DISSIPATIVE_H_
#define SRC_DISSIPATIVE_H_

#include <iostream>
#include "./util.h"
#include "./grid.h"
#include "./data.h"
#include "./minmod.h"

class Diss {
 private:
    EOS *eos;
    Minmod *minmod;
    Util *util;
    InitData *DATA_ptr;

 public:
    Diss(EOS *eosIn, InitData* DATA_in);
    ~Diss();
  
    void MakeWSource(double tau, double **qi_array,
                     int n_cell_eta, int n_cell_x, int n_cell_y,
                     double **vis_array,
                     double **vis_nbr_tau, double **vis_nbr_x,
                     double **vis_nbr_y, double **vis_nbr_eta,
                     double **qi_array_new);

    int Make_uWRHS(double tau, int n_cell_eta, int n_cell_x, int n_cell_y,
                   double **vis_array, double **vis_nbr_x,
                   double **vis_nbr_y, double **vis_nbr_eta,
                   double **velocity_array, double **vis_array_new);

    double Make_uWSource(double tau, int n_cell_eta, int n_cell_x,
                         int n_cell_y,
                         double **vis_array, double **velocity_array,
                         double **grid_array, double **vis_array_new);
    
    int Make_uPRHS(double tau, double *p_rhs, double **vis_array,
                   double **vis_nbr_x, double **vis_nbr_y,
                   double **vis_nbr_eta, double **velocity_array);

    double Make_uPiSource(double tau, int n_cell_eta, int n_cell_x,
                          int n_cell_y,
                          double **vis_array, double **velocity_array,
                          double **grid_array, double **vis_array_new);

    int Make_uqRHS(double tau, double **w_rhs, double **vis_array,
                   double **vis_nbr_x, double **vis_nbr_y,
                   double **vis_nbr_eta);

    double Make_uqSource(double tau, int n_cell_eta, int n_cell_x,
                         int n_cell_y,
                         double **vis_array, double **velocity_array,
                         double **grid_array, double **vis_array_new);

    double get_temperature_dependent_eta_s(double T);
    double get_temperature_dependent_zeta_s(double temperature);
};

#endif  // SRC_DISSIPATIVE_H_
