// Copyright 2011 @ Bjoern Schenke, Sangyong Jeon, and Charles Gale
#ifndef SRC_ADVANCE_H_
#define SRC_ADVANCE_H_

#include <iostream>
#include "./data.h"
#include "./grid.h"
#include "./dissipative.h"
#include "./minmod.h"
#include "./u_derivative.h"
#include "./reconst.h"

//! advance routines separate for
//! T^{0 nu} \del T^{i\nu} (T)
//! W
//! T^{0 nu} with W source (TS)
//! W with source (WS)
class Advance {
 private:
    InitData* DATA_ptr;
    Util *util;
    Diss *diss;
    Reconst *reconst_ptr;
    EOS *eos;
    Minmod *minmod;
    U_derivative *u_derivative_ptr;
    
    int grid_nx, grid_ny, grid_neta;
    int rk_order;

 public:
    Advance(EOS *eosIn, InitData* DATA_in);
    ~Advance();

    int AdvanceIt(double tau_init, InitData *DATA, Grid ***arena, int rk_flag);

    void prepare_qi_array(
        double tau, Grid ***arena, int rk_flag, int ieta, int ix, int iy,
        int n_cell_eta, int n_cell_x, double **qi_array,
        double **qi_nbr_x, double **qi_nbr_y, double **qi_nbr_eta,
        double **qi_rk0, double **grid_array);

    void prepare_vis_array(
        Grid ***arena, int rk_flag, int ieta, int ix, int iy,
        int n_cell_eta, int n_cell_x, double **vis_array, double **vis_nbr_tau,
        double **vis_nbr_x, double **vis_nbr_y, double **vis_nbr_eta);

    void prepare_velocity_array(double tau_rk, Grid ***arena,
                                int ieta, int ix, int iy, int rk_flag,
                                int n_cell_eta, int n_cell_x,
                                double **velocity_array,
                                double **grid_array);

    int FirstRKStepT(double tau, Grid *grid_pt, int rk_flag,
                     double **qi_array, double **qi_nbr_x, double **qi_nbr_y,
                     double **qi_nbr_eta, int n_cell_eta, int n_cell_x,
                     double **vis_array, double **vis_nbr_tau,
                     double **vis_nbr_x, double **vis_nbr_y,
                     double **vis_nbr_eta,
                     double **qi_rk0, double **grid_array);

    int FirstRKStepW(double tau_it, InitData *DATA, Grid *grid_pt,
                     int rk_flag, double theta_local, double* a_local,
                     double *sigma_local, double **vis_array,
                     double **vis_nbr_tau, double **vis_nbr_x,
                     double **vis_nbr_y, double **vis_nbr_eta,
                     double **velocity_array, double **grid_array,
                     double **vis_array_new);

    
    void update_grid_array_from_grid_cell(Grid *grid_p, double *grid_array,
                                          int rk_flag);
    void update_grid_cell_from_grid_array(Grid *grid_p, double *grid_array);

    void update_vis_array_from_grid_cell(Grid *grid_p, double *vis_array,
                                         int rk_flag);
    void update_vis_prev_tau_from_grid_cell(Grid *grid_p, double *vis_array,
                                            int rk_flag);

    void UpdateTJbRK(double *grid_array, Grid *grid_pt, int rk_flag);

    void update_grid_cell(double **grid_array, Grid ***arena, int rk_flag,
                          int ieta, int ix, int iy,
                          int n_cell_eta, int n_cell_x);

    int QuestRevert(double tau, Grid *grid_pt, int rk_flag,
                    double **vis_array, double **grid_array);
    int QuestRevert_qmu(double tau, Grid *grid_pt, int rk_flag,
                        double **vis_array, double **grid_array);

    void MakeDeltaQI(double tau, double **qi_array, double **qi_nbr_x,
                     double **qi_nbr_y, double **qi_nbr_eta,
                     int n_cell_eta, int n_cell_x, double **grid_array);

    double MaxSpeed(double tau, int direc, double *grid_array);

    double get_TJb(Grid *grid_p, int rk_flag, int mu, int nu);
    double get_TJb_new(double *grid_array, int mu, int nu);

    //! This function computes the vector [T^\tau\mu, J^\tau] from the
    //! grid_array [e, v^i, rhob]
    void get_qmu_from_grid_array(double tau, double *qi, double *grid_array);
};

#endif  // SRC_ADVANCE_H_
