// Copyright 2011 @ Bjoern Schenke, Sangyong Jeon, and Charles Gale
#ifndef SRC_ADVANCE_H_
#define SRC_ADVANCE_H_

#include <iostream>
#include "./data.h"
#include "./grid.h"
#include "./field.h"
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

    int AdvanceIt(double tau_init, InitData *DATA, Field *hydro_fields,
                  int rk_flag);
#pragma acc routine seq
    double get_pressure(double e_local, double rhob);
#pragma acc routine seq
    double get_cs2(double e_local, double rhob);
#pragma acc routine seq
    double p_e_func(double e_local, double rhob);
#pragma acc routine seq
    double p_rho_func(double e_local, double rhob);
#pragma acc routine seq
    double get_temperature(double e_local, double rhob);
#pragma acc routine seq
    double get_mu(double e_local, double rhob);

    //int FirstRKStepT(double tau, int rk_flag,
    //                 double **qi_array, double **qi_nbr_x,
    //                 double **qi_nbr_y, double **qi_nbr_eta,
    //                 int n_cell_eta, int n_cell_x, int n_cell_y,
    //                 double **vis_array, double **vis_nbr_tau,
    //                 double **vis_nbr_x, double **vis_nbr_y,
    //                 double **vis_nbr_eta, double **qi_rk0,
    //                 double **qi_array_new, double **grid_array,
    //                 double *rhs, double *qiphL, double *qiphR,
    //                 double *qimhL, double *qimhR,
    //                 double *grid_array_hL, double *grid_array_hR,
    //                 InitData* DATA);
#pragma acc routine seq
    int FirstRKStepT(double tau, int rk_flag,
                          double qi_array[][5], double qi_nbr_x[][5],
                          double qi_nbr_y[][5], double qi_nbr_eta[][5],
                          int n_cell_eta, int n_cell_x, int n_cell_y,
                          double vis_array[][19], double vis_nbr_tau[][19],
                          double vis_nbr_x[][19], double vis_nbr_y[][19],
                          double vis_nbr_eta[][19], double qi_rk0[][5],
                          double qi_array_new[][5], double grid_array[][5],
                          double *rhs, double *qiphL, double *qiphR,
                          double *qimhL, double *qimhR,
                          double *grid_array_hL, double *grid_array_hR,
                          InitData *DATA);


#pragma acc routine seq
    void MakeDeltaQI(double tau, double qi_array[][5], double qi_nbr_x[][5],
                     double qi_nbr_y[][5], double qi_nbr_eta[][5],
                     int n_cell_eta, int n_cell_x, int n_cell_y,
                     double qi_array_new[][5], double grid_array[][5],
                     double *rhs, double *qiphL, double *qiphR,
                     double *qimhL, double *qimhR,
                     double *grid_array_hL, double *grid_array_hR,
                     InitData* DATA);
    
#pragma acc routine seq
    double MaxSpeed(double tau, int direc, double *grid_array);
    
#pragma acc routine seq
    void revert_grid(double *grid_array, double *grid_prev);
#pragma acc routine seq
    int ReconstIt_velocity_Newton(double *grid_array, double tau, double *uq,
                                  double *grid_array_p);
#pragma acc routine seq
    double reconst_velocity_f(double v, double T00, double M,
                              double J0);
#pragma acc routine seq
    double reconst_velocity_f_Newton(double v, double T00, double M,
                                     double J0);
#pragma acc routine seq
    double reconst_velocity_df(double v, double T00, double M, double J0);
    
#pragma acc routine seq
    double reconst_u0_f(double u0, double T00, double K00, double M,
                        double J0);
#pragma acc routine seq
    double reconst_u0_f_Newton(double u0, double T00, double K00,
                               double M, double J0);
#pragma acc routine seq
    double reconst_u0_df(double u0, double T00, double K00, double M,
                         double J0);
    

#pragma acc routine seq
    double get_TJb_new(double *grid_array, int mu, int nu);

    

    //void prepare_qi_array(
    //    double tau, Field *hydro_fields, int rk_flag, int ieta, int ix, int iy,
    //    int n_cell_eta, int n_cell_x, int n_cell_y, double **qi_array,
    //    double **qi_nbr_x, double **qi_nbr_y, double **qi_nbr_eta,
    //    double **qi_rk0, double **grid_array, double *grid_array_temp);
#pragma acc routine seq
    void prepare_qi_array(
        double tau, Field *hydro_fields, int rk_flag, int ieta, int ix, int iy,
        int n_cell_eta, int n_cell_x, int n_cell_y,
        double qi_array[][5], double qi_nbr_x[][5],
        double qi_nbr_y[][5], double qi_nbr_eta[][5],
        double qi_rk0[][5], double grid_array[][5], double *grid_array_temp);

#pragma acc routine seq
    void prepare_vis_array(
        Field *hydro_fields, int rk_flag, int ieta, int ix, int iy,
        int n_cell_eta, int n_cell_x, int n_cell_y,
        double vis_array[][19], double vis_nbr_tau[][19],
        double vis_nbr_x[][19], double vis_nbr_y[][19],
        double vis_nbr_eta[][19], InitData *DATA);

#pragma acc routine seq
    void prepare_velocity_array(double tau_rk, Field *hydro_fields,
                                     int ieta, int ix, int iy, int rk_flag,
                                     int n_cell_eta, int n_cell_x,
                                     int n_cell_y, double velocity_array[][20], 
                                     double grid_array[][5],
                                     double vis_array_new[][19],
                                     double *grid_array_temp);

#pragma acc routine seq
    int FirstRKStepW(double tau, int rk_flag, int n_cell_eta,
                     int n_cell_x, int n_cell_y, double vis_array[][19],
                     double vis_nbr_tau[][19], double vis_nbr_x[][19],
                     double vis_nbr_y[][19], double vis_nbr_eta[][19],
                     double velocity_array[][20], double grid_array[][5],
                     double vis_array_new[][19]);

#pragma acc routine seq
    void MakeWSource(double tau, double qi_array[][5],
                     int n_cell_eta, int n_cell_x, int n_cell_y,
                     double vis_array[][19],
                     double vis_nbr_tau[][19], double vis_nbr_x[][19],
                     double vis_nbr_y[][19], double vis_nbr_eta[][19],
                     double qi_array_new[][5], InitData* DATA);

#pragma acc routine seq
    int Make_uWRHS(double tau, int n_cell_eta, int n_cell_x, int n_cell_y,
                   double vis_array[][19], double vis_nbr_x[][19],
                   double vis_nbr_y[][19], double vis_nbr_eta[][19],
                   double velocity_array[][20],
                   double vis_array_new[][19]);

#pragma acc routine seq
    double Make_uWSource(double tau, int n_cell_eta, int n_cell_x,
                         int n_cell_y, double vis_array[][19],
                         double velocity_array[][20],
                         double grid_array[][5],
                         double vis_array_new[][19]);

#pragma acc routine seq
    double Make_uPiSource(double tau, int n_cell_eta, int n_cell_x,
                          int n_cell_y, double vis_array[][19],
                          double velocity_array[][20],
                          double grid_array[][5],
                          double vis_array_new[][19]);

#pragma acc routine seq
    double Make_uqSource(double tau, int n_cell_eta, int n_cell_x,
                         int n_cell_y,
                         double vis_array[][19],
                         double velocity_array[][20],
                         double grid_array[][5],
                         double vis_array_new[][19]);

#pragma acc routine seq
    double calculate_expansion_rate_1(
            double tau, Field *hydro_fields, int idx, int rk_flag);

#pragma acc routine seq
    void calculate_Du_supmu_1(double tau, Field *hydro_fields,
                                        int idx, int rk_flag, double *a);

#pragma acc routine seq
    void calculate_velocity_shear_tensor_2(
                    double tau, Field *hydro_fields, int idx, int rk_flag,
                    double *velocity_array);


#pragma acc routine seq
    double get_temperature_dependent_zeta_s(double temperature);

#pragma acc routine seq
    void update_grid_array_from_field(
                Field *hydro_fields, int idx, double *grid_array, int rk_flag);

#pragma acc routine seq
    void update_vis_array_from_field(Field *hydro_fields, int idx,
                                     double *vis_array, int rk_flag);

#pragma acc routine seq
    void update_vis_prev_tau_from_field(Field *hydro_fields, int idx,
                                        double *vis_array,int rk_flag);

#pragma acc routine seq
    void update_grid_array_to_hydro_fields(
            double *grid_array, Field *hydro_fields, int idx, int rk_flag);

#pragma acc routine seq
    void update_grid_cell(double grid_array[][5], Field *hydro_fields, int rk_flag,
                          int ieta, int ix, int iy,
                          int n_cell_eta, int n_cell_x, int n_cell_y);
#pragma acc routine seq
    void update_grid_cell_viscous(double vis_array[][19], Field *hydro_fields,
                                  int rk_flag, int ieta, int ix, int iy,
                                  int n_cell_eta, int n_cell_x, int n_cell_y);

#pragma acc routine seq
    int QuestRevert(double tau, double *vis_array, double *grid_array);
#pragma acc routine seq
    int QuestRevert_qmu(double tau, double *vis_array, double *grid_array);

    //! This function computes the vector [T^\tau\mu, J^\tau] from the
    //! grid_array [e, v^i, rhob]
#pragma acc routine seq
    void get_qmu_from_grid_array(double tau, double *qi, double *grid_array);
#pragma acc routine seq
    double minmod_dx(double up1, double u, double um1);
#pragma acc routine seq
    int map_2d_idx_to_1d(int a, int b);
};

#endif  // SRC_ADVANCE_H_
