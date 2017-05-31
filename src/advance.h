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


    int AdvanceLocalT(double tau_init, InitData *DATA, int ieta, int ix,
                      Grid ***arena, int rk_flag);

    int FirstRKStepT(double tau, InitData *DATA, Grid *grid_pt, int rk_flag);

    int FirstRKStepW(double tau_it, InitData *DATA, Grid *grid_pt,
                     int rk_flag, double theta_local, double* a_local,
                     double *sigma_local);

    void UpdateTJbRK(Grid *grid_rk, Grid *grid_pt, int rk_flag);
    int QuestRevert(double tau, Grid *grid_pt, int rk_flag, InitData *DATA);
    int QuestRevert_qmu(double tau, Grid *grid_pt, int rk_flag,
                        InitData *DATA);

    void MakeDeltaQI(double tau, Grid *grid_pt, double *qi, int rk_flag);
    double MaxSpeed(double tau, int direc, Grid *grid_p);
    double get_TJb(Grid *grid_p, int rk_flag, int mu, int nu);
};

#endif  // SRC_ADVANCE_H_
