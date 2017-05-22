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

 public:
    Diss(EOS *eosIn, InitData* DATA_in);
    ~Diss();
  
    double MakeWSource(double tau, int alpha, Grid *grid_pt, InitData *DATA,
                       int rk_flag);
    int Make_uWRHS(double tau, Grid *grid_pt, double **w_rhs, InitData *DATA,
                   int rk_flag);
    void Get_uWmns(double tau, Grid *grid_pt, int mu, int nu, int direc,
                   double *g, double *f, double *gp1, double *fp1, double *gp2,
                   double *fp2, double *gm1, double *fm1, double *gm2,
                   double *fm2, InitData *DATA, int rk_flag);
    double Make_uWSource(double tau, Grid *grid_pt, int mu, int nu,
                         InitData *DATA, int rk_flag);
    
    int Make_uPRHS(double tau, Grid *grid_pt, double *p_rhs, InitData *DATA,
                   int rk_flag);
    void Get_uPis(double tau, Grid *grid_pt, int direc, double *g, double *f,
                  double *gp1, double *fp1, double *gp2, double *fp2,
                  double *gm1, double *fm1, double *gm2, double *fm2,
                  InitData *DATA, int rk_flag); 

    double Make_uPiSource(double tau, Grid *grid_pt, InitData *DATA,
                          int rk_flag);
    int Make_uqRHS(double tau, Grid *grid_pt, double **w_rhs, InitData *DATA,
                   int rk_flag);
    double Make_uqSource(double tau, Grid *grid_pt, int nu, InitData *DATA,
                         int rk_flag); 
    double get_temperature_dependent_eta_s(InitData *DATA, double T);
    double get_temperature_dependent_zeta_s(double temperature);
};

#endif  // SRC_DISSIPATIVE_H_
