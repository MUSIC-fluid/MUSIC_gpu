// Copyright 2012 Bjoern Schenke, Sangyong Jeon, and Charles Gale
#ifndef SRC_EVOLVE_H_
#define SRC_EVOLVE_H_

#include <time.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "./util.h"
#include "./data.h"
#include "./grid.h"
#include "./field.h"
#include "./grid_info.h"
#include "./eos.h"
#include "./advance.h"
#include "./u_derivative.h"

// this is a control class for the hydrodynamic evolution
class Evolve {
 private:
    EOS *eos;        // declare EOS object
    Grid *grid;      // declare Grid object
    Grid_info *grid_info;
    Util *util;
    Advance *advance;
    U_derivative *u_derivative;

    InitData *DATA_ptr;

    // simulation information
    int rk_order;
    int grid_nx, grid_ny, grid_neta;

    double SUM;
    int warnings;
    int cells;
    int weirdCases;
    int facTau;

    // information about freeze-out surface
    // (only used when freezeout_method == 4)
    int n_freeze_surf;
    vector<double> epsFO_list;

 public:
    Evolve(EOS *eos, InitData *DATA_in);
    ~Evolve();
    int EvolveIt(InitData *DATA, Grid ***arena);

    void initialize_hydro_fields(Field *hydro_fields);
    void clean_up_hydro_fields(Field *hydro_fields);
    void copy_dUsup_from_grid_to_field(Grid ***arena, Field *hydro_fields);
    void copy_fields_to_grid(Field *hydro_fields, Grid ***arena);
    void convert_grid_to_field(Grid ***arena, Field *hydro_fields);

    int AdvanceRK(double tau, InitData *DATA, Field *hydro_fields);
    int Update_prev_Arena(Grid ***arena);
    void Update_prev_Arena_XY(int ieta, Grid ***arena);

    void update_prev_field(Field *hydro_fields);

    int FreezeOut_equal_tau_Surface(double tau, InitData *DATA, Grid ***arena);
    void FreezeOut_equal_tau_Surface_XY(double tau, InitData *DATA,
                                        int ieta, Grid ***arena,
                                        int thread_id, double epsFO);
    // void FindFreezeOutSurface(double tau, InitData *DATA,
    //                          Grid ***arena, int size, int rank);
    // void FindFreezeOutSurface2(double tau, InitData *DATA,
    //                           Grid ***arena, int size, int rank);
    // int FindFreezeOutSurface3(double tau, InitData *DATA,
    //                          Grid ***arena, int size, int rank);
    int FindFreezeOutSurface_Cornelius(double tau, InitData *DATA,
                                       Grid ***arena);
    int FindFreezeOutSurface_Cornelius_XY(double tau, InitData *DATA,
                                          int ieta, Grid ***arena,
                                          int thread_id, double epsFO);
    int FindFreezeOutSurface_boostinvariant_Cornelius(
                                    double tau, InitData *DATA, Grid ***arena);

    void store_previous_step_for_freezeout(Grid ***arena);

    void regulate_qmu(double* u, double* q, double* q_regulated);
    void regulate_Wmunu(double* u, double** Wmunu, double** Wmunu_regulated);

    void initialize_freezeout_surface_info();
};

#endif  // SRC_EVOLVE_H_

