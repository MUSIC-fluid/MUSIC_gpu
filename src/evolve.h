// Copyright 2012 Bjoern Schenke, Sangyong Jeon, and Charles Gale
//
#define GUBSER_Q 2.0

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
#include "./field.h"
#include "./grid_info.h"
#include "./eos.h"
#include "./advance.h"

// this is a control class for the hydrodynamic evolution
class Evolve {
 private:
    EOS *eos;        // declare EOS object
    Grid_info *grid_info;
    Util *util;
    Advance *advance;

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
    int EvolveIt(InitData *DATA, Field *hydro_fields);

    void clean_up_hydro_fields(Field *hydro_fields);

    int AdvanceRK(double tau, InitData *DATA, Field *hydro_fields);

    //! This function is a shell function to freeze-out fluid cells
    //! outside the freeze-out energy density at the first time step
    //! of the evolution
    int FreezeOut_equal_tau_Surface(double tau, InitData *DATA,
                                    Field *hydro_fields);

    //! This function freeze-outs fluid cells
    //! outside the freeze-out energy density at the first time step
    //! of the evolution in the transverse plane
    void FreezeOut_equal_tau_Surface_XY(double tau, InitData *DATA,
                                        int ieta, Field *hydro_fields,
                                        int thread_id, double epsFO);

    //! This is a function to prepare the freeze-out cube
    void prepare_freeze_out_cube(double ****cube,
                                 double* data_prev, double* data_array,
                                 int ieta, int ix, int iy,
                                 int fac_eta, int fac_x, int fac_y);

    //! This is a function to prepare the freeze-out cube for boost-invariant
    //! surface
    void prepare_freeze_out_cube_boost_invariant(
                    double ***cube, double* data_prev, double* data_array,
                    int ix, int iy, int fac_x, int fac_y);

    int FindFreezeOutSurface_Cornelius(double tau, InitData *DATA,
                                       Field *hydro_fields);

    int FindFreezeOutSurface_Cornelius_XY(double tau, InitData *DATA,
                                          int ieta, Field *hydro_fields,
                                          int thread_id, double epsFO);

    int FindFreezeOutSurface_boostinvariant_Cornelius(
                            double tau, InitData *DATA, Field *hydro_fields);

    void store_previous_step_for_freezeout(Field *hydro_fields);

    void regulate_qmu(double* u, double* q, double* q_regulated);
    void regulate_Wmunu(double* u, double** Wmunu, double** Wmunu_regulated);

    void initialize_freezeout_surface_info();
    double energy_gubser(double tau, double x, double y);
    void flow_gubser(double tau, double x, double y, double * utau, double * ux, double * uy);
    void check_field_with_ideal_Gubser(double tau, Field *hydro_fields);
    void initial_field_with_ideal_Gubser(double tau, Field *hydro_fields);
};

#endif  // SRC_EVOLVE_H_

