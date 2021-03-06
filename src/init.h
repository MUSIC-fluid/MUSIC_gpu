// Copyright 2011 @ Bjoern Schenke, Sangyong Jeon, and Charles Gale
#ifndef SRC_INIT_H_
#define SRC_INIT_H_

#include <stdio.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "./data.h"
#include "./grid.h"
#include "./field.h"

class Init {
 private:
    InitData *DATA_ptr;
    EOS *eos;
    Util *util;

 public:
    Init(EOS *eos, InitData *DATA_in);
    ~Init();  // destructor

    void initialize_hydro_fields(Field *hydro_fields, InitData *DATA);

    void InitArena(InitData *DATA, Field *hydro_fields);

    void LinkNeighbors(InitData *DATA, Grid ****arena);
    void LinkNeighbors_XY(InitData *DATA, int ieta, Grid ***arena);

    int InitTJb(InitData *DATA, Field *hydro_fields);
    void initial_Bjorken_XY(InitData *DATA, int ieta, Field *hydro_fields);
    void initial_Gubser_XY(InitData *DATA, int ieta, Field *hydro_fields);
    void initial_IPGlasma_XY(InitData *DATA, int ieta, Field *hydro_fields);

    double eta_profile_normalisation(InitData *DATA, double eta);
    double eta_rhob_profile_normalisation(InitData *DATA, double eta);
    double eta_profile_left_factor(InitData *Data, double eta);
    double eta_profile_right_factor(InitData *Data, double eta);
    double eta_rhob_left_factor(InitData *Data, double eta);
    double eta_rhob_right_factor(InitData *Data, double eta);
    void output_initial_density_profiles(InitData *DATA, Grid ***arena);
};

#endif  // SRC_INIT_H_
