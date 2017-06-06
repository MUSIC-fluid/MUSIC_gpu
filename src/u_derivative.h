#ifndef U_DERIVATIVE_H_
#define U_DERIVATIVE_H_

#include "util.h"
#include "data.h"
#include "grid.h"
#include "field.h"
#include <string.h>
#include <iostream>

class U_derivative {
 private:
     Minmod *minmod;
     // Sangyong Nov 18 2014: added EOS *eos;
     EOS *eos;
     InitData *DATA_ptr;
  
 public:
    // Sangyong Nov 18 2014: added EOS *eos in the argument
    U_derivative(EOS *eosIn, InitData* DATA_in);  // constructor
    ~U_derivative();
    int MakedU(double tau, Field *hydro_fields, int rk_flag);
    void MakedUXY(double tau, int ieta, InitData *DATA,
                  Grid ***arena, int rk_flag);

    //! this function returns the expansion rate on the grid
    double calculate_expansion_rate(double tau, Grid ***arena,
                                    int ieta, int ix, int iy, int rk_flag);
    double calculate_expansion_rate_1(double tau, Field *hydro_fields,
                                      int idx, int rk_flag);

    //! this function returns Du^\mu
    void calculate_Du_supmu(double tau, Grid ***arena, int ieta, int ix,
                            int iy, int rk_flag, double *a);
    void calculate_Du_supmu_1(double tau, Field *hydro_fields,
                              int idx, int rk_flag, double *a);

    //! This funciton returns the velocity shear tensor sigma^\mu\nu
    void calculate_velocity_shear_tensor(double tau, Grid ***arena,
        int ieta, int ix, int iy, int rk_flag, double *a_local, double *sigma);
    void calculate_velocity_shear_tensor_1(
                    double tau, Field *hydro_fields, int idx, int rk_flag,
                    double *a_local, double *sigma);
    void calculate_velocity_shear_tensor_2(
                    double tau, Field *hydro_fields, int idx, int rk_flag,
                    double *velocity_array);
    int MakeDSpatial(double tau, InitData *DATA, Grid *grid_pt, int rk_flag);
    int MakeDSpatial_1(double tau, Field *hydro_fields, int ieta, int ix, int iy, int rk_flag);
    int MakeDTau(double tau, InitData *DATA, Grid *grid_pt, int rk_flag);
    int MakeDTau_1(double tau, Field *hydro_fields, int ieta, int ix, int iy, int rk_flag);
};
#endif
