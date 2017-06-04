#include <omp.h>
#include "./util.h"
#include "./data.h"
#include "./grid.h"
#include "./minmod.h"
#include "./eos.h"
#include "./u_derivative.h"

using namespace std;

U_derivative::U_derivative(EOS *eosIn, InitData* DATA_in) {
   eos = eosIn;
   minmod = new Minmod(DATA_in);
   DATA_ptr = DATA_in;
}

// destructor
U_derivative::~U_derivative() {
   delete minmod;
}

int U_derivative::MakedU(double tau, Field *hydro_fields, int rk_flag) {
    // ideal hydro: no need to evaluate any flow derivatives
    if (DATA_ptr->viscosity_flag == 0) {
        return(1);
    }

    int neta = DATA_ptr->neta;
    int nx = DATA_ptr->nx + 1;
    int ny = DATA_ptr->ny + 1;
    int ieta;
    #pragma omp parallel private(ieta)
    {
        #pragma omp for
        for (ieta = 0; ieta < neta; ieta++) {
            for (int ix = 0; ix < nx; ix++) {
                for (int iy = 0; iy < ny; iy++) {
	               /* this calculates du/dx, du/dy, (du/deta)/tau */
                   MakeDSpatial_1(tau, hydro_fields, ieta, ix, iy, rk_flag);
                   /* this calculates du/dtau */
                   MakeDTau_1(tau, hydro_fields, ieta, ix, iy, rk_flag); 
                }
            }
        }
        #pragma omp barrier
    }

   return(1);
}

//! this function returns the expansion rate on the grid
double U_derivative::calculate_expansion_rate(
        double tau, Grid ***arena, int ieta, int ix, int iy, int rk_flag) {
    double partial_mu_u_supmu = 0.0;
    for (int mu = 0; mu < 4; mu++) {
        double gfac = (mu == 0 ? -1.0 : 1.0);
        // for expansion rate: theta
        partial_mu_u_supmu += arena[ieta][ix][iy].dUsup[0][mu][mu]*gfac;
    }
    double theta = partial_mu_u_supmu + arena[ieta][ix][iy].u[rk_flag][0]/tau;
    return(theta);
}

//! this function returns the expansion rate on the grid
double U_derivative::calculate_expansion_rate_1(
            double tau, Field *hydro_fields, int idx, int rk_flag) {
    double partial_mu_u_supmu = (- hydro_fields->dUsup[idx][0]
                                 + hydro_fields->dUsup[idx][5]
                                 + hydro_fields->dUsup[idx][10]
                                 + hydro_fields->dUsup[idx][15]);
    double theta = partial_mu_u_supmu;
    if (rk_flag == 0) {
        theta += hydro_fields->u_rk0[idx][0]/tau;
    } else {
        theta += hydro_fields->u_rk1[idx][0]/tau;
    }
    return(theta);
}


//! this function returns Du^\mu
void U_derivative::calculate_Du_supmu(double tau, Grid ***arena, int ieta,
                                      int ix, int iy, int rk_flag, double *a) {
    for (int mu = 0; mu <= 4; mu++) {
        double u_supnu_partial_nu_u_supmu = 0.0;
	    for (int nu = 0; nu < 4; nu++) {
            double tfac = (nu==0 ? -1.0 : 1.0);
            u_supnu_partial_nu_u_supmu += (
                tfac*arena[ieta][ix][iy].u[rk_flag][nu]
                *arena[ieta][ix][iy].dUsup[0][mu][nu]);
        }
        a[mu] = u_supnu_partial_nu_u_supmu;
    }
}
void U_derivative::calculate_Du_supmu_1(double tau, Field *hydro_fields,
                                        int idx, int rk_flag, double *a) {
    if (rk_flag == 0) {
        for (int mu = 0; mu < 5; mu++) {
            a[mu] = (
                - hydro_fields->u_rk0[idx][0]*hydro_fields->dUsup[idx][4*mu]
                + hydro_fields->u_rk0[idx][1]*hydro_fields->dUsup[idx][4*mu+1]
                + hydro_fields->u_rk0[idx][2]*hydro_fields->dUsup[idx][4*mu+2]
                + hydro_fields->u_rk0[idx][3]*hydro_fields->dUsup[idx][4*mu+3]
            );
        }
    } else {
        for (int mu = 0; mu < 5; mu++) {
            a[mu] = (
                - hydro_fields->u_rk1[idx][0]*hydro_fields->dUsup[idx][4*mu]
                + hydro_fields->u_rk1[idx][1]*hydro_fields->dUsup[idx][4*mu+1]
                + hydro_fields->u_rk1[idx][2]*hydro_fields->dUsup[idx][4*mu+2]
                + hydro_fields->u_rk1[idx][3]*hydro_fields->dUsup[idx][4*mu+3]
            );
        }
    }
}


//! This funciton returns the velocity shear tensor sigma^\mu\nu
void U_derivative::calculate_velocity_shear_tensor(double tau, Grid ***arena,
    int ieta, int ix, int iy, int rk_flag, double *a_local, double *sigma) {
    double dUsup_local[4][4];
    double u_local[4];
    double sigma_local[4][4];
    for (int i = 0; i < 4; i++) {
        u_local[i] = arena[ieta][ix][iy].u[rk_flag][i];
        for (int j = 0; j < 4; j++) {
            dUsup_local[i][j] = arena[ieta][ix][iy].dUsup[0][i][j];
        }
    }
    double theta_u_local = calculate_expansion_rate(tau, arena, ieta, ix,
                                                    iy, rk_flag);
    double gfac = 0.0;
    for (int a = 1; a < 4; a++) {
        for (int b = a; b < 4; b++) {
            if (b == a) {
                gfac = 1.0;
            } else {
                gfac = 0.0;
            }
            sigma_local[a][b] = ((dUsup_local[a][b] + dUsup_local[b][a])/2.
                - (gfac + u_local[a]*u_local[b])*theta_u_local/3.
                + u_local[0]/tau*DATA_ptr->gmunu[a][3]*DATA_ptr->gmunu[b][3]
                + u_local[3]*u_local[0]/tau/2.
                  *(DATA_ptr->gmunu[a][3]*u_local[b] 
                    + DATA_ptr->gmunu[b][3]*u_local[a])
                + (u_local[a]*a_local[b] + u_local[b]*a_local[a])/2.);
            sigma_local[b][a] = sigma_local[a][b];
        }
    }
    // make sigma[3][3] using traceless condition
    sigma_local[3][3] = (
        (  2.*(  u_local[1]*u_local[2]*sigma_local[1][2]
               + u_local[1]*u_local[3]*sigma_local[1][3]
               + u_local[2]*u_local[3]*sigma_local[2][3])
         - (u_local[0]*u_local[0] - u_local[1]*u_local[1])*sigma_local[1][1]
         - (u_local[0]*u_local[0] - u_local[2]*u_local[2])*sigma_local[2][2])
        /(u_local[0]*u_local[0] - u_local[3]*u_local[3]));
    // make sigma[0][i] using transversality
    for (int a = 1; a < 4; a++) {
        double temp = 0.0;
        for (int b = 1; b < 4; b++) {
            temp += sigma_local[a][b]*u_local[b];
        }
        sigma_local[0][a] = temp/u_local[0];
    }
    // make sigma[0][0]
    double temp = 0.0;
    for (int a = 1; a < 4; a++) {
        temp += sigma_local[0][a]*u_local[a];
    }
    sigma_local[0][0] = temp/u_local[0];

    sigma[0] = sigma_local[0][0];
    sigma[1] = sigma_local[0][1];
    sigma[2] = sigma_local[0][2];
    sigma[3] = sigma_local[0][3];
    sigma[4] = sigma_local[1][1];
    sigma[5] = sigma_local[1][2];
    sigma[6] = sigma_local[1][3];
    sigma[7] = sigma_local[2][2];
    sigma[8] = sigma_local[2][3];
    sigma[9] = sigma_local[3][3];
}
void U_derivative::calculate_velocity_shear_tensor_1(
                    double tau, Field *hydro_fields, int idx, int rk_flag,
                    double *a_local, double *sigma) {
    double dUsup_local[4][4];
    double u_local[4];
    double sigma_local[4][4];
    if (rk_flag == 0) {
        for (int i = 0; i < 4; i++) {
            u_local[i] = hydro_fields->u_rk0[idx][i];
        }
    } else {
        for (int i = 0; i < 4; i++) {
            u_local[i] = hydro_fields->u_rk1[idx][i];
        }
    }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            dUsup_local[i][j] = hydro_fields->dUsup[idx][j + 4*i];
        }
    }

    double theta_u_local = calculate_expansion_rate_1(tau, hydro_fields,
                                                      idx, rk_flag);
    double gfac = 0.0;
    for (int a = 1; a < 4; a++) {
        for (int b = a; b < 4; b++) {
            if (b == a) {
                gfac = 1.0;
            } else {
                gfac = 0.0;
            }
            sigma_local[a][b] = ((dUsup_local[a][b] + dUsup_local[b][a])/2.
                - (gfac + u_local[a]*u_local[b])*theta_u_local/3.
                + u_local[0]/tau*DATA_ptr->gmunu[a][3]*DATA_ptr->gmunu[b][3]
                + u_local[3]*u_local[0]/tau/2.
                  *(DATA_ptr->gmunu[a][3]*u_local[b] 
                    + DATA_ptr->gmunu[b][3]*u_local[a])
                + (u_local[a]*a_local[b] + u_local[b]*a_local[a])/2.);
            sigma_local[b][a] = sigma_local[a][b];
        }
    }
    // make sigma[3][3] using traceless condition
    sigma_local[3][3] = (
        (  2.*(  u_local[1]*u_local[2]*sigma_local[1][2]
               + u_local[1]*u_local[3]*sigma_local[1][3]
               + u_local[2]*u_local[3]*sigma_local[2][3])
         - (u_local[0]*u_local[0] - u_local[1]*u_local[1])*sigma_local[1][1]
         - (u_local[0]*u_local[0] - u_local[2]*u_local[2])*sigma_local[2][2])
        /(u_local[0]*u_local[0] - u_local[3]*u_local[3]));
    // make sigma[0][i] using transversality
    for (int a = 1; a < 4; a++) {
        double temp = 0.0;
        for (int b = 1; b < 4; b++) {
            temp += sigma_local[a][b]*u_local[b];
        }
        sigma_local[0][a] = temp/u_local[0];
    }
    // make sigma[0][0]
    double temp = 0.0;
    for (int a = 1; a < 4; a++) {
        temp += sigma_local[0][a]*u_local[a];
    }
    sigma_local[0][0] = temp/u_local[0];

    sigma[0] = sigma_local[0][0];
    sigma[1] = sigma_local[0][1];
    sigma[2] = sigma_local[0][2];
    sigma[3] = sigma_local[0][3];
    sigma[4] = sigma_local[1][1];
    sigma[5] = sigma_local[1][2];
    sigma[6] = sigma_local[1][3];
    sigma[7] = sigma_local[2][2];
    sigma[8] = sigma_local[2][3];
    sigma[9] = sigma_local[3][3];
}


int U_derivative::MakeDSpatial(double tau, InitData *DATA, Grid *grid_pt,
                               int rk_flag) {
    double g, f, fp1, fm1, taufactor;
    double delta[4];
    //Sangyong Nov 18 2014: added these doubles
    double rhob, eps, muB, T;
 
    delta[1] = DATA->delta_x;
    delta[2] = DATA->delta_y;
    delta[3] = DATA->delta_eta;

    /* dUsup[m][n] = partial_n u_m */
    /* for u[i] */
    for (int m = 1; m <= 3; m++) {
        // partial_n u[m]
        for (int n = 1; n <= 3; n++) {
            taufactor = 1.0;
            if (n == 3)
                taufactor = tau;
            f = grid_pt->u[rk_flag][m];
            fp1 = grid_pt->nbr_p_1[n]->u[rk_flag][m];
            fm1 = grid_pt->nbr_m_1[n]->u[rk_flag][m];
            g = minmod->minmod_dx(fp1, f, fm1);
            g /= delta[n]*taufactor;
            grid_pt->dUsup[0][m][n] = g;
        }  // n = x, y, eta
    }  // m = x, y, eta
    /* for u[0], use u[0]u[0] = 1 + u[i]u[i] */
    /* u[0]_m = u[i]_m (u[i]/u[0]) */
    /* for u[0] */
    for (int n = 1; n <= 3; n++) {
        f = 0.0;
        for (int m = 1; m <= 3; m++) {
	        /* (partial_n u^m) u[m] */
	        f += (grid_pt->dUsup[0][m][n])*(grid_pt->u[rk_flag][m]);
        }
        f /= grid_pt->u[rk_flag][0];
        grid_pt->dUsup[0][0][n] = f;
    }
    // Sangyong Nov 18 2014
    // Here we make derivatives of muB/T
    // dUsup[rk_flag][4][n] = partial_n (muB/T)
    // partial_x (muB/T) and partial_y (muB/T) first
    int m = 4; // means (muB/T)
    for (int n = 1; n <= 3; n++) {
        taufactor = 1.0;
        if (n == 3)
            taufactor = tau;

        // f = grid_pt->rhob_t;
        if (rk_flag == 0) {
            rhob = grid_pt->rhob;
            eps = grid_pt->epsilon;
        } else {
            rhob = grid_pt->rhob_t;
            eps = grid_pt->epsilon_t;
        }
        muB = eos->get_mu(eps, rhob);
        T = eos->get_temperature(eps, rhob);
        f = muB/T; 
    
        //fp1 = grid_pt->nbr_p_1[n]->rhob;
        if (rk_flag == 0) {
            rhob = grid_pt->nbr_p_1[n]->rhob;
            eps = grid_pt->nbr_p_1[n]->epsilon;
        } else {
            rhob = grid_pt->nbr_p_1[n]->rhob_t;
            eps = grid_pt->nbr_p_1[n]->epsilon_t;
        }
        muB = eos->get_mu(eps, rhob);
        T = eos->get_temperature(eps, rhob);
        fp1 = muB/T; 
       
        // fm1 = grid_pt->nbr_m_1[n]->rhob;
        if (rk_flag == 0) {
            rhob = grid_pt->nbr_m_1[n]->rhob;
            eps = grid_pt->nbr_m_1[n]->epsilon;
        } else {
            rhob = grid_pt->nbr_m_1[n]->rhob_t;
            eps = grid_pt->nbr_m_1[n]->epsilon_t;
        }
        muB = eos->get_mu(eps, rhob);
        T = eos->get_temperature(eps, rhob);
        fm1 = muB/T; 

        g = minmod->minmod_dx(fp1, f, fm1);
        g /= delta[n]*taufactor;
        grid_pt->dUsup[0][m][n] = g;
    }  // n = x, y, eta
    return 1;
}/* MakeDSpatial */

int U_derivative::MakeDSpatial_1(double tau, Field *hydro_fields, int ieta, int ix, int iy,
                                 int rk_flag) {
    int nx = DATA_ptr->nx + 1;
    int ny = DATA_ptr->ny + 1;
    int neta = DATA_ptr->neta;
    
    int idx = iy + ix*ny + ieta*ny*nx;

    double f, fp1, fm1, taufactor, deltafactor;
    double rhob, eps;
    int idx_p_1, idx_m_1;
    // dUsup[m][n] = partial_n u_m
    // for u[i]
    for (int m = 1; m < 5; m++) {
        for (int n = 1; n < 4; n++) {
            if (n == 1) {
                // compute partial_x u[m]
                if (ix + 1 > nx - 1) {
                    idx_p_1 = idx;
                } else {
                    idx_p_1 = idx + ny;
                }
                if (ix - 1 < 0) {
                    idx_m_1 = idx;
                } else {
                    idx_m_1 = idx - ny;
                }
                taufactor = 1.0;
                deltafactor = DATA_ptr->delta_x;
            } else if (n == 2) {
                // compute partial_y u[m]
                if (iy + 1 > ny - 1) {
                    idx_p_1 = idx;
                } else {
                    idx_p_1 = idx + 1;
                }
                if (iy - 1 < 0) {
                    idx_m_1 = idx;
                } else {
                    idx_m_1 = idx - 1;
                }
                taufactor = 1.0;
                deltafactor = DATA_ptr->delta_y;
            } else if (n == 3) {
                // compute partial_eta u[m]
                if (ieta + 1 > neta - 1) {
                    idx_p_1 = idx;
                } else {
                    idx_p_1 = idx + ny*nx;
                }
                if (ieta - 1 < 0) {
                    idx_m_1 = idx;
                } else {
                    idx_m_1 = idx - ny*nx;
                }
                taufactor = tau;
                deltafactor = DATA_ptr->delta_eta;
            }
            if (rk_flag == 0) {
                if (m < 4) {
                    f = hydro_fields->u_rk0[idx][m];
                    fp1 = hydro_fields->u_rk0[idx_p_1][m];
                    fm1 = hydro_fields->u_rk0[idx_m_1][m];
                } else if (m == 4) {
                    rhob = hydro_fields->rhob_rk0[idx];
                    eps = hydro_fields->e_rk0[idx];
                    f = eos->get_mu(eps, rhob)/eos->get_temperature(eps, rhob);
                    rhob = hydro_fields->rhob_rk0[idx_p_1];
                    eps = hydro_fields->e_rk0[idx_p_1];
                    fp1 = (eos->get_mu(eps, rhob)
                            /eos->get_temperature(eps, rhob));
                    rhob = hydro_fields->rhob_rk0[idx_m_1];
                    eps = hydro_fields->e_rk0[idx_m_1];
                    fm1 = (eos->get_mu(eps, rhob)
                            /eos->get_temperature(eps, rhob));
                }
            } else {
                if (m < 4) {
                    f = hydro_fields->u_rk1[idx][m];
                    fp1 = hydro_fields->u_rk1[idx_p_1][m];
                    fm1 = hydro_fields->u_rk1[idx_m_1][m];
                } else if (m == 4) {
                    rhob = hydro_fields->rhob_rk1[idx];
                    eps = hydro_fields->e_rk1[idx];
                    f = eos->get_mu(eps, rhob)/eos->get_temperature(eps, rhob);
                    rhob = hydro_fields->rhob_rk1[idx_p_1];
                    eps = hydro_fields->e_rk1[idx_p_1];
                    fp1 = (eos->get_mu(eps, rhob)
                             /eos->get_temperature(eps, rhob));
                    rhob = hydro_fields->rhob_rk1[idx_m_1];
                    eps = hydro_fields->e_rk1[idx_m_1];
                    fm1 = (eos->get_mu(eps, rhob)
                            /eos->get_temperature(eps, rhob));
                }
            }
            hydro_fields->dUsup[idx][4*m+n] = (minmod->minmod_dx(fp1, f, fm1)
                                               /(deltafactor*taufactor));
        }
    }


    // for u^tau, use u[0]u[0] = 1 + u[i]u[i]
    // partial^n u^tau = 1/u^tau (sum_i u^i partial^n u^i)
    if (rk_flag == 0) {
        for (int n = 1; n < 4; n++) {
            f = 0.0;
            for (int m = 1; m < 4; m++) {
	            f += (hydro_fields->dUsup[idx][4*m+n]
                      *hydro_fields->u_rk0[idx][m]);
            } 
            f /= hydro_fields->u_rk0[idx][0];
            hydro_fields->dUsup[idx][n] = f;
        }
    } else {
        for (int n = 1; n < 4; n++) {
            f = 0.0;
            for (int m = 1; m <= 3; m++) {
	            f += (hydro_fields->dUsup[idx][4*m+n]
                      *hydro_fields->u_rk1[idx][m]);
            }
            f /= hydro_fields->u_rk1[idx][0];
            hydro_fields->dUsup[idx][n] = f;
        }
    }
    return(1);
}/* MakeDSpatial */

int U_derivative::MakeDTau(double tau, InitData *DATA, Grid *grid_pt,
                           int rk_flag) {
    int m;
    double f;
    double tildemu, tildemu_prev, rhob, eps, muB, T;
    /* this makes dU[m][0] = partial^tau u^m */
    /* note the minus sign at the end because of g[0][0] = -1 */
    /* rk_flag is 0, 2, 4, ... */

    if (rk_flag == 0) {
        for (m=1; m<=3; m++) {
            /* first order is more stable */
            f = ((grid_pt->u[rk_flag][m] - grid_pt->prev_u[0][m])
                 /DATA->delta_tau);
            grid_pt->dUsup[0][m][0] = -f; /* g00 = -1 */
        }/* m */
    } else if (rk_flag > 0) {
        for (m=1; m<=3; m++) {
            /* first order */
            // this is from the prev full RK step 
            f = (grid_pt->u[rk_flag][m] - grid_pt->u[0][m])/(DATA->delta_tau);
            grid_pt->dUsup[0][m][0] = -f; /* g00 = -1 */
        }/* m */
    }

    /* I have now partial^tau u^i */
    /* I need to calculate (u^i partial^tau u^i) = u^0 partial^tau u^0 */
    /* u_0 d^0 u^0 + u_m d^0 u^m = 0 */
    /* -u^0 d^0 u^0 + u_m d^0 u^m = 0 */
    /* d^0 u^0 = u_m d^0 u^m/u^0 */

    f = 0.0;
    for (m=1; m<=3; m++) {
        /* (partial_0 u^m) u[m] */
        f += (grid_pt->dUsup[0][m][0])*(grid_pt->u[rk_flag][m]);
    }
    f /= grid_pt->u[rk_flag][0];
    grid_pt->dUsup[0][0][0] = f;

    // Sangyong Nov 18 2014
    // Here we make the time derivative of (muB/T)
    if (rk_flag == 0) {
        m = 4;  
        // first order is more stable 
        // backward derivative
        // current values
        // f = (grid_pt->rhob);
        rhob = grid_pt->rhob;
        eps = grid_pt->epsilon;
        muB = eos->get_mu(eps, rhob);
        T = eos->get_temperature(eps, rhob);
        tildemu = muB/T;
        // f -= (grid_pt->rhob_prev);
        rhob = grid_pt->prev_rhob;
        eps = grid_pt->prev_epsilon;
        muB = eos->get_mu(eps, rhob);
        T = eos->get_temperature(eps, rhob);
        tildemu_prev = muB/T;
        f = (tildemu - tildemu_prev)/(DATA->delta_tau);
        grid_pt->dUsup[0][m][0] = -f; /* g00 = -1 */
    } else if (rk_flag > 0) {
        m = 4;  
        // first order 
        // forward derivative
        // f = (grid_pt->rhob_t); // this is from the prev full RK step 
        // f -= (grid_pt->rhob_prev);
        // f /= (DATA->delta_tau);
        rhob = grid_pt->rhob_t;
        eps = grid_pt->epsilon_t;
        muB = eos->get_mu(eps, rhob);
        T = eos->get_temperature(eps, rhob);
        tildemu = muB/T;
        // f -= (grid_pt->rhob_prev);
        rhob = grid_pt->rhob;
        eps = grid_pt->epsilon;
        muB = eos->get_mu(eps, rhob);
        T = eos->get_temperature(eps, rhob);
        tildemu_prev = muB/T;
        f = (tildemu - tildemu_prev)/(DATA->delta_tau);
        grid_pt->dUsup[0][m][0] = -f; /* g00 = -1 */
    }
    // Ends Sangyong's addition Nov 18 2014
    return 1;
}/* MakeDTau */

int U_derivative::MakeDTau_1(double tau, Field *hydro_fields, int ieta, int ix, int iy,
                             int rk_flag) {
    int nx = DATA_ptr->nx + 1;
    int ny = DATA_ptr->ny + 1;
    
    int idx = iy + ix*ny + ieta*ny*nx;

    double f;
    /* this makes dU[m][0] = partial^tau u^m */
    /* note the minus sign at the end because of g[0][0] = -1 */
    if (rk_flag == 0) {
        for (int m = 1; m < 4; m++) {
            f = ((hydro_fields->u_rk0[idx][m] - hydro_fields->u_prev[idx][m])
                 /DATA_ptr->delta_tau);
            hydro_fields->dUsup[idx][4*m] = -f;  // g00 = -1
        }
    } else {
        for (int m = 1; m < 4; m++) {
            f = ((hydro_fields->u_rk1[idx][m] - hydro_fields->u_rk0[idx][m])
                 /DATA_ptr->delta_tau);
            hydro_fields->dUsup[idx][4*m] = -f;  // g00 = -1
        }
    }

    /* I have now partial^tau u^i */
    /* I need to calculate (u^i partial^tau u^i) = u^0 partial^tau u^0 */
    /* u_0 d^0 u^0 + u_m d^0 u^m = 0 */
    /* -u^0 d^0 u^0 + u_m d^0 u^m = 0 */
    /* d^0 u^0 = u_m d^0 u^m/u^0 */

    f = 0.0;
    if (rk_flag == 0) {
        for (int m = 1; m < 4; m++) {
            f += hydro_fields->dUsup[idx][4*m]*hydro_fields->u_rk0[idx][m];
        }
        f /= hydro_fields->u_rk0[idx][0];
        hydro_fields->dUsup[idx][0] = f;
    } else {
        for (int m = 1; m < 4; m++) {
            f += hydro_fields->dUsup[idx][4*m]*hydro_fields->u_rk1[idx][m];
        }
        f /= hydro_fields->u_rk1[idx][0];
        hydro_fields->dUsup[idx][0] = f;
    }

    // Here we make the time derivative of (muB/T)
    double tildemu, tildemu_prev, rhob, eps;
    if (rk_flag == 0) {
        rhob = hydro_fields->rhob_rk0[idx];
        eps = hydro_fields->e_rk0[idx];
        tildemu = eos->get_mu(eps, rhob)/eos->get_temperature(eps, rhob);
        rhob = hydro_fields->rhob_prev[idx];
        eps = hydro_fields->e_prev[idx];
        tildemu_prev = eos->get_mu(eps, rhob)/eos->get_temperature(eps, rhob);
        f = (tildemu - tildemu_prev)/(DATA_ptr->delta_tau);
        hydro_fields->dUsup[0][16] = -f;   // g00 = -1
    } else if (rk_flag > 0) {
        rhob = hydro_fields->rhob_rk1[idx];
        eps = hydro_fields->e_rk1[idx];
        tildemu = eos->get_mu(eps, rhob)/eos->get_temperature(eps, rhob);
        rhob = hydro_fields->rhob_rk0[idx];
        eps = hydro_fields->e_rk0[idx];
        tildemu_prev = eos->get_mu(eps, rhob)/eos->get_temperature(eps, rhob);
        f = (tildemu - tildemu_prev)/(DATA_ptr->delta_tau);
        hydro_fields->dUsup[0][16] = -f;   // g00 = -1
    }
    return(1);
}

