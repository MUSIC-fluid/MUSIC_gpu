// Copyright 2011 @ Bjoern Schenke, Sangyong Jeon, and Charles Gale
#include <omp.h>
#include "./data.h"
#include "./evolve.h"
#include "./advance.h"

#define THETA_FLUX 1.8
#define RHO_MAX 0.1

using namespace std;

Advance::Advance(InitData *DATA_in) {
    DATA_ptr = DATA_in;

    grid_nx = DATA_in->nx;
    grid_ny = DATA_in->ny;
    grid_neta = DATA_in->neta;
    rk_order = DATA_in->rk_order;
}

// destructor
Advance::~Advance() {
}


// evolve Runge-Kutta step in tau
int Advance::AdvanceIt(double tau, Field *hydro_fields,
                       int rk_flag) {
    double tmp[1]={-1.1};

    double grid_array[5];
    
    // enter GPU parallel region, variable tmp is used to check whether the
    // code is running on GPU
    #pragma acc parallel loop gang worker vector collapse(3) independent \
        present(hydro_fields) private(this[0:1])
    for (int ieta = 0; ieta < GRID_SIZE_ETA; ieta++) {
        for (int ix = 0; ix <= GRID_SIZE_X; ix++) {
            for (int iy = 0; iy <= GRID_SIZE_Y; iy++) {
                int idx = get_indx(ieta, ix, iy);
                double tau_rk = tau + rk_flag*DELTA_TAU;
                calculate_qi_array(tau_rk, hydro_fields, idx);
            }
        }
    }

    #pragma acc parallel loop gang worker vector collapse(3) independent \
        present(hydro_fields) \
        private(this[0:1], grid_array[0:5])
    for (int ieta = 0; ieta < GRID_SIZE_ETA; ieta++) {
        for (int ix = 0; ix <= GRID_SIZE_X; ix++) {
            for (int iy = 0; iy <= GRID_SIZE_Y; iy++) {
                //tmp[0]=tau;  // check code is running on GPU

                FirstRKStepT(tau, rk_flag, hydro_fields, ieta, ix, iy,
                             grid_array);
            }
        }
    }
    
    #pragma acc parallel loop gang worker vector collapse(3) independent \
        present(hydro_fields) private(this[0:1])
    for (int ieta = 0; ieta < GRID_SIZE_ETA; ieta++) {
        for (int ix = 0; ix <= GRID_SIZE_X; ix++) {
            for (int iy = 0; iy <= GRID_SIZE_Y; iy++) {
                int idx = get_indx(ieta, ix, iy);
                double tau_rk = tau + rk_flag*DELTA_TAU;
                // now MakeWSource returns partial_a W^{a mu}
                // (including geometric terms) 
                MakeWSource(tau_rk, hydro_fields, ieta, ix, iy);

                if (rk_flag == 1) {
                    // if rk_flag == 1, we now have q0 + k1 + k2. 
                    // So add q0 and multiply by 1/2
                    double pressure = get_pressure(
                                                hydro_fields->e_prev[idx],
                                                hydro_fields->rhob_prev[idx]);
                    hydro_fields->qi_array_new[0][idx] += (
                        tau*((hydro_fields->e_prev[idx] + pressure)
                              *hydro_fields->u_prev[0][idx]
                              *hydro_fields->u_prev[0][idx] - pressure));
                    hydro_fields->qi_array_new[1][idx] += (
                        tau*((hydro_fields->e_prev[idx] + pressure)
                              *hydro_fields->u_prev[0][idx]
                              *hydro_fields->u_prev[1][idx]));
                    hydro_fields->qi_array_new[2][idx] += (
                        tau*((hydro_fields->e_prev[idx] + pressure)
                              *hydro_fields->u_prev[0][idx]
                              *hydro_fields->u_prev[2][idx]));
                    hydro_fields->qi_array_new[3][idx] += (
                        tau*((hydro_fields->e_prev[idx] + pressure)
                              *hydro_fields->u_prev[0][idx]
                              *hydro_fields->u_prev[3][idx]));
                    hydro_fields->qi_array_new[4][idx] += (
                        tau*(hydro_fields->rhob_prev[idx]
                             *hydro_fields->u_prev[0][idx]));
                    for (int alpha = 0; alpha < 5; alpha++) {
                        hydro_fields->qi_array_new[alpha][idx] *= 0.5;
                    }
                }
            }
        }
    }
    #pragma acc parallel loop gang worker vector collapse(3) independent \
        present(hydro_fields) \
        private(this[0:1], grid_array[0:5])
    for (int ieta = 0; ieta < GRID_SIZE_ETA; ieta++) {
        for (int ix = 0; ix <= GRID_SIZE_X; ix++) {
            for (int iy = 0; iy <= GRID_SIZE_Y; iy++) {
                update_grid_cell(grid_array, hydro_fields, ieta, ix, iy,
                                 tau + DELTA_TAU);
            }
        }
    }

    if (VISCOUS_FLAG == 1) {
        #pragma acc parallel loop gang worker vector collapse(3) independent \
            present(hydro_fields) private(this[0:1])
        for (int ieta = 0; ieta < GRID_SIZE_ETA; ieta++) {
            for (int ix = 0; ix <= GRID_SIZE_X; ix++) {
                for (int iy = 0; iy <= GRID_SIZE_Y; iy++) {
                    double tau_rk = tau + rk_flag*DELTA_TAU;
                    calculate_u_derivatives(tau_rk, hydro_fields,
                                            ieta, ix, iy);
                }
            }
        }
        
        if (INCLUDE_DIFF == 1) {
            #pragma acc parallel loop gang worker vector collapse(3) independent \
                present(hydro_fields) private(this[0:1])
            for (int ieta = 0; ieta < GRID_SIZE_ETA; ieta++) {
                for (int ix = 0; ix <= GRID_SIZE_X; ix++) {
                    for (int iy = 0; iy <= GRID_SIZE_Y; iy++) {
                        double tau_rk = tau + rk_flag*DELTA_TAU;
                        calculate_D_mu_muB_over_T(tau_rk, hydro_fields,
                                                  ieta, ix, iy);
                    }
                }
            }
        }

        #pragma acc parallel loop gang worker vector collapse(3) independent \
            present(hydro_fields)\
            private(this[0:1])
        for (int ieta = 0; ieta < GRID_SIZE_ETA; ieta++) {
            for (int ix = 0; ix <= GRID_SIZE_X; ix++) {
                for (int iy = 0; iy <= GRID_SIZE_Y; iy++) {
                    FirstRKStepW(tau, rk_flag, hydro_fields, ieta, ix, iy);
                }
            }
        }
        
        if (INITIAL_PROFILE > 2) {
            #pragma acc parallel loop gang worker vector collapse(3) \
                independent present(hydro_fields)
            for (int ieta = 0; ieta < GRID_SIZE_ETA; ieta++) {
                for (int ix = 0; ix <= GRID_SIZE_X; ix++) {
                    for (int iy = 0; iy <= GRID_SIZE_Y; iy++) {
                        int idx = get_indx(ieta, ix, iy);
                        QuestRevert(hydro_fields, idx);
                    }
                }
            }
        }
    }
    

    // update the hydro fields
    if (rk_flag == 0) {
        #pragma acc parallel loop gang worker vector collapse(3) independent \
            present(hydro_fields) private(this[0:1])
        for (int ieta = 0; ieta < GRID_SIZE_ETA; ieta++) {
            for (int ix = 0; ix <= GRID_SIZE_X; ix++) {
                for (int iy = 0; iy <= GRID_SIZE_Y; iy++) {
                    int indx = get_indx(ieta, ix, iy);
                    update_field_rk0_to_prev(hydro_fields, indx);
                    update_field_rk1_to_rk0(hydro_fields, indx);
                }
            }
        }
    } else {
        #pragma acc parallel loop gang worker vector collapse(3) independent \
            present(hydro_fields) private(this[0:1])
        for (int ieta = 0; ieta < GRID_SIZE_ETA; ieta++) {
            for (int ix = 0; ix <= GRID_SIZE_X; ix++) {
                for (int iy = 0; iy <= GRID_SIZE_Y; iy++) {
                    int indx = get_indx(ieta, ix, iy);
                    update_field_rk1_to_rk0(hydro_fields, indx);
                }
            }
        }
    }

    //std::cout << "check tmp=" << tmp[0]  << "\n";

    return(1);
}/* AdvanceIt */


void Advance::calculate_qi_array(double tau, Field *hydro_fields, int idx) {
    double e = hydro_fields->e_rk0[idx];
    double pressure = get_pressure(e, hydro_fields->rhob_rk0[idx]);
    for (int i = 0; i < 4; i++) {
        // here we compute tau*T^\tau\mu = tau*((e + P)*u^0*u^mu)
        hydro_fields->qi_array[i][idx] = (
                tau*((e + pressure)*hydro_fields->u_rk0[0][idx]
                                   *hydro_fields->u_rk0[i][idx]));
    }
    // for tau*T^\tau\tau need one more term + tau*(P*g^00)
    hydro_fields->qi_array[0][idx] -= tau*pressure;

    // J^tau = rhob*u^tau
    hydro_fields->qi_array[4][idx] = (
            tau*hydro_fields->rhob_rk0[idx]*hydro_fields->u_rk0[0][idx]);

    for (int i = 0; i < 5; i++) {
        hydro_fields->qi_array_new[i][idx] = 0.0;
    }
}


/* %%%%%%%%%%%%%%%%%%%%%% First steps begins here %%%%%%%%%%%%%%%%%% */
int Advance::FirstRKStepT(double tau, int rk_flag,
                          Field *hydro_fields, int ieta, int ix, int iy,
                          double *grid_array) {

    // this advances the ideal part
    double tau_rk = tau + rk_flag*DELTA_TAU;

    int idx = get_indx(ieta, ix, iy);
    update_grid_array_from_field(hydro_fields, idx, grid_array);
    
    // Solve partial_a T^{a mu} = -partial_a W^{a mu}
    // Update T^{mu nu}

    // MakeDelatQI gets
    //   qi = q0 if rk_flag = 0 or
    //   qi = q0 + k1 if rk_flag = 1
    // rhs[alpha] is what MakeDeltaQI outputs. 
    // It is the spatial derivative part of partial_a T^{a mu}
    // (including geometric terms)
    MakeDeltaQI(tau_rk, grid_array, hydro_fields, ieta, ix, iy);
    return(0);
}

int Advance::ReconstIt_velocity_Newton(
    double *grid_array, double tau, double *uq, double *grid_array_p) {
    double abs_err = 1e-9;
    double rel_err = 1e-8;
    int max_iter = 100;
    double v_critical = 0.53;
    /* prepare for the iteration */
    /* uq = qiphL, qiphR, etc 
       qiphL[alpha] means, for instance, TJ[alpha][0] 
       in the cell at x+dx/2 calculated from the left */
    
    /* uq are the conserved charges. That is, the ones appearing in
       d_tau (Ttautau/tau) + d_eta(Ttaueta/tau) + d_perp(Tperptau) = -Tetaeta
       d_tau (Ttaueta) + d_eta(Tetaeta) + d_v(tau Tveta) = -Ttaueta/tau 
       d_tau (Ttauv) + d_eta(Tetav) + d_w(tau Twv) = 0
       d_tau (Jtau) + d_eta Jeta + d_perp(tau Jperp) = 0 */
    
    /* q[0] = Ttautau/tau, q[1] = Ttaux, q[2] = Ttauy, q[3] = Ttaueta
       q[4] = Jtau */
    /* uq = qiphL, qiphR, qimhL, qimhR, qirk */

    double K00 = (uq[1]*uq[1] + uq[2]*uq[2] + uq[3]*uq[3])/(tau*tau);
    double M = sqrt(K00);
    double T00 = uq[0]/tau;
    double J0 = uq[4]/tau;

    if ((T00 < abs_err) || ((T00 - K00/T00) < 0.0)) {
        revert_grid(grid_array, grid_array_p);
        return(-1);
    }/* if t00-k00/t00 < 0.0 */

    double u0, u1, u2, u3, epsilon, pressure, rhob;

    double v_guess = 0.0;
    double u0_guess = 1.0;
    int v_status = 1;
    int iter = 0;
    double rel_error_v = 10.0;
    double v_next = 1.0;
    double v_prev = v_guess;
    double abs_error_v = reconst_velocity_f_Newton(v_prev, T00, M, J0);
    do {
        iter++;
        v_next = (v_prev
                  - (abs_error_v/reconst_velocity_df(v_prev, T00, M, J0)));
        if (v_next < 0.0) {
            v_next = 0.0 + 1e-10;
        } else if (v_next > 1.0) {
            v_next = 1.0 - 1e-10;
        }
        abs_error_v = reconst_velocity_f_Newton(v_next, T00, M, J0);
        rel_error_v = 2.*abs_error_v/(v_next + v_prev + 1e-15);
        v_prev = v_next;
        if (iter > max_iter) {
            v_status = 0;
            break;
        }
    } while (fabs(abs_error_v) > abs_err && fabs(rel_error_v) > rel_err);

    double v_solution;
    if (v_status == 1 && v_next >= 0. && v_next <= 1.0) {
        v_solution = v_next;
    } else {
        revert_grid(grid_array, grid_array_p);
        return(-1);
    }/* if iteration is unsuccessful, revert */
   
    // for large velocity, solve u0
    double u0_solution = 1.0;
    double abs_error_u0, rel_error_u0;
    if (v_solution > v_critical) {
        double u0_prev = 1./sqrt(1. - v_solution*v_solution);
        int u0_status = 1;
        iter = 0;
        double u0_next;
        abs_error_u0 = reconst_u0_f_Newton(u0_prev, T00, K00, M, J0);
        rel_error_u0 = 1.0;
        do {
            iter++;
            u0_next = (u0_prev
                       - abs_error_u0/reconst_u0_df(u0_prev, T00, K00, M, J0));
            abs_error_u0 = reconst_u0_f_Newton(u0_next, T00, K00, M, J0);
            rel_error_u0 = 2.*abs_error_u0/(u0_next + u0_prev + 1e-15);
            u0_prev = u0_next;
            if (iter > max_iter) {
                u0_status = 0;
                break;
            }
        } while (fabs(abs_error_u0) > abs_err && fabs(rel_error_u0) > rel_err);

        if (u0_status == 1 && u0_next >= 1.0) {
            u0_solution = u0_next;
        } else {
            revert_grid(grid_array, grid_array_p);
            return(-1);
        }  // if iteration is unsuccessful, revert
    }

    // successfully found velocity, now update everything else
    if (v_solution < v_critical) {
        u0 = 1./(sqrt(1. - v_solution*v_solution) + v_solution*abs_err);
        epsilon = T00 - v_solution*sqrt(K00);
        rhob = J0/u0;
    } else {
        u0 = u0_solution;
        epsilon = T00 - sqrt((1. - 1./(u0_solution*u0_solution))*K00);
        rhob = J0/u0_solution;
    }

    double check_u0_var = (fabs(u0 - u0_guess)/u0_guess);
    if (check_u0_var > 100.) {
        revert_grid(grid_array, grid_array_p);
        return(-1);
    }

    grid_array[0] = epsilon;
    grid_array[4] = rhob;

    pressure = get_pressure(epsilon, rhob);

    // individual components of velocity
    double velocity_inverse_factor = u0/(T00 + pressure);

    double u_max = 242582597.70489514; // cosh(20)
    //remove if for speed
    if(u0 > u_max) {
        revert_grid(grid_array, grid_array_p);
        return(-1);
    } else {
        u1 = uq[1]*velocity_inverse_factor/tau; 
        u2 = uq[2]*velocity_inverse_factor/tau; 
        u3 = uq[3]*velocity_inverse_factor/tau;
    }

    // Correcting normalization of 4-velocity
    double temp_usq = u0*u0 - u1*u1 - u2*u2 - u3*u3;
    // Correct velocity when unitarity is not satisfied to numerical accuracy
    if (fabs(temp_usq - 1.0) > abs_err) {
        // If the deviation is too large, exit MUSIC
        if (fabs(temp_usq - 1.0) > 0.1*u0) {
            revert_grid(grid_array, grid_array_p);
            return(-1);
        }
        // Rescaling spatial components of velocity so that unitarity 
        // is exactly satisfied (u[0] is not modified)
        double scalef = sqrt((u0*u0 - 1.0)
                             /(u1*u1 + u2*u2 + u3*u3 + abs_err));
        u1 *= scalef;
        u2 *= scalef;
        u3 *= scalef;
    }  // if u^mu u_\mu != 1 
    // End: Correcting normalization of 4-velocity
   
    grid_array[1] = u1/u0;
    grid_array[2] = u2/u0;
    grid_array[3] = u3/u0;

    return(1);  /* on successful execution */
}

double Advance::reconst_velocity_f(double v, double T00, double M,
                                   double J0) {
    // this function returns f(v) = M/(M0 + P)
    double epsilon = T00 - v*M;
    double rho = J0*sqrt(1 - v*v);
   
    double pressure = get_pressure(epsilon, rho);
    double fv = M/(T00 + pressure);
    return(fv);
}

double Advance::reconst_velocity_f_Newton(double v, double T00, double M,
                                          double J0) {
    double fv = v - reconst_velocity_f(v, T00, M, J0);
    return(fv);
}

double Advance::reconst_velocity_df(double v, double T00, double M,
                                    double J0) {
    // this function returns df'(v)/dv where f' = v - f(v)
    double epsilon = T00 - v*M;
    double temp = sqrt(1. - v*v);
    double rho = J0*temp;
    double temp2 = v/temp;
   
    double pressure = get_pressure(epsilon, rho);
    double dPde = p_e_func(epsilon, rho);
    double dPdrho = p_rho_func(epsilon, rho);
    
    double temp1 = T00 + pressure;

    double dfdv = 1. - M/(temp1*temp1)*(M*dPde + J0*temp2*dPdrho);
    return(dfdv);
}

double Advance::reconst_u0_f(double u0, double T00, double K00, double M,
                             double J0) {
    // this function returns f(u0) = (M0+P)/sqrt((M0+P)^2 - M^2)
    double epsilon = T00 - sqrt(1. - 1./u0/u0)*M;
    double rho = J0/u0;
    
    double pressure = get_pressure(epsilon, rho);
    double fu = (T00 + pressure)/sqrt((T00 + pressure)*(T00 + pressure) - K00);
    return(fu);
}

double Advance::reconst_u0_f_Newton(double u0, double T00, double K00,
                                    double M, double J0) {
    // this function returns f(u0) = u0 - (M0+P)/sqrt((M0+P)^2 - M^2)
    double fu = u0 - reconst_u0_f(u0, T00, K00, M, J0);
    return(fu);
}

double Advance::reconst_u0_df(double u0, double T00, double K00, double M,
                              double J0) {
    // this function returns df'/du0 where f'(u0) = u0 - f(u0)
    double v = sqrt(1. - 1./(u0*u0));
    double epsilon = T00 - v*M;
    double rho = J0/u0;
    double dedu0 = - M/(u0*u0*u0*v);
    double drhodu0 = - J0/(u0*u0);
    
    double pressure = get_pressure(epsilon, rho);
    double dPde = p_e_func(epsilon, rho);
    double dPdrho = p_rho_func(epsilon, rho);

    double denorm = pow(((T00 + pressure)*(T00 + pressure) - K00), 1.5);
    double dfdu0 = 1. + (dedu0*dPde + drhodu0*dPdrho)*K00/denorm;
    return(dfdu0);
}


int Advance::FirstRKStepW(double tau, int rk_flag, Field *hydro_fields,
                          int ieta, int ix, int iy) {
    double tau_rk = tau + rk_flag*DELTA_TAU;

    int idx = get_indx(ieta, ix, iy);
    double u0 = hydro_fields->u_rk1[0][idx];
    double u1 = hydro_fields->u_rk1[1][idx];
    double u2 = hydro_fields->u_rk1[2][idx];
    double u3 = hydro_fields->u_rk1[3][idx];

    // Solve partial_a (u^a W^{mu nu}) = 0
    // Update W^{mu nu}
    // mu = 4 is the baryon current qmu

    // calculate delta uWmunu
    // need to use u[0][mu], remember rk_flag = 0 here
    // with the KT flux 
    // solve partial_tau (u^0 W^{kl}) = -partial_i (u^i W^{kl}
 
    /* Advance uWmunu */
    // add partial_\mu uW^\mu\nu terms using KT
    Make_uWRHS(tau_rk, hydro_fields, ieta, ix, iy);

    // add source terms
    if (INCLUDE_SHEAR) {
        Make_uWSource(tau_rk, hydro_fields, ieta, ix, iy);
    }

    if (INCLUDE_BULK) {
        Make_uPiSource(tau_rk, hydro_fields, ieta, ix, iy);
    }
    
    if (rk_flag == 0) {
        for (int alpha = 0; alpha < 15; alpha++) {
            hydro_fields->Wmunu_rk1[alpha][idx] /= u0;
        }
    } else {
        for (int alpha = 0; alpha < 15; alpha++) {
            double rk0 = (hydro_fields->Wmunu_prev[alpha][idx]
                          *hydro_fields->u_prev[0][idx]);
            hydro_fields->Wmunu_rk1[alpha][idx] += rk0;
            hydro_fields->Wmunu_rk1[alpha][idx] *= 0.5;
            hydro_fields->Wmunu_rk1[alpha][idx] /= u0;
        }
    }

    // reconstruct other components
    // re-make Wmunu[3][3] so that Wmunu[mu][nu] is traceless
    hydro_fields->Wmunu_rk1[9][idx] = (
            (2.*(u1*u2*hydro_fields->Wmunu_rk1[5][idx]
                 + u1*u3*hydro_fields->Wmunu_rk1[6][idx]
                 + u2*u3*hydro_fields->Wmunu_rk1[8][idx])
                - (u0*u0 - u1*u1)*hydro_fields->Wmunu_rk1[4][idx]
                - (u0*u0 - u2*u2)*hydro_fields->Wmunu_rk1[7][idx])
            /(u0*u0 - u3*u3));

    // make Wmunu^0i using the transversality
    hydro_fields->Wmunu_rk1[1][idx] = (
        hydro_fields->Wmunu_rk1[4][idx]*u1
        + hydro_fields->Wmunu_rk1[5][idx]*u2
        + hydro_fields->Wmunu_rk1[6][idx]*u3)/u0;
    hydro_fields->Wmunu_rk1[2][idx] = (
        hydro_fields->Wmunu_rk1[5][idx]*u1
        + hydro_fields->Wmunu_rk1[7][idx]*u2
        + hydro_fields->Wmunu_rk1[8][idx]*u3)/u0;
    hydro_fields->Wmunu_rk1[3][idx] = (
        hydro_fields->Wmunu_rk1[6][idx]*u1
        + hydro_fields->Wmunu_rk1[8][idx]*u2
        + hydro_fields->Wmunu_rk1[9][idx]*u3)/u0;

    // make Wmunu^00
    hydro_fields->Wmunu_rk1[0][idx] = (
        hydro_fields->Wmunu_rk1[1][idx]*u1
        + hydro_fields->Wmunu_rk1[2][idx]*u2
        + hydro_fields->Wmunu_rk1[3][idx]*u3)/u0;

    // diffusion need to be added
    hydro_fields->Wmunu_rk1[10][idx] = 0.0;
    hydro_fields->Wmunu_rk1[11][idx] = 0.0;
    hydro_fields->Wmunu_rk1[12][idx] = 0.0;
    hydro_fields->Wmunu_rk1[13][idx] = 0.0;
    
    return(1);
}


void Advance::update_grid_array_from_field(
                Field *hydro_fields, int idx, double *grid_array) {
    grid_array[0] = hydro_fields->e_rk0[idx];
    grid_array[4] = hydro_fields->rhob_rk0[idx];
    for (int i = 1; i < 4; i++) {
        grid_array[i] = (hydro_fields->u_rk0[i][idx]
                         /hydro_fields->u_rk0[0][idx]);
    }
}


void Advance::update_grid_array_from_field_prev(
                Field *hydro_fields, int idx, double *grid_array) {
    grid_array[0] = hydro_fields->e_prev[idx];
    grid_array[4] = hydro_fields->rhob_prev[idx];
    for (int i = 1; i < 4; i++) {
        grid_array[i] = (hydro_fields->u_prev[i][idx]
                         /hydro_fields->u_prev[0][idx]);
    }
}


//! this function reduce the size of shear stress tensor and bulk pressure
//! in the dilute region to stablize numerical simulations
void Advance::QuestRevert(Field *hydro_fields, int idx) {
    //double eps_scale = 1.0;  // 1/fm^4
    //double factor = 300.*tanh(e_local/eps_scale);
    double factor = 300.*tanh(hydro_fields->e_rk1[idx]/1.0);

    double vis_size = (
        hydro_fields->Wmunu_rk1[0][idx]*hydro_fields->Wmunu_rk1[0][idx]
        + hydro_fields->Wmunu_rk1[4][idx]*hydro_fields->Wmunu_rk1[4][idx]
        + hydro_fields->Wmunu_rk1[7][idx]*hydro_fields->Wmunu_rk1[7][idx]
        + hydro_fields->Wmunu_rk1[9][idx]*hydro_fields->Wmunu_rk1[9][idx]
        - 2.*(hydro_fields->Wmunu_rk1[1][idx]*hydro_fields->Wmunu_rk1[1][idx]
              + hydro_fields->Wmunu_rk1[2][idx]*hydro_fields->Wmunu_rk1[2][idx]
              + hydro_fields->Wmunu_rk1[3][idx]*hydro_fields->Wmunu_rk1[3][idx])
        + 2.*(hydro_fields->Wmunu_rk1[5][idx]*hydro_fields->Wmunu_rk1[5][idx]
              + hydro_fields->Wmunu_rk1[6][idx]*hydro_fields->Wmunu_rk1[6][idx]
              + hydro_fields->Wmunu_rk1[8][idx]*hydro_fields->Wmunu_rk1[8][idx]));
  

    double p_local = get_pressure(hydro_fields->e_rk1[idx],
                                  hydro_fields->rhob_rk1[idx]);
    double eq_size = (hydro_fields->e_rk1[idx]*hydro_fields->e_rk1[idx]
                      + 3.*p_local*p_local);
       
    double rho_vis = sqrt(vis_size/eq_size)/factor; 
    // Reducing the shear stress tensor 
    if (rho_vis > RHO_MAX) {
        for (int mu = 0; mu < 10; mu++) {
            hydro_fields->Wmunu_rk1[mu][idx] *= (RHO_MAX/rho_vis);
        }
    }
   
    // bulk viscous pressure
    vis_size = (3.*hydro_fields->Wmunu_rk1[14][idx]
                  *hydro_fields->Wmunu_rk1[14][idx]);
    rho_vis  = sqrt(vis_size/eq_size)/factor;
    // Reducing bulk viscous pressure 
    if (rho_vis > RHO_MAX) {
        hydro_fields->Wmunu_rk1[14][idx] *= (RHO_MAX/rho_vis);
    }
}




//! It computes the spatial derivatives of T^\mu\nu using the KT algorithm
void Advance::MakeDeltaQI(double tau, double *grid_array,
                          Field *hydro_fields, int ieta, int ix, int iy) {
    /* \partial_tau (tau Ttautau) + \partial_eta Tetatau 
            + \partial_x (tau Txtau) + \partial_y (tau Tytau) + Tetaeta = 0 */
    /* \partial_tau (tau Ttaueta) + \partial_eta Teteta 
            + \partial_x (tau Txeta) + \partial_y (tau Txeta) + Tetatau = 0 */
    /* \partial_tau (tau Txtau) + \partial_eta Tetax + \partial_x tau T_xx
            + \partial_y tau Tyx = 0 */
    
    // tau*Tmu0

    double qiphL[5];
    double qiphR[5];
    double qimhL[5];
    double qimhR[5];
    
    double grid_array_hL[5];
    double grid_array_hR[5];
    
    int idx = get_indx(ieta, ix, iy);
    double FihL, FihR, Fih, aih;
    double gp, ghR, ghL;
    double minmod_temp;

    // implement Kurganov-Tadmor scheme
    // here computes the half way T^\tau\mu currents
    // x-direction
    int idx_p_1 = get_indx(ieta, MIN(ix + 1, GRID_SIZE_X), iy);
    int idx_m_1 = get_indx(ieta, MAX(ix - 1, 0), iy);
    int idx_p_2 = get_indx(ieta, MIN(ix + 2, GRID_SIZE_X), iy);
    int idx_m_2 = get_indx(ieta, MAX(ix - 2, 0), iy);
    for (int alpha = 0; alpha < 5; alpha++) {
        gp = hydro_fields->qi_array[alpha][idx];
        ghR = hydro_fields->qi_array[alpha][idx_p_1];
        ghL = hydro_fields->qi_array[alpha][idx_m_1];
        minmod_temp = 0.5*minmod_dx(ghR, gp, ghL);

        qiphL[alpha] = gp + minmod_temp;
        qiphR[alpha] = ghR - 0.5*minmod_dx(
                            hydro_fields->qi_array[alpha][idx_p_2], ghR, gp);
        qimhL[alpha] = ghL + 0.5*minmod_dx(
                            gp, ghL, hydro_fields->qi_array[alpha][idx_m_2]);
        qimhR[alpha] = gp - minmod_temp;
    }
    // for each direction, reconstruct half-way cells
    // reconstruct e, rhob, and u[4] for half way cells
    ReconstIt_velocity_Newton(
                    grid_array_hL, tau, qiphL, grid_array);
    ReconstIt_velocity_Newton(
                    grid_array_hR, tau, qiphR, grid_array);

    aih = MAX(MaxSpeed(tau, 1, grid_array_hL),
              MaxSpeed(tau, 1, grid_array_hR));
    for (int alpha = 0; alpha < 5; alpha++) {
        FihL = tau*get_TJb_new(grid_array_hL, alpha, 1);
        FihR = tau*get_TJb_new(grid_array_hR, alpha, 1);
        // KT: H_{j+1/2} = (f(u^+_{j+1/2}) + f(u^-_{j+1/2})/2
        //              - a_{j+1/2}(u_{j+1/2}^+ - u^-_{j+1/2})/2
        Fih = 0.5*((FihL + FihR)
                            - aih*(qiphR[alpha] - qiphL[alpha]));
        hydro_fields->qi_array_new[alpha][idx] = -Fih/DELTA_X*DELTA_TAU;
    }
    
    ReconstIt_velocity_Newton(grid_array_hL, tau,
                                         qimhL, grid_array);
    ReconstIt_velocity_Newton(grid_array_hR, tau,
                                         qimhR, grid_array);
    aih = MAX(MaxSpeed(tau, 1, grid_array_hL),
              MaxSpeed(tau, 1, grid_array_hR));

    for (int alpha = 0; alpha < 5; alpha++) {
        FihL = tau*get_TJb_new(grid_array_hL, alpha, 1);
        FihR = tau*get_TJb_new(grid_array_hR, alpha, 1);
        // KT: H_{j+1/2} = (f(u^+_{j+1/2}) + f(u^-_{j+1/2})/2
        //              - a_{j+1/2}(u_{j+1/2}^+ - u^-_{j+1/2})/2
        Fih = 0.5*((FihL + FihR)
                            - aih*(qimhR[alpha] - qimhL[alpha]));
        hydro_fields->qi_array_new[alpha][idx] += Fih/DELTA_X*DELTA_TAU;
    }
    //cout << "x-direction" << endl;
    
    // y-direction
    idx_p_1 = get_indx(ieta, ix, MIN(iy + 1, GRID_SIZE_Y));
    idx_m_1 = get_indx(ieta, ix, MAX(iy - 1, 0));
    idx_p_2 = get_indx(ieta, ix, MIN(iy + 2, GRID_SIZE_Y));
    idx_m_2 = get_indx(ieta, ix, MAX(iy - 2, 0));
    for (int alpha = 0; alpha < 5; alpha++) {
        gp =  hydro_fields->qi_array[alpha][idx];
        ghR = hydro_fields->qi_array[alpha][idx_p_1];
        ghL = hydro_fields->qi_array[alpha][idx_m_1];

        minmod_temp = 0.5*minmod_dx(ghR, gp, ghL);
        qiphL[alpha] = gp + minmod_temp;
        qiphR[alpha] = ghR - 0.5*minmod_dx(
                            hydro_fields->qi_array[alpha][idx_p_2], ghR, gp);
        qimhL[alpha] = ghL + 0.5*minmod_dx(
                            gp, ghL, hydro_fields->qi_array[alpha][idx_m_2]);
        qimhR[alpha] = gp - minmod_temp;
    }
    // for each direction, reconstruct half-way cells
    // reconstruct e, rhob, and u[4] for half way cells
    ReconstIt_velocity_Newton(
                    grid_array_hL, tau, qiphL, grid_array);
    ReconstIt_velocity_Newton(
                    grid_array_hR, tau, qiphR, grid_array);
    aih = MAX(MaxSpeed(tau, 2, grid_array_hL),
              MaxSpeed(tau, 2, grid_array_hR));
    for (int alpha = 0; alpha < 5; alpha++) {
        FihL = tau*get_TJb_new(grid_array_hL, alpha, 2);
        FihR = tau*get_TJb_new(grid_array_hR, alpha, 2);
        // KT: H_{j+1/2} = (f(u^+_{j+1/2}) + f(u^-_{j+1/2})/2
        //              - a_{j+1/2}(u_{j+1/2}^+ - u^-_{j+1/2})/2
        Fih = 0.5*((FihL + FihR)
                            - aih*(qiphR[alpha] - qiphL[alpha]));
        hydro_fields->qi_array_new[alpha][idx] -= Fih/DELTA_Y*DELTA_TAU;
    }

    ReconstIt_velocity_Newton(grid_array_hL, tau,
                                         qimhL, grid_array);
    ReconstIt_velocity_Newton(grid_array_hR, tau,
                                         qimhR, grid_array);
    aih = MAX(MaxSpeed(tau, 2, grid_array_hL),
              MaxSpeed(tau, 2, grid_array_hR));

    for (int alpha = 0; alpha < 5; alpha++) {
        FihL = tau*get_TJb_new(grid_array_hL, alpha, 2);
        FihR = tau*get_TJb_new(grid_array_hR, alpha, 2);
        // KT: H_{j+1/2} = (f(u^+_{j+1/2}) + f(u^-_{j+1/2})/2
        //              - a_{j+1/2}(u_{j+1/2}^+ - u^-_{j+1/2})/2
        Fih = 0.5*((FihL + FihR)
                            - aih*(qimhR[alpha] - qimhL[alpha]));
        hydro_fields->qi_array_new[alpha][idx] += Fih/DELTA_Y*DELTA_TAU;
    }
    //cout << "y-direction" << endl;
    
    // eta-direction
    idx_p_1 = get_indx(MIN(ieta + 1, GRID_SIZE_ETA - 1), ix, iy);
    idx_m_1 = get_indx(MAX(ieta - 1, 0), ix, iy);
    idx_p_2 = get_indx(MIN(ieta + 2, GRID_SIZE_ETA - 1), ix, iy);
    idx_m_2 = get_indx(MAX(ieta - 2, 0), ix, iy);
    for (int alpha = 0; alpha < 5; alpha++) {
        gp =  hydro_fields->qi_array[alpha][idx];
        ghR = hydro_fields->qi_array[alpha][idx_p_1];
        ghL = hydro_fields->qi_array[alpha][idx_m_1];

        minmod_temp = 0.5*minmod_dx(ghR, gp, ghL);
        qiphL[alpha] = gp + minmod_temp;
        qiphR[alpha] = ghR - 0.5*minmod_dx(
                            hydro_fields->qi_array[alpha][idx_p_2], ghR, gp);
        qimhL[alpha] = ghL + 0.5*minmod_dx(
                            gp, ghL, hydro_fields->qi_array[alpha][idx_m_2]);
        qimhR[alpha] = gp - minmod_temp;
    }
    // for each direction, reconstruct half-way cells
    // reconstruct e, rhob, and u[4] for half way cells
    ReconstIt_velocity_Newton(
                    grid_array_hL, tau, qiphL, grid_array);
    ReconstIt_velocity_Newton(
                    grid_array_hR, tau, qiphR, grid_array);
    aih = MAX(MaxSpeed(tau, 3, grid_array_hL),
              MaxSpeed(tau, 3, grid_array_hR));
    for (int alpha = 0; alpha < 5; alpha++) {
        FihL = 1.0*get_TJb_new(grid_array_hL, alpha, 3);
        FihR = 1.0*get_TJb_new(grid_array_hR, alpha, 3);
        // KT: H_{j+1/2} = (f(u^+_{j+1/2}) + f(u^-_{j+1/2})/2
        //              - a_{j+1/2}(u_{j+1/2}^+ - u^-_{j+1/2})/2
        Fih = 0.5*((FihL + FihR)
                            - aih*(qiphR[alpha] - qiphL[alpha]));

        hydro_fields->qi_array_new[alpha][idx] -= Fih/DELTA_ETA*DELTA_TAU;
    }

    ReconstIt_velocity_Newton(grid_array_hL, tau,
                                         qimhL, grid_array);
    ReconstIt_velocity_Newton(grid_array_hR, tau,
                                         qimhR, grid_array);
    aih = MAX(MaxSpeed(tau, 3, grid_array_hL),
              MaxSpeed(tau, 3, grid_array_hR));

    for (int alpha = 0; alpha < 5; alpha++) {
        FihL = 1.0*get_TJb_new(grid_array_hL, alpha, 3);
        FihR = 1.0*get_TJb_new(grid_array_hR, alpha, 3);
        // KT: H_{j+1/2} = (f(u^+_{j+1/2}) + f(u^-_{j+1/2})/2
        //              - a_{j+1/2}(u_{j+1/2}^+ - u^-_{j+1/2})/2
        Fih = 0.5*((FihL + FihR)
                            - aih*(qimhR[alpha] - qimhL[alpha]));
        hydro_fields->qi_array_new[alpha][idx] += Fih/DELTA_ETA*DELTA_TAU;
    }
    //cout << "eta-direction" << endl;

    // geometric terms
    hydro_fields->qi_array_new[0][idx] -= (get_TJb_new(grid_array, 3, 3)
                                           *DELTA_TAU);
    hydro_fields->qi_array_new[3][idx] -= (get_TJb_new(grid_array, 3, 0)
                                           *DELTA_TAU);
    
    for (int alpha = 0; alpha < 5; alpha++) {
        hydro_fields->qi_array_new[alpha][idx] += (
                                    hydro_fields->qi_array[alpha][idx]);
    }
}


/* Calculate the right-hand-side */
/*
 du/dt = (1/Delta x)(F_imh - F_iph) + D
 F_iph(a) = (1/2)(f_a(qiphR) + f_a(qiphL))- (1/2)aiph(qiphR - qiphL)
 F_imh(a) = (1/2)(f_a(qimhR) + f_a(qimhL))- (1/2)aimh(qimhR - qimhL)
 RK 2nd order Heun's rules:

 k1 = f(t, u);
 u1 = u + k1*h
 k2 = f(t+h, u1);
 u2 = u1 + k2*h = u + (k1+k2)*h;
 u_next = u + (1/2)(k1+k2)*h
        = (1/2)(u + u2);
*/

//! determine the maximum signal propagation speed at the given direction
double Advance::MaxSpeed(double tau, int direc, double *grid_array) {
    //grid_p = grid_p_h_L, grid_p_h_R, grid_m_h_L, grid_m_h_R
    //these are reconstructed by Reconst, grid_array = [e, v^i, rhob]
    //double utau = (grid_p->u[0][0]);
    double utau = 1./sqrt(1. - grid_array[1]*grid_array[1]
                             - grid_array[2]*grid_array[2]
                             - grid_array[3]*grid_array[3]);
    double utau2 = utau*utau;
    //double ux = fabs((grid_p->u[0][direc]));
    double ux = fabs(utau*grid_array[direc]);
    double ux2 = ux*ux;
    double ut2mux2 = utau2 - ux2;
  
    //double eps = grid_p->epsilon;
    //double rhob = grid_p->rhob;
    double eps = grid_array[0];
    double rhob = grid_array[4];
  
    double vs2 = get_cs2(eps, rhob);

    double den = utau2*(1. - vs2) + vs2;
    double num_temp_sqrt = (ut2mux2 - (ut2mux2 - 1.)*vs2)*vs2;
    double num;
    if (num_temp_sqrt >= 0) {
        num = utau*ux*(1. - vs2) + sqrt(num_temp_sqrt);
    } else {
        double dpde = p_e_func(eps, rhob);
        double p = get_pressure(eps, rhob);
        double h = p + eps;
        if (dpde < 0.001) {
            num = (sqrt(-(h*dpde*h*(dpde*(-1.0 + ut2mux2) - ut2mux2))) 
                   - h*(-1.0 + dpde)*utau*ux);
        } else {
            num = 0.0;
        }
    }
    
    double f = num/(den + 1e-15);
    if (f > 1.0) {
        f = 1.0;
    }
    if (f < ux/utau) {
        f = ux/utau;
    }

    if (direc == 3) {
        f /= tau;
    }

    return(f);
}



double Advance::get_TJb_new(double *grid_array, int mu, int nu) {
    double rhob = grid_array[4];
    double gamma = 1./sqrt(1. - grid_array[1]*grid_array[1]
                              - grid_array[2]*grid_array[2]
                              - grid_array[3]*grid_array[3]);
    double u_nu = gamma;
    if (nu > 0) {
        u_nu *= grid_array[nu];
    }

    if (mu == 4) {
        double J_nu = rhob*u_nu;
        return(J_nu);
    } else if (mu < 4) {
        double e = grid_array[0];
        double gfac = 0.0;
        double u_mu = gamma;
        if (mu == nu) {
            u_mu = u_nu;
            if (mu == 0) {
                gfac = -1.0;
            } else {
                gfac = 1.0;
            }
        } else {
            if (mu > 0) {
                u_mu *= grid_array[mu];
            }
        }
        double pressure = get_pressure(e, rhob);
        double T_munu = (e + pressure)*u_mu*u_nu + pressure*gfac;
        return(T_munu);
    } else {
        return(0.0);
    }
}


//! This function computes the vector [T^\tau\mu, J^\tau] from the
//! grid_array [e, v^i, rhob]
void Advance::get_qmu_from_grid_array(double tau, double qi[5],
                                      double grid_array[5]) {
    double rhob = grid_array[4];
    double e = grid_array[0];
    double pressure = get_pressure(e, rhob);
    double gamma = 1./sqrt(1. - grid_array[1]*grid_array[1]
                              - grid_array[2]*grid_array[2]
                              - grid_array[3]*grid_array[3]);
    double gamma_sq = gamma*gamma;
    qi[0] = tau*((e + pressure)*gamma_sq - pressure);
    qi[1] = tau*(e + pressure)*gamma_sq*grid_array[1];
    qi[2] = tau*(e + pressure)*gamma_sq*grid_array[2];
    qi[3] = tau*(e + pressure)*gamma_sq*grid_array[3];
    qi[4] = tau*rhob*gamma;
}

void Advance::update_grid_cell(double *grid_array, Field *hydro_fields,
                               int ieta, int ix, int iy, double tau_next) {
    int field_idx = get_indx(ieta, ix, iy);
    update_grid_array_from_field(hydro_fields, field_idx, grid_array);
    double qi_array[5];
    for (int alpha = 0; alpha < 5; alpha++) {
        qi_array[alpha] = hydro_fields->qi_array_new[alpha][field_idx];
    }

    ReconstIt_velocity_Newton(grid_array, tau_next, qi_array, grid_array);

    double gamma = 1./sqrt(1. - grid_array[1]*grid_array[1]
                              - grid_array[2]*grid_array[2]
                              - grid_array[3]*grid_array[3]);
    hydro_fields->e_rk1[field_idx] = grid_array[0];
    hydro_fields->rhob_rk1[field_idx] = grid_array[4];
    hydro_fields->u_rk1[0][field_idx] = gamma;
    for (int i = 1; i < 4; i++) {
        hydro_fields->u_rk1[i][field_idx] = gamma*grid_array[i];
    }
}           


//! This function reverts the grid information back its values
//! at the previous time step
void Advance::revert_grid(double *grid_array, double *grid_prev) {
    for (int i = 0; i < 5; i++) {
        grid_array[i] = grid_prev[i];
    }
}

double Advance::get_pressure(double e_local, double rhob) {
    double num = e_local;
    double denorm = 3.;

#if EOS_TYPE == 1
    num = - 0.25181736420168666;
    denorm = 45829.44617893836;
    double en = e_local;
    num += 9737.845799644809*en;
    denorm += 4.0574329080826794e6*en;
    en *= e_local;
    num += 1.077580993288114e6*en;
    denorm += 2.0931169138134286e7*en;
    en *= e_local;
    num += 3.1729694865420084e6*en;
    denorm += 1.3512402226067686e7*en;
    en *= e_local;
    num += 1.6357487344679043e6*en;
    denorm += 1.7851642641834426e6*en;
    en *= e_local;
    num += 334334.4309240126*en;
    denorm += 278581.2989342773*en;
    en *= e_local;
    num += 41913.439282708554*en;
    denorm += 26452.34905933697*en;
    en *= e_local;
    num += 6340.448389300905*en;
    denorm += 499.04919730607065*en;
    en *= e_local;
    num += 141.5073484468774*en;
    denorm += 2.3405487982094204*en;
    en *= e_local;
    num += 0.7158279081255019*en;
    denorm += 0.002962497695527404*en;
    en *= e_local;
    num += 0.0009417586777847889*en;
    denorm += 9.601103399348206e-7*en;
    en *= e_local;
    num += 3.1188455176941583e-7*en;
    denorm += 5.928138360995685e-11*en;
    en *= e_local;
    num += 1.9531729608963267e-11*en;
    denorm += 3.2581066229887368e-18*en;
    if (num < 1e-16) {
        num = 1e-16;
    }
#endif
    return(num/denorm);
}

double Advance::get_cs2(double e_local, double rhob) {
    double cs2 = 1./3.;

#if EOS_TYPE == 1
    double num = 5.191934309650155e-32;
    double denorm = 1.4637868900982493e-30;
    double en = e_local;
    num += 4.123605749683891e-23*en;
    denorm += 6.716598285341542e-22*en;
    en *= e_local;
    num += 3.1955868410879504e-16*en;
    denorm += 3.5477700458515908e-15*en;
    en *= e_local;
    num += 1.4170364808063119e-10*en;
    denorm += 1.1225580509306008e-9*en;
    en *= e_local;
    num += 6.087136671592452e-6*en;
    denorm += 0.00003551782901018317*en;
    en *= e_local;
    num += 0.02969737949090831*en;
    denorm += 0.13653226327408863*en;
    en *= e_local;
    num += 15.382615282179595*en;
    denorm += 60.85769171450653*en;
    en *= e_local;
    num += 460.6487249985994*en;
    denorm += 1800.5461219450308*en;
    en *= e_local;
    num += 1612.4245252438795*en;
    denorm += 15190.225535036281*en;
    en *= e_local;
    num += 275.0492627924299*en;
    denorm += 590.2572000057821*en;
    en *= e_local;
    num += 58.60283714484669*en;
    denorm += 293.99144775704605*en;
    en *= e_local;
    num += 6.504847576502024*en;
    denorm += 21.461303090563028*en;
    en *= e_local;
    num += 0.03009027913262399*en;
    denorm += 0.09301685073435291*en;
    en *= e_local;
    num += 8.189430244031285e-6*en;
    denorm += 0.000024810902623582917*en;
    cs2 = num/denorm;

	//cs2 = ((5.191934309650155e-32 + 4.123605749683891e-23*e1
    //        + 3.1955868410879504e-16*e2 + 1.4170364808063119e-10*e3
    //        + 6.087136671592452e-6*e4 + 0.02969737949090831*e5
    //        + 15.382615282179595*e6 + 460.6487249985994*e7
    //        + 1612.4245252438795*e8 + 275.0492627924299*e9
    //        + 58.60283714484669*e10 + 6.504847576502024*e11
    //        + 0.03009027913262399*e12 + 8.189430244031285e-6*e13)
	//	   /(1.4637868900982493e-30 + 6.716598285341542e-22*e1
    //         + 3.5477700458515908e-15*e2 + 1.1225580509306008e-9*e3
    //         + 0.00003551782901018317*e4 + 0.13653226327408863*e5
    //         + 60.85769171450653*e6 + 1800.5461219450308*e7
    //         + 15190.225535036281*e8 + 590.2572000057821*e9
    //         + 293.99144775704605*e10 + 21.461303090563028*e11
    //         + 0.09301685073435291*e12 + 0.000024810902623582917*e13));
#endif
    return(cs2);
}

double Advance::p_e_func(double e_local, double rhob) {
    double dPde = get_cs2(e_local, rhob);
    return(dPde);
}

double Advance::p_rho_func(double e_local, double rhob) {
    double dPdrho = 0.0;
    return(dPdrho);
}

double Advance::get_mu(double e_local, double rhob) {
    double mu = 0.0;
    return(mu);
}

double Advance::get_temperature(double e_local, double rhob) {
    double num = 90.0/M_PI/M_PI*(e_local/3.0)/(2*(9.-1)+7./2*3.*2.5);
    double denorm = 1.;
    double temperature = pow(num, 0.25);

#if EOS_TYPE == 1
    num = 1.510073201405604e-29;
    denorm = 7.558667139355393e-28;
    double en = e_local;
    num += 8.014062800678687e-18*en;
    denorm += 1.3686372302041508e-16*en;
    en *= e_local;
    num += 2.4954778310451065e-10*en;
    denorm += 2.998130743142826e-9*en;
    en *= e_local;
    num += 0.000063810382643387*en;
    denorm += 0.0005036835870305458*en;
    en *= e_local;
    num += 0.4873490574161924*en;
    denorm += 2.316902328874072*en;
    en *= e_local;
    num += 207.48582344326206*en;
    denorm += 578.0778724946719*en;
    en *= e_local;
    num += 6686.07424325115*en;
    denorm += 11179.193315394154*en;
    en *= e_local;
    num += 14109.766109389702*en;
    denorm += 17965.67607192861*en;
    en *= e_local;
    num += 1471.6180520527757*en;
    denorm += 1051.0730543534657*en;
    en *= e_local;
    num += 14.055788949565482*en;
    denorm += 5.916312075925817*en;
    en *= e_local;
    num += 0.015421252394182246*en;
    denorm += 0.003778342768228011*en;
    en *= e_local;
    num += 1.5780479034557783e-6*en;
    denorm += 1.8472801679382593e-7*en;
    temperature = num/denorm;

	//temperature = ((1.510073201405604e-29 + 8.014062800678687e-18*e1
    //                + 2.4954778310451065e-10*e2 + 0.000063810382643387*e3
    //                + 0.4873490574161924*e4 + 207.48582344326206*e5
    //                + 6686.07424325115*e6 + 14109.766109389702*e7
    //                + 1471.6180520527757*e8 + 14.055788949565482*e9
    //                + 0.015421252394182246*e10 + 1.5780479034557783e-6*e11)
    //               /(7.558667139355393e-28 + 1.3686372302041508e-16*e1
    //                 + 2.998130743142826e-9*e2 + 0.0005036835870305458*e3
    //                 + 2.316902328874072*e4 + 578.0778724946719*e5
    //                 + 11179.193315394154*e6 + 17965.67607192861*e7
    //                 + 1051.0730543534657*e8 + 5.916312075925817*e9
    //                 + 0.003778342768228011*e10 + 1.8472801679382593e-7*e11));
#endif
    return(temperature);
}

double Advance::minmod_dx(double up1, double u, double um1) {
    if ((up1 > u) && (u > um1)) {
        return MIN((up1 - u)*THETA_FLUX,
                   MIN((u - um1)*THETA_FLUX, (up1 - um1)*0.5));
    } else if ((up1 < u) && (u < um1)) {
        return MAX((up1 - u)*THETA_FLUX,
                   MAX((u - um1)*THETA_FLUX, (up1 - um1)*0.5));
    } else {
      return 0.0;
    }
}/* minmod_dx */


//! calculate d_m (tau W^{m,alpha}) + (geom source terms)
//! partial_tau W^tau alpha
//! this is partial_tau evaluated at tau
//! this is the first step. so rk_flag = 0
//! change: alpha first which is the case
//!         for everywhere else. also, this change is necessary
//!         to use Wmunu[rk_flag][4][mu] as the dissipative baryon current
void Advance::MakeWSource(double tau, Field *hydro_fields,
                          int ieta, int ix, int iy) {
    int field_idx = get_indx(ieta, ix, iy);
    for (int alpha = 0; alpha < 4 + INCLUDE_DIFF; alpha++) {
        int idx_1d, field_idx_p_1, field_idx_m_1;
        double sg, sgp1, sgm1;
        double dWdxmu = 0.0;
        double dPidxmu = 0.0;
        // shear part
        // dW/dtau
        // backward time derivative (first order is more stable)
        dWdxmu = ((hydro_fields->Wmunu_rk0[alpha][field_idx]
                    - hydro_fields->Wmunu_prev[alpha][field_idx])
                  /DELTA_TAU);

        // x-direction
        idx_1d = map_2d_idx_to_1d(alpha, 1);
        field_idx_p_1 = get_indx(ieta, MIN(ix + 1, GRID_SIZE_X), iy);
        field_idx_m_1 = get_indx(ieta, MAX(ix - 1, 0), iy);
        sg = hydro_fields->Wmunu_rk0[idx_1d][field_idx];
        sgp1 = hydro_fields->Wmunu_rk0[idx_1d][field_idx_p_1];
        sgm1 = hydro_fields->Wmunu_rk0[idx_1d][field_idx_m_1];
        //dWdxmu += (sgp1 - sgm1)/(2.*DELTA_X);
        dWdxmu += minmod_dx(sgp1, sg, sgm1)/DELTA_X;

        // y-direction
        idx_1d = map_2d_idx_to_1d(alpha, 2);
        field_idx_p_1 = get_indx(ieta, ix, MIN(iy + 1, GRID_SIZE_Y));
        field_idx_m_1 = get_indx(ieta, ix, MAX(iy - 1, 0));
        sg = hydro_fields->Wmunu_rk0[idx_1d][field_idx];
        sgp1 = hydro_fields->Wmunu_rk0[idx_1d][field_idx_p_1];
        sgm1 = hydro_fields->Wmunu_rk0[idx_1d][field_idx_m_1];
        //dWdxmu += (sgp1 - sgm1)/(2.*DELTA_Y);
        dWdxmu += minmod_dx(sgp1, sg, sgm1)/DELTA_Y;

        // eta-direction
        idx_1d = map_2d_idx_to_1d(alpha, 3);
        field_idx_p_1 = get_indx(MIN(ieta + 1, GRID_SIZE_ETA - 1), ix, iy);
        field_idx_m_1 = get_indx(MAX(ieta - 1, 0), ix, iy);
        sg = hydro_fields->Wmunu_rk0[idx_1d][field_idx];
        sgp1 = hydro_fields->Wmunu_rk0[idx_1d][field_idx_p_1];
        sgm1 = hydro_fields->Wmunu_rk0[idx_1d][field_idx_m_1];
        //dWdeta = (sgp1 - sgm1)/(2.*DELTA_ETA*tau);
        dWdxmu += minmod_dx(sgp1, sg, sgm1)/(DELTA_ETA*tau);
        
        // partial_m (tau W^mn) = W^0n + tau partial_m W^mn
        //double sf = (tau*(dWdtau + dWdx_perp + dWdeta)
        //             + hydro_fields->Wmunu_rk0[alpha][field_idx]);
        dWdxmu = tau*dWdxmu + hydro_fields->Wmunu_rk0[alpha][field_idx];

        // bulk pressure term
        if (alpha < 4 && INCLUDE_BULK) {
            double Pi_alpha0 = 0.0;
            // dPi/dtau
            // backward time derivative (first order is more stable)
            double gfac = (alpha == 0 ? -1.0 : 0.0);
            //Pi_alpha0 = (vis_array[14]
            //             *(gfac + vis_array[15+alpha]
            //                      *vis_array[15]));
            Pi_alpha0 = (hydro_fields->Wmunu_rk0[14][field_idx]
                         *(gfac + hydro_fields->u_rk0[alpha][field_idx]
                                  *hydro_fields->u_rk0[0][field_idx]));

            dPidxmu = (
                (Pi_alpha0 - hydro_fields->Wmunu_prev[14][field_idx]
                             *(gfac + hydro_fields->u_prev[alpha][field_idx]
                               *hydro_fields->u_prev[0][field_idx]))/DELTA_TAU);
            // x-direction
            gfac = (alpha == 1 ? 1.0 : 0.0);
            field_idx_p_1 = get_indx(ieta, MIN(ix + 1, GRID_SIZE_X), iy);
            field_idx_m_1 = get_indx(ieta, MAX(ix - 1, 0), iy);
            sg = (hydro_fields->Wmunu_rk0[14][field_idx]
                  *(gfac + hydro_fields->u_rk0[alpha][field_idx]
                            *hydro_fields->u_rk0[1][field_idx]));
            sgp1 = (hydro_fields->Wmunu_rk0[14][field_idx_p_1]
                    *(gfac + hydro_fields->u_rk0[alpha][field_idx_p_1]
                              *hydro_fields->u_rk0[1][field_idx_p_1]));
            sgm1 = (hydro_fields->Wmunu_rk0[14][field_idx_m_1]
                    *(gfac + hydro_fields->u_rk0[alpha][field_idx_m_1]
                              *hydro_fields->u_rk0[1][field_idx_m_1]));
            //dPidxmu += (sgp1 - sgm1)/(2.*DELTA_X);
            dPidxmu += minmod_dx(sgp1, sg, sgm1)/DELTA_X;
            
            // y-direction
            gfac = (alpha == 2 ? 1.0 : 0.0);
            field_idx_p_1 = get_indx(ieta, ix, MIN(iy + 1, GRID_SIZE_Y));
            field_idx_m_1 = get_indx(ieta, ix, MAX(iy - 1, 0));
            sg = (hydro_fields->Wmunu_rk0[14][field_idx]
                  *(gfac + hydro_fields->u_rk0[alpha][field_idx]
                            *hydro_fields->u_rk0[2][field_idx]));
            sgp1 = (hydro_fields->Wmunu_rk0[14][field_idx_p_1]
                    *(gfac + hydro_fields->u_rk0[alpha][field_idx_p_1]
                              *hydro_fields->u_rk0[2][field_idx_p_1]));
            sgm1 = (hydro_fields->Wmunu_rk0[14][field_idx_m_1]
                    *(gfac + hydro_fields->u_rk0[alpha][field_idx_m_1]
                              *hydro_fields->u_rk0[2][field_idx_m_1]));
            //dPidxmu += (sgp1 - sgm1)/(2.*DELTA_Y);
            dPidxmu += minmod_dx(sgp1, sg, sgm1)/DELTA_Y;

            // eta-direction
            gfac = (alpha == 3 ? 1.0 : 0.0);
            field_idx_p_1 = get_indx(MIN(ieta + 1, GRID_SIZE_ETA - 1), ix, iy);
            field_idx_m_1 = get_indx(MAX(ieta - 1, 0), ix, iy);
            sg = (hydro_fields->Wmunu_rk0[14][field_idx]
                  *(gfac + hydro_fields->u_rk0[alpha][field_idx]
                            *hydro_fields->u_rk0[3][field_idx]));
            sgp1 = (hydro_fields->Wmunu_rk0[14][field_idx_p_1]
                    *(gfac + hydro_fields->u_rk0[alpha][field_idx_p_1]
                              *hydro_fields->u_rk0[3][field_idx_p_1]));
            sgm1 = (hydro_fields->Wmunu_rk0[14][field_idx_m_1]
                    *(gfac + hydro_fields->u_rk0[alpha][field_idx_m_1]
                              *hydro_fields->u_rk0[3][field_idx_m_1]));
            //dPidxmu = ((sgp1 - sgm1)/(2.*DELTA_ETA*tau));
            dPidxmu += minmod_dx(sgp1, sg, sgm1)/(DELTA_ETA*tau);
            
            // partial_m (tau W^mn) = W^0n + tau partial_m W^mn
            dPidxmu = tau*dPidxmu + Pi_alpha0;
        }

        if (alpha < 4) {
            hydro_fields->qi_array_new[alpha][field_idx] -= (
                (dWdxmu*INCLUDE_SHEAR + dPidxmu*INCLUDE_BULK)*DELTA_TAU);
        } else if (alpha == 4) {
            hydro_fields->qi_array_new[alpha][field_idx] -= (
                (dWdxmu*INCLUDE_DIFF)*DELTA_TAU);
        }
    }
    
    // sources due to coordinate transform
    // for tau*T^tautau
    hydro_fields->qi_array_new[0][field_idx] -= (
        hydro_fields->Wmunu_rk0[9][field_idx]*INCLUDE_SHEAR
        + (hydro_fields->Wmunu_rk0[14][field_idx]
           *(1.0 + hydro_fields->u_rk0[3][field_idx]
                   *hydro_fields->u_rk0[3][field_idx]))*INCLUDE_BULK)*DELTA_TAU;
    // for tau*T^taueta
    hydro_fields->qi_array_new[3][field_idx] -= (
        hydro_fields->Wmunu_rk0[3][field_idx]*INCLUDE_SHEAR
        + (hydro_fields->Wmunu_rk0[14][field_idx]
           *(hydro_fields->u_rk0[0][field_idx]
             *hydro_fields->u_rk0[3][field_idx]))*INCLUDE_BULK)*DELTA_TAU;
}

int Advance::Make_uWRHS(double tau, Field* hydro_fields,
                        int ieta, int ix, int iy) {

    int idx = get_indx(ieta, ix, iy);

    // Kurganov-Tadmor for Wmunu */
    // implement 
    // partial_tau (utau Wmn) + (1/tau)partial_eta (ueta Wmn) 
    // + partial_x (ux Wmn) + partial_y (uy Wmn) + utau Wmn/tau
    // = SW 
    // or the right hand side of,
    // partial_tau (utau Wmn) = 
    //                  - (1/tau)partial_eta (ueta Wmn)
    //                  - partial_x (ux Wmn) - partial_y (uy Wmn) 
    //                  - utau Wmn/tau + SW*/
    
    // the local velocity is just u_x/u_tau, u_y/u_tau,
    //                            u_eta/tau/u_tau
    // KT flux is given by 
    // H_{j+1/2} = (fRph + fLph)/2 - ax(uRph - uLph) 
    // Here fRph = ux WmnRph and ax uRph = |ux/utau|_max utau Wmn
    // This is the second step in the operator splitting. it uses
    // rk_flag+1 as initial condition
    
    double u0 = hydro_fields->u_rk0[0][idx];
    double u1 = hydro_fields->u_rk0[1][idx];
    double u2 = hydro_fields->u_rk0[2][idx];
    double u3 = hydro_fields->u_rk0[3][idx];

    for (unsigned int idx_1d = 4; idx_1d < 15; idx_1d++) {
        double taufactor;
        double g, gp1, gm1, gp2, gm2, a, am1, ap1, ax;
        double f, fp1, fm1, fp2, fm2;
        double uWphR, uWphL, uWmhR, uWmhL, WphR, WphL, WmhR, WmhL;
        double HWph, HWmh, HW;
        int idx_p_2, idx_p_1, idx_m_1, idx_m_2;
        double sum;

        // the derivative part is the same for all viscous
        // components
        hydro_fields->Wmunu_rk1[idx_1d][idx] = (
                                    hydro_fields->Wmunu_rk0[idx_1d][idx]*u0);

        sum = 0.0;
        // x-direction
        taufactor = 1.0;
        /* Get_uWmns */
        g = hydro_fields->Wmunu_rk0[idx_1d][idx]*u0;
        f = hydro_fields->Wmunu_rk0[idx_1d][idx]*u1;
        a = fabs(u1)/u0;

        idx_p_2 = get_indx(ieta, MIN(ix + 2, GRID_SIZE_X), iy);
        gp2 = hydro_fields->Wmunu_rk0[idx_1d][idx_p_2];
        fp2 = gp2*hydro_fields->u_rk0[1][idx_p_2];
        gp2 *= hydro_fields->u_rk0[0][idx_p_2];
        
        idx_p_1 = get_indx(ieta, MIN(ix + 1, GRID_SIZE_X), iy);
        gp1 = hydro_fields->Wmunu_rk0[idx_1d][idx_p_1];
        fp1 = gp1*hydro_fields->u_rk0[1][idx_p_1];
        gp1 *= hydro_fields->u_rk0[0][idx_p_1];
        ap1 = fabs(hydro_fields->u_rk0[1][idx_p_1]
                   /hydro_fields->u_rk0[0][idx_p_1]);
        
        idx_m_1 = get_indx(ieta, MAX(ix - 1, 0), iy);
        gm1 = hydro_fields->Wmunu_rk0[idx_1d][idx_m_1];
        fm1 = gm1*hydro_fields->u_rk0[1][idx_m_1];
        gm1 *= hydro_fields->u_rk0[0][idx_m_1];
        am1 = fabs(hydro_fields->u_rk0[1][idx_m_1]
                   /hydro_fields->u_rk0[0][idx_m_1]);
        
        idx_m_2 = get_indx(ieta, MAX(ix - 2, 0), iy);
        gm2 = hydro_fields->Wmunu_rk0[idx_1d][idx_m_2];
        fm2 = gm2*hydro_fields->u_rk0[1][idx_m_2];
        gm2 *= hydro_fields->u_rk0[0][idx_m_2];

        // MakeuWmnHalfs uWmn
        uWphR = fp1 - 0.5*minmod_dx(fp1, f, fm1);
        double ddx = minmod_dx(fp1, f, fm1);
        uWphL = f + 0.5*ddx;
        uWmhR = f - 0.5*ddx;
        uWmhL = fm1 + 0.5*minmod_dx(f, fm1, fm2);
        // just Wmn
        WphR = gp1 - 0.5*minmod_dx(gp2, gp1, g);
        ddx = minmod_dx(gp1, g, gm1);
        WphL = g + 0.5*ddx;
        WmhR = g - 0.5*ddx;
        WmhL = gm1 + 0.5*minmod_dx(g, gm1, gm2);

        ax = maxi(a, ap1);
        HWph = ((uWphR + uWphL) - ax*(WphR - WphL))*0.5;
        ax = maxi(a, am1);
        HWmh = ((uWmhR + uWmhL) - ax*(WmhR - WmhL))*0.5;
        HW = (HWph - HWmh)/DELTA_X/taufactor;
            
        // make partial_i (u^i Wmn)
        sum += -HW;
    
        // y-direction
        taufactor = 1.0;
        /* Get_uWmns */
        g = hydro_fields->Wmunu_rk0[idx_1d][idx]*u0;
        f = hydro_fields->Wmunu_rk0[idx_1d][idx]*u2;
        a = fabs(u2)/u0;

        idx_p_2 = get_indx(ieta, ix, MIN(iy + 2, GRID_SIZE_Y));
        gp2 = hydro_fields->Wmunu_rk0[idx_1d][idx_p_2];
        fp2 = gp2*hydro_fields->u_rk0[2][idx_p_2];
        gp2 *= hydro_fields->u_rk0[0][idx_p_2];
        
        idx_p_1 = get_indx(ieta, ix, MIN(iy + 1, GRID_SIZE_Y));
        gp1 = hydro_fields->Wmunu_rk0[idx_1d][idx_p_1];
        fp1 = gp1*hydro_fields->u_rk0[2][idx_p_1];
        gp1 *= hydro_fields->u_rk0[0][idx_p_1];
        ap1 = fabs(hydro_fields->u_rk0[2][idx_p_1]
                   /hydro_fields->u_rk0[0][idx_p_1]);
        
        idx_m_1 = get_indx(ieta, ix, MAX(iy - 1, 0));
        gm1 = hydro_fields->Wmunu_rk0[idx_1d][idx_m_1];
        fm1 = gm1*hydro_fields->u_rk0[2][idx_m_1];
        gm1 *= hydro_fields->u_rk0[0][idx_m_1];
        am1 = fabs(hydro_fields->u_rk0[2][idx_m_1]
                   /hydro_fields->u_rk0[0][idx_m_1]);
        
        idx_m_2 = get_indx(ieta, ix, MAX(iy - 2, 0));
        gm2 = hydro_fields->Wmunu_rk0[idx_1d][idx_m_2];
        fm2 = gm2*hydro_fields->u_rk0[2][idx_m_2];
        gm2 *= hydro_fields->u_rk0[0][idx_m_2];

        // MakeuWmnHalfs uWmn
        uWphR = fp1 - 0.5*minmod_dx(fp2, fp1, f);
        ddx = minmod_dx(fp1, f, fm1);
        uWphL = f + 0.5*ddx;
        uWmhR = f - 0.5*ddx;
        uWmhL = fm1 + 0.5*minmod_dx(f, fm1, fm2);
        // just Wmn
        WphR = gp1 - 0.5*minmod_dx(gp2, gp1, g);
        ddx = minmod_dx(gp1, g, gm1);
        WphL = g + 0.5*ddx;
        WmhR = g - 0.5*ddx;
        WmhL = gm1 + 0.5*minmod_dx(g, gm1, gm2);
        ax = maxi(a, ap1);
        HWph = ((uWphR + uWphL) - ax*(WphR - WphL))*0.5;
        ax = maxi(a, am1);
        HWmh = ((uWmhR + uWmhL) - ax*(WmhR - WmhL))*0.5;
        HW = (HWph - HWmh)/DELTA_Y/taufactor;
        // make partial_i (u^i Wmn)
        sum += -HW;
    
        // eta-direction
        taufactor = tau;
        /* Get_uWmns */
        g = hydro_fields->Wmunu_rk0[idx_1d][idx]*u0;
        f = hydro_fields->Wmunu_rk0[idx_1d][idx]*u3;
        a = fabs(u3)/u0;

        idx_p_2 = get_indx(MIN(ieta + 2, GRID_SIZE_ETA - 1), ix, iy);
        gp2 = hydro_fields->Wmunu_rk0[idx_1d][idx_p_2];
        fp2 = gp2*hydro_fields->u_rk0[3][idx_p_2];
        gp2 *= hydro_fields->u_rk0[0][idx_p_2];
        
        idx_p_1 = get_indx(MIN(ieta + 1, GRID_SIZE_ETA - 1), ix, iy);
        gp1 = hydro_fields->Wmunu_rk0[idx_1d][idx_p_1];
        fp1 = gp1*hydro_fields->u_rk0[3][idx_p_1];
        gp1 *= hydro_fields->u_rk0[0][idx_p_1];
        ap1 = fabs(hydro_fields->u_rk0[3][idx_p_1]
                   /hydro_fields->u_rk0[0][idx_p_1]);
        
        idx_m_1 = get_indx(MAX(ieta - 1, 0), ix, iy);
        gm1 = hydro_fields->Wmunu_rk0[idx_1d][idx_m_1];
        fm1 = gm1*hydro_fields->u_rk0[3][idx_m_1];
        gm1 *= hydro_fields->u_rk0[0][idx_m_1];
        am1 = fabs(hydro_fields->u_rk0[3][idx_m_1]
                   /hydro_fields->u_rk0[0][idx_m_1]);
        
        idx_m_2 = get_indx(MAX(ieta - 2, 0), ix, iy);
        gm2 = hydro_fields->Wmunu_rk0[idx_1d][idx_m_2];
        fm2 = gm2*hydro_fields->u_rk0[3][idx_m_2];
        gm2 *= hydro_fields->u_rk0[0][idx_m_2];

        // MakeuWmnHalfs uWmn
        uWphR = fp1 - 0.5*minmod_dx(fp2, fp1, f);
        ddx = minmod_dx(fp1, f, fm1);
        uWphL = f + 0.5*ddx;
        uWmhR = f - 0.5*ddx;
        uWmhL = fm1 + 0.5*minmod_dx(f, fm1, fm2);
        // just Wmn
        WphR = gp1 - 0.5*minmod_dx(gp2, gp1, g);
        ddx = minmod_dx(gp1, g, gm1);
        WphL = g + 0.5*ddx;
        WmhR = g - 0.5*ddx;
        WmhL = gm1 + 0.5*minmod_dx(g, gm1, gm2);
        ax = maxi(a, ap1);
        HWph = ((uWphR + uWphL) - ax*(WphR - WphL))*0.5;
        ax = maxi(a, am1);
        HWmh = ((uWmhR + uWmhL) - ax*(WmhR - WmhL))*0.5;
        HW = (HWph - HWmh)/DELTA_ETA/taufactor;
        // make partial_i (u^i Wmn)
        sum += -HW;
        
        hydro_fields->Wmunu_rk1[idx_1d][idx] += sum*(DELTA_TAU);
    }

    // the following geometric parts are different for
    // individual pi^\mu\nu, and Pi
    // geometric terms for shear pi^\mu\nu
    // add a source term -u^tau Wmn/tau
    //   due to the coordinate change to tau-eta

    // this is from udW = d(uW) - Wdu = RHS
    // or d(uW) = udW + Wdu
    // this term is being added to the rhs so that
    // -4/3 + 1 = -1/3
    // other source terms due to the coordinate
    // change to tau-eta
    //tempf = (
    //    - (DATA_ptr->gmunu[3][mu])*(Wmunu_local[0][nu])
    //    - (DATA_ptr->gmunu[3][nu])*(Wmunu_local[0][mu])
    //    + (DATA_ptr->gmunu[0][mu])*(Wmunu_local[3][nu])
    //    + (DATA_ptr->gmunu[0][nu])*(Wmunu_local[3][mu])
    //    + (Wmunu_local[3][nu])
    //      //*(grid_pt->u[rk_flag][mu])*(grid_pt->u[rk_flag][0])
    //      *(vis_array[idx][15+mu])*(vis_array[idx][15])
    //    + (Wmunu_local[3][mu])
    //      //*(grid_pt->u[rk_flag][nu])*(grid_pt->u[rk_flag][0])
    //      *(vis_array[idx][15+nu])*(vis_array[idx][15])
    //    - (Wmunu_local[0][nu])
    //      //*(grid_pt->u[rk_flag][mu])*(grid_pt->u[rk_flag][3])
    //      *(vis_array[idx][15+mu])*(vis_array[idx][18])
    //    - (Wmunu_local[0][mu])
    //      //*(grid_pt->u[rk_flag][nu])*(grid_pt->u[rk_flag][3]))
    //      //*(grid_pt->u[rk_flag][3]/tau);
    //      *(vis_array[idx][15+nu])*(vis_array[idx][18]))
    //      *(vis_array[idx][18]/tau);
    //for (int ic = 0; ic < 4; ic++) {
    //    double ic_fac = (ic == 0 ? -1.0 : 1.0);
    //    tempf += (
    //          (Wmunu_local[ic][nu])*(vis_array[idx][15+mu])
    //           *(velocity_array[idx][1+ic])*ic_fac
    //        + (Wmunu_local[ic][mu])*(vis_array[idx][15+nu])
    //           *(velocity_array[idx][1+ic])*ic_fac);
    //}
    double tempf = 0.0;
    // W^11
    double sum = (- (u0*hydro_fields->Wmunu_rk0[4][idx])/tau
                  + (hydro_fields->expansion_rate[idx]
                      *hydro_fields->Wmunu_rk0[4][idx]));
    tempf = ((u3/tau)*2.*u1*(hydro_fields->Wmunu_rk0[6][idx]*u0
                             - hydro_fields->Wmunu_rk0[1][idx]*u3));
    tempf += 2.*(
        - hydro_fields->Du_mu[0][idx]*(hydro_fields->Wmunu_rk0[1][idx]*u1)
        + hydro_fields->Du_mu[1][idx]*(hydro_fields->Wmunu_rk0[4][idx]*u1)
        + hydro_fields->Du_mu[2][idx]*(hydro_fields->Wmunu_rk0[5][idx]*u1)
        + hydro_fields->Du_mu[3][idx]*(hydro_fields->Wmunu_rk0[6][idx]*u1));
    hydro_fields->Wmunu_rk1[4][idx] += (sum + tempf)*DELTA_TAU;

    // W^12
    sum = (- (u0*hydro_fields->Wmunu_rk0[5][idx])/tau
           + (hydro_fields->expansion_rate[idx]
               *hydro_fields->Wmunu_rk0[5][idx]));
    tempf = ((u3/tau)*((hydro_fields->Wmunu_rk0[8][idx]*u1
                        + hydro_fields->Wmunu_rk0[6][idx]*u2)*u0
                       - (hydro_fields->Wmunu_rk0[1][idx]*u2
                          + hydro_fields->Wmunu_rk0[2][idx]*u1)*u3));
    tempf += (
        - hydro_fields->Du_mu[0][idx]*(hydro_fields->Wmunu_rk0[2][idx]*u1
                                       + hydro_fields->Wmunu_rk0[1][idx]*u2)
        + hydro_fields->Du_mu[1][idx]*(hydro_fields->Wmunu_rk0[5][idx]*u1
                                       + hydro_fields->Wmunu_rk0[4][idx]*u2)
        + hydro_fields->Du_mu[2][idx]*(hydro_fields->Wmunu_rk0[7][idx]*u1
                                       + hydro_fields->Wmunu_rk0[5][idx]*u2)
        + hydro_fields->Du_mu[3][idx]*(hydro_fields->Wmunu_rk0[8][idx]*u1
                                       + hydro_fields->Wmunu_rk0[6][idx]*u2));
    hydro_fields->Wmunu_rk1[5][idx] += (sum + tempf)*DELTA_TAU;

    
    // W^13
    sum = (- (u0*hydro_fields->Wmunu_rk0[6][idx])/tau
           + (hydro_fields->expansion_rate[idx]
               *hydro_fields->Wmunu_rk0[6][idx]));
    tempf = ((u3/tau)*(- hydro_fields->Wmunu_rk0[1][idx]
                       + (hydro_fields->Wmunu_rk0[9][idx]*u1
                           + hydro_fields->Wmunu_rk0[6][idx]*u3)*u0
                       - (hydro_fields->Wmunu_rk0[1][idx]*u3
                           + hydro_fields->Wmunu_rk0[3][idx]*u1)*u3));
    tempf += (
        - hydro_fields->Du_mu[0][idx]*(hydro_fields->Wmunu_rk0[3][idx]*u1
                                       + hydro_fields->Wmunu_rk0[1][idx]*u3)
        + hydro_fields->Du_mu[1][idx]*(hydro_fields->Wmunu_rk0[6][idx]*u1
                                       + hydro_fields->Wmunu_rk0[4][idx]*u3)
        + hydro_fields->Du_mu[2][idx]*(hydro_fields->Wmunu_rk0[8][idx]*u1
                                       + hydro_fields->Wmunu_rk0[5][idx]*u3)
        + hydro_fields->Du_mu[3][idx]*(hydro_fields->Wmunu_rk0[9][idx]*u1
                                       + hydro_fields->Wmunu_rk0[6][idx]*u3));
    hydro_fields->Wmunu_rk1[6][idx] += (sum + tempf)*DELTA_TAU;
    
    // W^22
    sum = (- (u0*hydro_fields->Wmunu_rk0[7][idx])/tau
           + (hydro_fields->expansion_rate[idx]
               *hydro_fields->Wmunu_rk0[7][idx]));
    tempf = ((u3/tau)*2.*u2*(hydro_fields->Wmunu_rk0[8][idx]*u0
                             - hydro_fields->Wmunu_rk0[2][idx]*u3));
    tempf += 2.*(
        - hydro_fields->Du_mu[0][idx]*(hydro_fields->Wmunu_rk0[2][idx]*u2)
        + hydro_fields->Du_mu[1][idx]*(hydro_fields->Wmunu_rk0[5][idx]*u2)
        + hydro_fields->Du_mu[2][idx]*(hydro_fields->Wmunu_rk0[7][idx]*u2)
        + hydro_fields->Du_mu[3][idx]*(hydro_fields->Wmunu_rk0[8][idx]*u2));
    hydro_fields->Wmunu_rk1[7][idx] += (sum + tempf)*DELTA_TAU;
    
    // W^23
    sum = (- (u0*hydro_fields->Wmunu_rk0[8][idx])/tau
           + (hydro_fields->expansion_rate[idx]
               *hydro_fields->Wmunu_rk0[8][idx]));
    tempf = ((u3/tau)*(- hydro_fields->Wmunu_rk0[2][idx]
                       + (hydro_fields->Wmunu_rk0[9][idx]*u2
                           + hydro_fields->Wmunu_rk0[8][idx]*u3)*u0
                       - (hydro_fields->Wmunu_rk0[2][idx]*u3
                           + hydro_fields->Wmunu_rk0[3][idx]*u2)*u3));
    tempf += (
        - hydro_fields->Du_mu[0][idx]*(hydro_fields->Wmunu_rk0[2][idx]*u3
                                       + hydro_fields->Wmunu_rk0[3][idx]*u2)
        + hydro_fields->Du_mu[1][idx]*(hydro_fields->Wmunu_rk0[5][idx]*u3
                                       + hydro_fields->Wmunu_rk0[6][idx]*u2)
        + hydro_fields->Du_mu[2][idx]*(hydro_fields->Wmunu_rk0[7][idx]*u3
                                       + hydro_fields->Wmunu_rk0[8][idx]*u2)
        + hydro_fields->Du_mu[3][idx]*(hydro_fields->Wmunu_rk0[8][idx]*u3
                                       + hydro_fields->Wmunu_rk0[9][idx]*u2));
    hydro_fields->Wmunu_rk1[8][idx] += (sum + tempf)*DELTA_TAU;

    // W^33
    sum = (- (u0*hydro_fields->Wmunu_rk0[9][idx])/tau
           + (hydro_fields->expansion_rate[idx]
               *hydro_fields->Wmunu_rk0[9][idx]));
    tempf = ((u3/tau)*2.*(u3*(hydro_fields->Wmunu_rk0[9][idx]*u0
                              - hydro_fields->Wmunu_rk0[3][idx]*u3)
                          - hydro_fields->Wmunu_rk0[3][idx]));
    tempf += 2.*(
        - hydro_fields->Du_mu[0][idx]*(hydro_fields->Wmunu_rk0[3][idx]*u3)
        + hydro_fields->Du_mu[1][idx]*(hydro_fields->Wmunu_rk0[6][idx]*u3)
        + hydro_fields->Du_mu[2][idx]*(hydro_fields->Wmunu_rk0[8][idx]*u3)
        + hydro_fields->Du_mu[3][idx]*(hydro_fields->Wmunu_rk0[9][idx]*u3));
    hydro_fields->Wmunu_rk1[9][idx] += (sum + tempf)*DELTA_TAU;

    // bulk pressure (idx_1d == 14)
    // geometric terms for bulk Pi
    //sum -= (pi_b[rk_flag])*(u[rk_flag][0])/tau;
    //sum += (pi_b[rk_flag])*theta_local;
    hydro_fields->Wmunu_rk1[14][idx] += (
        (- hydro_fields->Wmunu_rk0[14][idx]*hydro_fields->u_rk0[0][idx]/tau
         + hydro_fields->Wmunu_rk0[14][idx]*hydro_fields->expansion_rate[idx])
        *(DELTA_TAU));
    return(1);
}

double Advance::Make_uWSource(double tau, Field *hydro_fields,
                              int ieta, int ix, int iy) {
    int idx = get_indx(ieta, ix, iy);

    int include_WWterm = 1;
    int include_Vorticity_term = 0;
    int include_Wsigma_term = 1;
    if (INITIAL_PROFILE < 2) {
        include_WWterm = 0;
        include_Wsigma_term = 0;
        include_Vorticity_term = 0;
    }

    //double sigma[4][4];
    //double Wmunu[4][4];

    //for (int a = 0; a < 4; a++) {
    //    for (int b = a; b < 4; b++) {
    //        int idx_1d = util->map_2d_idx_to_1d(a, b);
    //        //Wmunu[a][b] = grid_pt->Wmunu[rk_flag][idx_1d];
    //        //sigma[a][b] = sigma_1d[idx_1d];
    //        Wmunu[a][b] = vis_array[idx][idx_1d];
    //        sigma[a][b] = velocity_array[idx][6+idx_1d];
    //    }
    //}
    //for (int a = 0; a < 4; a++) {
    //    for (int b = a+1; b < 4; b++) {
    //        Wmunu[b][a] = Wmunu[a][b];
    //        sigma[b][a] = sigma[a][b];
    //    }
    //}

    // Useful variables to define
    double epsilon = hydro_fields->e_rk0[idx];
    double rhob = hydro_fields->rhob_rk0[idx];

    double T = get_temperature(epsilon, rhob);

    double shear_to_s = 0.0;
    shear_to_s = SHEAR_TO_S;


    //  Defining transport coefficients  
    double pressure = get_pressure(epsilon, rhob);
    double shear = (shear_to_s)*(epsilon + pressure)/(T + 1e-15);
    double tau_pi = 5.0*shear/(epsilon + pressure + 1e-15);
    if (tau_pi < 0.01) {
        tau_pi = 0.01;
    }

    // transport coefficient for nonlinear terms
    // -- shear only terms -- 4Mar2013
    // transport coefficients of a massless gas of
    // single component particles
    double transport_coefficient  = 9./70.*tau_pi/shear*(4./5.);
    double transport_coefficient2 = 4./3.*tau_pi;
    double transport_coefficient3 = 10./7.*tau_pi;
    double transport_coefficient4 = 2.*tau_pi;

    // transport coefficient for nonlinear terms
    // -- coupling to bulk viscous pressure -- 4Mar2013
    // transport coefficients not yet known -- fixed to zero
    double transport_coefficient_b  = 6./5.*tau_pi;
    double transport_coefficient2_b = 0.;

    // This source has many terms
    // everything in the 1/(tau_pi) piece is here
    // third step in the split-operator time evol
    //  use Wmunu[rk_flag] and u[rk_flag] with rk_flag = 0

    // Wmunu + transport_coefficient2*Wmunu*theta

    for (int idx_1d = 4; idx_1d < 10; idx_1d++) {
        // full term is
        //- (1.0 + transport_coefficient2*theta_local)
        double tempf = (
            - (1.0 + transport_coefficient2*hydro_fields->expansion_rate[idx])
              *(hydro_fields->Wmunu_rk0[idx_1d][idx]));

        // Navier-Stokes Term -- -2.*shear*sigma^munu
        // full Navier-Stokes term is
        // sign changes according to metric sign convention
        double NS_term = - 2.*shear*hydro_fields->sigma_munu[idx_1d][idx];

        // Vorticity Term
        double Vorticity_term = 0.0;
        // for future
        // remember: dUsup[m][n] = partial^n u^m  ///
        // remember:  a[n]  =  u^m*partial_m u^n  ///
        //if (include_Vorticity_term == 1) {
        //    double term1_Vorticity;
        //    double omega[4][4];
        //    for (a = 0; a < 4; a++) {
        //        for (b = 0; b <4; b++) {
        //            omega[a][b] = (
        //                (grid_pt->dUsup[0][a][b]
        //                 - grid_pt->dUsup[0][b][a])/2.
        //                + ueta/tau/2.*(DATA->gmunu[a][0]*DATA->gmunu[b][3]
        //                               - DATA->gmunu[b][0]*DATA->gmunu[a][3])
        //                - ueta*gamma/tau/2.
        //                  *(DATA->gmunu[a][3]*grid_pt->u[rk_flag][b]
        //                    - DATA->gmunu[b][3]*grid_pt->u[rk_flag][a])
        //                + ueta*ueta/tau/2.
        //                  *(DATA->gmunu[a][0]*grid_pt->u[rk_flag][b]
        //                     - DATA->gmunu[b][0]*grid_pt->u[rk_flag][a])
        //                + (grid_pt->u[rk_flag][a]*a_local[b]
        //                   - grid_pt->u[rk_flag][b]*a_local[a])/2.);
        //        }
        //    }
        //    term1_Vorticity = (- Wmunu[mu][0]*omega[nu][0]
        //                       - Wmunu[nu][0]*omega[mu][0]
        //                       + Wmunu[mu][1]*omega[nu][1]
        //                       + Wmunu[nu][1]*omega[mu][1]
        //                       + Wmunu[mu][2]*omega[nu][2]
        //                       + Wmunu[nu][2]*omega[mu][2]
        //                       + Wmunu[mu][3]*omega[nu][3]
        //                       + Wmunu[nu][3]*omega[mu][3])/2.;
        //    // multiply term by its respective transport coefficient
        //    term1_Vorticity = transport_coefficient4*term1_Vorticity;
        //    // full term is
        //    Vorticity_term = term1_Vorticity;
        //    Vorticity_term = 0.0;
        //} else {
        //    Vorticity_term = 0.0;
        //}

        // Add nonlinear term in shear-stress tensor
        //  transport_coefficient3*Delta(mu nu)(alpha beta)*Wmu
        //  gamma sigma nu gamma
        double Wsigma, Wsigma_term;
        double term1_Wsigma, term2_Wsigma;
        if (include_Wsigma_term == 1) {
            Wsigma = (
                 //  Wmunu[0][0]*sigma[0][0]
                 //+ Wmunu[1][1]*sigma[1][1]
                 //+ Wmunu[2][2]*sigma[2][2]
                 //+ Wmunu[3][3]*sigma[3][3]
                   hydro_fields->Wmunu_rk0[0][idx]*hydro_fields->sigma_munu[0][idx]
                 + hydro_fields->Wmunu_rk0[4][idx]*hydro_fields->sigma_munu[4][idx]
                 + hydro_fields->Wmunu_rk0[7][idx]*hydro_fields->sigma_munu[7][idx]
                 + hydro_fields->Wmunu_rk0[9][idx]*hydro_fields->sigma_munu[9][idx]
                 //- 2.*(  Wmunu[0][1]*sigma[0][1]
                 //      + Wmunu[0][2]*sigma[0][2]
                 //      + Wmunu[0][3]*sigma[0][3])
                 - 2.*(  hydro_fields->Wmunu_rk0[1][idx]*hydro_fields->sigma_munu[1][idx]
                       + hydro_fields->Wmunu_rk0[2][idx]*hydro_fields->sigma_munu[2][idx]
                       + hydro_fields->Wmunu_rk0[3][idx]*hydro_fields->sigma_munu[3][idx])
                 //+2.*(  Wmunu[1][2]*sigma[1][2]
                 //     + Wmunu[1][3]*sigma[1][3]
                 //     + Wmunu[2][3]*sigma[2][3]));
                 +2.*(  hydro_fields->Wmunu_rk0[5][idx]*hydro_fields->sigma_munu[5][idx]
                      + hydro_fields->Wmunu_rk0[6][idx]*hydro_fields->sigma_munu[6][idx]
                      + hydro_fields->Wmunu_rk0[8][idx]*hydro_fields->sigma_munu[8][idx]));

            //term1_Wsigma = ( - Wmunu[mu][0]*sigma[nu][0]
            //                 - Wmunu[nu][0]*sigma[mu][0]
            //                 + Wmunu[mu][1]*sigma[nu][1]
            //                 + Wmunu[nu][1]*sigma[mu][1]
            //                 + Wmunu[mu][2]*sigma[nu][2]
            //                 + Wmunu[nu][2]*sigma[mu][2]
            //                 + Wmunu[mu][3]*sigma[nu][3]
            //                 + Wmunu[nu][3]*sigma[mu][3])/2.;
            //term2_Wsigma = (-(1./3.)*(DATA_ptr->gmunu[mu][nu]
            //                          + vis_array[15+mu]
            //                            *vis_array[15+nu])
            //                         *Wsigma);
            if (idx_1d == 4) {  // pi^xx
                term1_Wsigma = (
                    - hydro_fields->Wmunu_rk0[1][idx]*hydro_fields->sigma_munu[1][idx]
                    + hydro_fields->Wmunu_rk0[4][idx]*hydro_fields->sigma_munu[4][idx]
                    + hydro_fields->Wmunu_rk0[5][idx]*hydro_fields->sigma_munu[5][idx]
                    + hydro_fields->Wmunu_rk0[6][idx]*hydro_fields->sigma_munu[6][idx]);
                term2_Wsigma = (-(1./3.)*(1.+ hydro_fields->u_rk0[1][idx]
                                              *hydro_fields->u_rk0[1][idx])
                                         *Wsigma);
            } else if (idx_1d == 5) {  // pi^xy
                term1_Wsigma = 0.5*(
                    - (hydro_fields->Wmunu_rk0[1][idx]*hydro_fields->sigma_munu[2][idx]
                        + hydro_fields->Wmunu_rk0[2][idx]*hydro_fields->sigma_munu[1][idx])
                    + (hydro_fields->Wmunu_rk0[4][idx]*hydro_fields->sigma_munu[5][idx]
                        + hydro_fields->Wmunu_rk0[5][idx]*hydro_fields->sigma_munu[4][idx])
                    + (hydro_fields->Wmunu_rk0[5][idx]*hydro_fields->sigma_munu[7][idx]
                        + hydro_fields->Wmunu_rk0[7][idx]*hydro_fields->sigma_munu[5][idx])
                    + (hydro_fields->Wmunu_rk0[6][idx]*hydro_fields->sigma_munu[8][idx]
                        + hydro_fields->Wmunu_rk0[8][idx]*hydro_fields->sigma_munu[6][idx])
                );
                term2_Wsigma = (-(1./3.)*(hydro_fields->u_rk0[1][idx]
                                          *hydro_fields->u_rk0[2][idx])
                                         *Wsigma);
            } else if (idx_1d == 6) {  // pi^xeta
                term1_Wsigma = 0.5*(
                    - (   hydro_fields->Wmunu_rk0[1][idx]*hydro_fields->sigma_munu[3][idx]
                        + hydro_fields->Wmunu_rk0[3][idx]*hydro_fields->sigma_munu[1][idx])
                    + (   hydro_fields->Wmunu_rk0[4][idx]*hydro_fields->sigma_munu[6][idx]
                        + hydro_fields->Wmunu_rk0[6][idx]*hydro_fields->sigma_munu[4][idx])
                    + (   hydro_fields->Wmunu_rk0[5][idx]*hydro_fields->sigma_munu[8][idx]
                        + hydro_fields->Wmunu_rk0[8][idx]*hydro_fields->sigma_munu[5][idx])
                    + (   hydro_fields->Wmunu_rk0[6][idx]*hydro_fields->sigma_munu[9][idx]
                        + hydro_fields->Wmunu_rk0[9][idx]*hydro_fields->sigma_munu[6][idx])
                );
                term2_Wsigma = (-(1./3.)*(hydro_fields->u_rk0[1][idx]
                                              *hydro_fields->u_rk0[3][idx])
                                         *Wsigma);
            } else if (idx_1d == 7) {  // pi^yy
                term1_Wsigma = (
                    - hydro_fields->Wmunu_rk0[2][idx]*hydro_fields->sigma_munu[2][idx]
                    + hydro_fields->Wmunu_rk0[5][idx]*hydro_fields->sigma_munu[5][idx]
                    + hydro_fields->Wmunu_rk0[7][idx]*hydro_fields->sigma_munu[7][idx]
                    + hydro_fields->Wmunu_rk0[8][idx]*hydro_fields->sigma_munu[8][idx]);
                term2_Wsigma = (-(1./3.)*(1.+ hydro_fields->u_rk0[2][idx]
                                              *hydro_fields->u_rk0[2][idx])
                                         *Wsigma);
            } else if (idx_1d == 8) {  // pi^yeta
                term1_Wsigma = 0.5*(
                    - (   hydro_fields->Wmunu_rk0[2][idx]*hydro_fields->sigma_munu[3][idx]
                        + hydro_fields->Wmunu_rk0[3][idx]*hydro_fields->sigma_munu[2][idx])
                    + (   hydro_fields->Wmunu_rk0[5][idx]*hydro_fields->sigma_munu[6][idx]
                        + hydro_fields->Wmunu_rk0[6][idx]*hydro_fields->sigma_munu[5][idx])
                    + (   hydro_fields->Wmunu_rk0[7][idx]*hydro_fields->sigma_munu[8][idx]
                        + hydro_fields->Wmunu_rk0[8][idx]*hydro_fields->sigma_munu[7][idx])
                    + (   hydro_fields->Wmunu_rk0[8][idx]*hydro_fields->sigma_munu[9][idx]
                        + hydro_fields->Wmunu_rk0[9][idx]*hydro_fields->sigma_munu[8][idx])
                );
                term2_Wsigma = (-(1./3.)*(hydro_fields->u_rk0[2][idx]
                                          *hydro_fields->u_rk0[3][idx])
                                         *Wsigma);
            } else if (idx_1d == 9) {  // pi^etaeta
                term1_Wsigma = (
                    - hydro_fields->Wmunu_rk0[3][idx]*hydro_fields->sigma_munu[3][idx]
                    + hydro_fields->Wmunu_rk0[6][idx]*hydro_fields->sigma_munu[6][idx]
                    + hydro_fields->Wmunu_rk0[8][idx]*hydro_fields->sigma_munu[8][idx]
                    + hydro_fields->Wmunu_rk0[9][idx]*hydro_fields->sigma_munu[9][idx]);
                term2_Wsigma = (-(1./3.)*(1.+ hydro_fields->u_rk0[3][idx]
                                              *hydro_fields->u_rk0[3][idx])
                                         *Wsigma);
            }

            // multiply term by its respective transport coefficient
            term1_Wsigma = transport_coefficient3*term1_Wsigma;
            term2_Wsigma = transport_coefficient3*term2_Wsigma;

            // full term is
            Wsigma_term = -term1_Wsigma - term2_Wsigma;
        } else {
            Wsigma_term = 0.0;
        }
        // Add nonlinear term in shear-stress tensor
        // transport_coefficient*Delta(mu nu)(alpha beta)*Wmu
        // gamma Wnu gamma
        double Wsquare, WW_term;
        double term1_WW, term2_WW;
        if (include_WWterm == 1) {
            //Wsquare = (  Wmunu[0][0]*Wmunu[0][0]
            //           + Wmunu[1][1]*Wmunu[1][1]
            //           + Wmunu[2][2]*Wmunu[2][2]
            //           + Wmunu[3][3]*Wmunu[3][3]
            //    - 2.*(  Wmunu[0][1]*Wmunu[0][1]
            //          + Wmunu[0][2]*Wmunu[0][2]
            //          + Wmunu[0][3]*Wmunu[0][3])
            //    + 2.*(  Wmunu[1][2]*Wmunu[1][2]
            //          + Wmunu[1][3]*Wmunu[1][3]
            //          + Wmunu[2][3]*Wmunu[2][3]));
            Wsquare = (  hydro_fields->Wmunu_rk0[0][idx]*hydro_fields->Wmunu_rk0[0][idx]
                       + hydro_fields->Wmunu_rk0[4][idx]*hydro_fields->Wmunu_rk0[4][idx]
                       + hydro_fields->Wmunu_rk0[7][idx]*hydro_fields->Wmunu_rk0[7][idx]
                       + hydro_fields->Wmunu_rk0[9][idx]*hydro_fields->Wmunu_rk0[9][idx]
                 - 2.*(  hydro_fields->Wmunu_rk0[1][idx]*hydro_fields->Wmunu_rk0[1][idx]
                       + hydro_fields->Wmunu_rk0[2][idx]*hydro_fields->Wmunu_rk0[2][idx]
                       + hydro_fields->Wmunu_rk0[3][idx]*hydro_fields->Wmunu_rk0[3][idx])
                 + 2.*(  hydro_fields->Wmunu_rk0[5][idx]*hydro_fields->Wmunu_rk0[5][idx]
                       + hydro_fields->Wmunu_rk0[6][idx]*hydro_fields->Wmunu_rk0[6][idx]
                       + hydro_fields->Wmunu_rk0[8][idx]*hydro_fields->Wmunu_rk0[8][idx]));

            //term1_WW = ( - Wmunu[mu][0]*Wmunu[nu][0]
            //             + Wmunu[mu][1]*Wmunu[nu][1]
            //             + Wmunu[mu][2]*Wmunu[nu][2]
            //             + Wmunu[mu][3]*Wmunu[nu][3]);
            //term2_WW = (
            //    -(1./3.)*(DATA_ptr->gmunu[mu][nu]
            //              + vis_array[15+mu]
            //                *vis_array[15+nu])
            //    *Wsquare);
            if (idx_1d == 4) {  // pi^xx
                term1_WW = (
                    - hydro_fields->Wmunu_rk0[1][idx]*hydro_fields->Wmunu_rk0[1][idx]
                    + hydro_fields->Wmunu_rk0[4][idx]*hydro_fields->Wmunu_rk0[4][idx]
                    + hydro_fields->Wmunu_rk0[5][idx]*hydro_fields->Wmunu_rk0[5][idx]
                    + hydro_fields->Wmunu_rk0[6][idx]*hydro_fields->Wmunu_rk0[6][idx]);
                term2_WW = (- (1./3.)*(1.+ hydro_fields->u_rk0[1][idx]
                                           *hydro_fields->u_rk0[1][idx])
                                      *Wsquare);
            } else if (idx_1d == 5) {  // pi^xy
                term1_WW = (
                    - hydro_fields->Wmunu_rk0[1][idx]*hydro_fields->Wmunu_rk0[2][idx]
                    + hydro_fields->Wmunu_rk0[4][idx]*hydro_fields->Wmunu_rk0[5][idx]
                    + hydro_fields->Wmunu_rk0[5][idx]*hydro_fields->Wmunu_rk0[7][idx]
                    + hydro_fields->Wmunu_rk0[6][idx]*hydro_fields->Wmunu_rk0[8][idx]);
                term2_WW = (- (1./3.)*(hydro_fields->u_rk0[1][idx]
                                       *hydro_fields->u_rk0[2][idx])
                                       *Wsquare);
            } else if (idx_1d == 6) {  // pi^xeta
                term1_WW = (
                    - hydro_fields->Wmunu_rk0[1][idx]*hydro_fields->Wmunu_rk0[3][idx]
                    + hydro_fields->Wmunu_rk0[4][idx]*hydro_fields->Wmunu_rk0[6][idx]
                    + hydro_fields->Wmunu_rk0[5][idx]*hydro_fields->Wmunu_rk0[8][idx]
                    + hydro_fields->Wmunu_rk0[6][idx]*hydro_fields->Wmunu_rk0[9][idx]);
                term2_WW = (- (1./3.)*(hydro_fields->u_rk0[1][idx]
                                           *hydro_fields->u_rk0[3][idx])
                                      *Wsquare);
            } else if (idx_1d == 7) {  // pi^yy
                term1_WW = (
                    - hydro_fields->Wmunu_rk0[2][idx]*hydro_fields->Wmunu_rk0[2][idx]
                    + hydro_fields->Wmunu_rk0[5][idx]*hydro_fields->Wmunu_rk0[5][idx]
                    + hydro_fields->Wmunu_rk0[7][idx]*hydro_fields->Wmunu_rk0[7][idx]
                    + hydro_fields->Wmunu_rk0[8][idx]*hydro_fields->Wmunu_rk0[8][idx]);
                term2_WW = (- (1./3.)*(1.+ hydro_fields->u_rk0[2][idx]
                                           *hydro_fields->u_rk0[2][idx])
                                      *Wsquare);
            } else if (idx_1d == 8) {  // pi^yeta
                term1_WW = (
                    - hydro_fields->Wmunu_rk0[2][idx]*hydro_fields->Wmunu_rk0[3][idx]
                    + hydro_fields->Wmunu_rk0[5][idx]*hydro_fields->Wmunu_rk0[6][idx]
                    + hydro_fields->Wmunu_rk0[7][idx]*hydro_fields->Wmunu_rk0[8][idx]
                    + hydro_fields->Wmunu_rk0[8][idx]*hydro_fields->Wmunu_rk0[9][idx]);
                term2_WW = (- (1./3.)*(hydro_fields->u_rk0[2][idx]
                                           *hydro_fields->u_rk0[3][idx])
                                      *Wsquare);
            } else if (idx_1d == 9) {  // pi^etaeta
                term1_WW = (
                    - hydro_fields->Wmunu_rk0[3][idx]*hydro_fields->Wmunu_rk0[3][idx]
                    + hydro_fields->Wmunu_rk0[6][idx]*hydro_fields->Wmunu_rk0[6][idx]
                    + hydro_fields->Wmunu_rk0[8][idx]*hydro_fields->Wmunu_rk0[8][idx]
                    + hydro_fields->Wmunu_rk0[9][idx]*hydro_fields->Wmunu_rk0[9][idx]);
                term2_WW = (- (1./3.)*(1.+ hydro_fields->u_rk0[3][idx]
                                           *hydro_fields->u_rk0[3][idx])
                                      *Wsquare);
            }


            // multiply term by its respective transport coefficient
            term1_WW = term1_WW*transport_coefficient;
            term2_WW = term2_WW*transport_coefficient;

            // full term is
            // sign changes according to metric sign convention
            WW_term = -term1_WW - term2_WW;
        } else {
            WW_term = 0.0;
        }

        // Add coupling to bulk viscous pressure
        // transport_coefficient_b*Bulk*sigma^mu nu
        // transport_coefficient2_b*Bulk*W^mu nu
        double Bulk_Sigma, Bulk_Sigma_term;
        double Bulk_W, Bulk_W_term;
        double Coupling_to_Bulk;

        //Bulk_Sigma = grid_pt->pi_b[rk_flag]*sigma[mu][nu];
        //Bulk_W = grid_pt->pi_b[rk_flag]*Wmunu[mu][nu];
        Bulk_Sigma = (hydro_fields->Wmunu_rk0[14][idx]
                        *hydro_fields->sigma_munu[idx_1d][idx]);
        Bulk_W = hydro_fields->Wmunu_rk0[14][idx]*hydro_fields->Wmunu_rk0[idx_1d][idx];

        // multiply term by its respective transport coefficient
        Bulk_Sigma_term = Bulk_Sigma*transport_coefficient_b;
        Bulk_W_term = Bulk_W*transport_coefficient2_b;

        // full term is
        // first term: 
        // sign changes according to metric sign convention
        Coupling_to_Bulk = -Bulk_Sigma_term + Bulk_W_term;

        // final answer is
        double SW = ((NS_term + tempf + Vorticity_term
                      + Wsigma_term + WW_term
                      + Coupling_to_Bulk)
                     /(tau_pi));
        hydro_fields->Wmunu_rk1[idx_1d][idx] += SW*(DELTA_TAU);
    }
    return(0);
}


double Advance::Make_uPiSource(double tau, Field *hydro_fields,
                               int ieta, int ix, int iy) {
    int idx = get_indx(ieta, ix, iy);
    // switch to include non-linear coupling terms in the bulk pi evolution
    int include_BBterm = 1;
    int include_coupling_to_shear = 1;
 
    // defining bulk viscosity coefficient
    double epsilon = hydro_fields->e_rk0[idx];
    double rhob = hydro_fields->rhob_rk0[idx];
    double temperature = get_temperature(epsilon, rhob);

    // cs2 is the velocity of sound squared
    double cs2 = get_cs2(epsilon, rhob);  
    double pressure = get_pressure(epsilon, rhob);

    // T dependent bulk viscosity from Gabriel
    double bulk = get_temperature_dependent_zeta_s(temperature);
    bulk = bulk*(epsilon + pressure)/temperature;

    // defining bulk relaxation time and
    // additional transport coefficients
    // Bulk relaxation time from kinetic theory
    double Bulk_Relax_time = (1./14.55/(1./3.-cs2)/(1./3.-cs2)
                              /(epsilon + pressure)*bulk);

    // from kinetic theory, small mass limit
    double transport_coeff1 = 2.0/3.0*(Bulk_Relax_time);
    double transport_coeff2 = 0.;  // not known; put 0

    // from kinetic theory
    double transport_coeff1_s = (8./5.*(1./3.-cs2)
                                 *Bulk_Relax_time);
    double transport_coeff2_s = 0.;  // not known;  put 0

    // Computing Navier-Stokes term (-bulk viscosity * theta)
    //double NS_term = -bulk*theta_local;
    double NS_term = -bulk*hydro_fields->expansion_rate[idx];

    // Computing relaxation term and nonlinear term:
    // - Bulk - transport_coeff1*Bulk*theta
    //double tempf = (-(grid_pt->pi_b[rk_flag])
    //         - transport_coeff1*theta_local
    //           *(grid_pt->pi_b[rk_flag]));
    double tempf = (- hydro_fields->Wmunu_rk0[14][idx]
                    - transport_coeff1*hydro_fields->expansion_rate[idx]
                      *hydro_fields->Wmunu_rk0[14][idx]);

    // Computing nonlinear term: + transport_coeff2*Bulk*Bulk
    double BB_term = 0.0;
    if (include_BBterm == 1) {
        //BB_term = (transport_coeff2*(grid_pt->pi_b[rk_flag])
        //           *(grid_pt->pi_b[rk_flag]));
        BB_term = (transport_coeff2*hydro_fields->Wmunu_rk0[14][idx]
                   *hydro_fields->Wmunu_rk0[14][idx]);
    }

    // Computing terms that Couple with shear-stress tensor
    double Wsigma, WW, Shear_Sigma_term, Shear_Shear_term;
    double Coupling_to_Shear;

    if (include_coupling_to_shear == 1) {
        // Computing sigma^mu^nu
        //double sigma[4][4], Wmunu[4][4];
        //for (int a = 0; a < 4 ; a++) {
        //    for (int b = a; b < 4; b++) {
        //        int idx_1d = util->map_2d_idx_to_1d(a, b);
        //        sigma[a][b] = velocity_array[6+idx_1d];
        //        Wmunu[a][b] = vis_array[idx_1d];
        //    }
        //}

        //Wsigma = (  Wmunu[0][0]*sigma[0][0]
        //          + Wmunu[1][1]*sigma[1][1]
        //          + Wmunu[2][2]*sigma[2][2]
        //          + Wmunu[3][3]*sigma[3][3]
        //          - 2.*(  Wmunu[0][1]*sigma[0][1]
        //                + Wmunu[0][2]*sigma[0][2]
        //                + Wmunu[0][3]*sigma[0][3])
        //          + 2.*(  Wmunu[1][2]*sigma[1][2]
        //                + Wmunu[1][3]*sigma[1][3]
        //                + Wmunu[2][3]*sigma[2][3]));
        Wsigma = (  hydro_fields->Wmunu_rk0[0][idx]*hydro_fields->sigma_munu[0][idx]
                  + hydro_fields->Wmunu_rk0[4][idx]*hydro_fields->sigma_munu[4][idx]
                  + hydro_fields->Wmunu_rk0[7][idx]*hydro_fields->sigma_munu[7][idx]
                  + hydro_fields->Wmunu_rk0[9][idx]*hydro_fields->sigma_munu[9][idx]
                  - 2.*(  hydro_fields->Wmunu_rk0[1][idx]*hydro_fields->sigma_munu[1][idx]
                        + hydro_fields->Wmunu_rk0[2][idx]*hydro_fields->sigma_munu[8][idx]
                        + hydro_fields->Wmunu_rk0[3][idx]*hydro_fields->sigma_munu[9][idx]) 
                  + 2.*(  hydro_fields->Wmunu_rk0[5][idx]*hydro_fields->sigma_munu[5][idx]
                        + hydro_fields->Wmunu_rk0[6][idx]*hydro_fields->sigma_munu[6][idx]
                        + hydro_fields->Wmunu_rk0[8][idx]*hydro_fields->sigma_munu[8][idx])
                  );

        //WW = (   Wmunu[0][0]*Wmunu[0][0]
        //       + Wmunu[1][1]*Wmunu[1][1]
        //       + Wmunu[2][2]*Wmunu[2][2]
        //       + Wmunu[3][3]*Wmunu[3][3]
        //       - 2.*(  Wmunu[0][1]*Wmunu[0][1]
        //             + Wmunu[0][2]*Wmunu[0][2]
        //             + Wmunu[0][3]*Wmunu[0][3])
        //       + 2.*(  Wmunu[1][2]*Wmunu[1][2]
        //             + Wmunu[1][3]*Wmunu[1][3]
        //             + Wmunu[2][3]*Wmunu[2][3]));
        WW = (  hydro_fields->Wmunu_rk0[0][idx]*hydro_fields->Wmunu_rk0[0][idx]
              + hydro_fields->Wmunu_rk0[4][idx]*hydro_fields->Wmunu_rk0[4][idx]
              + hydro_fields->Wmunu_rk0[8][idx]*hydro_fields->Wmunu_rk0[8][idx]
              + hydro_fields->Wmunu_rk0[9][idx]*hydro_fields->Wmunu_rk0[9][idx]
              - 2.*(  hydro_fields->Wmunu_rk0[1][idx]*hydro_fields->Wmunu_rk0[1][idx]
                    + hydro_fields->Wmunu_rk0[2][idx]*hydro_fields->Wmunu_rk0[2][idx]
                    + hydro_fields->Wmunu_rk0[3][idx]*hydro_fields->Wmunu_rk0[3][idx])
              + 2.*(  hydro_fields->Wmunu_rk0[5][idx]*hydro_fields->Wmunu_rk0[5][idx]
                    + hydro_fields->Wmunu_rk0[6][idx]*hydro_fields->Wmunu_rk0[6][idx]
                    + hydro_fields->Wmunu_rk0[8][idx]*hydro_fields->Wmunu_rk0[8][idx]));
        // multiply term by its transport coefficient
        Shear_Sigma_term = Wsigma*transport_coeff1_s;
        Shear_Shear_term = WW*transport_coeff2_s;

        // full term that couples to shear is
        Coupling_to_Shear = (- Shear_Sigma_term + Shear_Shear_term);
    } else {
        Coupling_to_Shear = 0.0;
    }
    
    // Final Answer
    double Final_Answer = (NS_term + tempf + BB_term
                            + Coupling_to_Shear)/Bulk_Relax_time;
    hydro_fields->Wmunu_rk1[14][idx] += Final_Answer*(DELTA_TAU);
    return(0);
}


/* baryon current parts */
/* this contains the source terms
   that is, all the terms that are not part of the current */
/* for the q part, we don't do tau*u*q we just do u*q 
   this part contains 
    -(1/tau_rho)(q[a] + kappa g[a][b]Dtildemu[b]
                 + kappa u[a] u[b]g[b][c]Dtildemu[c])
    +Delta[a][tau] u[eta] q[eta]/tau
    -Delta[a][eta] u[eta] q[tau]/tau
    -u[a]u[b]g[b][e] Dq[e]
*/

int Advance::map_2d_idx_to_1d(int a, int b) {
    // this function maps the 2d indeices of a symmetric matrix to the index
    // in a 1-d array, which only stores the 10 independent components
    if (a == 4) {
        return(10 + b);
    } else if (a > b) {  // symmetric matrix
        return(map_2d_idx_to_1d(b, a));
    }
    if (a == 0) {
        return(b);
    } else if (a == 1) {
        return(3 + b);
    } else if (a == 2) {
        return(5 + b);
    } else if (a == 3) {
        return(9);
    } else {
        return(-1);
    }
}


double Advance::get_temperature_dependent_zeta_s(double temperature) {
    // T dependent bulk viscosity from Gabriel
    /////////////////////////////////////////////
    //           Parametrization 1             //
    /////////////////////////////////////////////
    double Ttr=0.18/0.1973;
    double dummy=temperature/Ttr;
    double A1=-13.77, A2=27.55, A3=13.45;
    double lambda1=0.9, lambda2=0.25, lambda3=0.9, lambda4=0.22;
    double sigma1=0.025, sigma2=0.13, sigma3=0.0025, sigma4=0.022;
 
    double bulk = A1*dummy*dummy + A2*dummy - A3;
    if (temperature < 0.995*Ttr) {
        bulk = (lambda3*exp((dummy-1)/sigma3)
                + lambda4*exp((dummy-1)/sigma4) + 0.03);
    }
    if (temperature > 1.05*Ttr) {
        bulk = (lambda1*exp(-(dummy-1)/sigma1)
                + lambda2*exp(-(dummy-1)/sigma2) + 0.001);
    }
    

    return(bulk);
}


//! This function computes derivatives of flow velocity
void Advance::calculate_u_derivatives(double tau, Field *hydro_fields,
                                      int ieta, int ix, int iy) {
    int idx = get_indx(ieta, ix, iy);

    double u0 = hydro_fields->u_rk0[0][idx];
    double u1 = hydro_fields->u_rk0[1][idx];
    double u2 = hydro_fields->u_rk0[2][idx];
    double u3 = hydro_fields->u_rk0[3][idx];

    // the tau-direction
    double D0u1 = - (u1 - hydro_fields->u_prev[1][idx])/DELTA_TAU;
    double D0u2 = - (u2 - hydro_fields->u_prev[2][idx])/DELTA_TAU;
    double D0u3 = - (u3 - hydro_fields->u_prev[3][idx])/DELTA_TAU;
    double D0u0 = (u1*D0u1 + u2*D0u2 + u3*D0u3)/u0;

    // the x-direction
    int idx_p_1 = get_indx(ieta, MIN(ix + 1, GRID_SIZE_X), iy);
    int idx_m_1 = get_indx(ieta, MAX(ix - 1, 0), iy);

    double D1u1 = minmod_dx(hydro_fields->u_rk0[1][idx_p_1], u1,
                            hydro_fields->u_rk0[1][idx_m_1])/DELTA_X;
    double D1u2 = minmod_dx(hydro_fields->u_rk0[2][idx_p_1], u2,
                            hydro_fields->u_rk0[2][idx_m_1])/DELTA_X;
    double D1u3 = minmod_dx(hydro_fields->u_rk0[3][idx_p_1], u3,
                            hydro_fields->u_rk0[3][idx_m_1])/DELTA_X;
    double D1u0 = (u1*D1u1 + u2*D1u2 + u3*D1u3)/u0;
    
    // the y-direction
    idx_p_1 = get_indx(ieta, ix, MIN(iy + 1, GRID_SIZE_Y));
    idx_m_1 = get_indx(ieta, ix, MAX(iy - 1, 0));

    double D2u1 = minmod_dx(hydro_fields->u_rk0[1][idx_p_1], u1,
                            hydro_fields->u_rk0[1][idx_m_1])/DELTA_Y;
    double D2u2 = minmod_dx(hydro_fields->u_rk0[2][idx_p_1], u2,
                            hydro_fields->u_rk0[2][idx_m_1])/DELTA_Y;
    double D2u3 = minmod_dx(hydro_fields->u_rk0[3][idx_p_1], u3,
                            hydro_fields->u_rk0[3][idx_m_1])/DELTA_Y;
    double D2u0 = (u1*D2u1 + u2*D2u2 + u3*D2u3)/u0;

    // the eta-direction
    idx_p_1 = get_indx(MIN(ieta + 1, GRID_SIZE_ETA - 1), ix, iy);
    idx_m_1 = get_indx(MAX(ieta - 1, 0), ix, iy);

    double D3u1 = minmod_dx(hydro_fields->u_rk0[1][idx_p_1], u1,
                            hydro_fields->u_rk0[1][idx_m_1])/(DELTA_ETA*tau);
    double D3u2 = minmod_dx(hydro_fields->u_rk0[2][idx_p_1], u2,
                            hydro_fields->u_rk0[2][idx_m_1])/(DELTA_ETA*tau);
    double D3u3 = minmod_dx(hydro_fields->u_rk0[3][idx_p_1], u3,
                            hydro_fields->u_rk0[3][idx_m_1])/(DELTA_ETA*tau);
    double D3u0 = (u1*D3u1 + u2*D3u2 + u3*D3u3)/u0;

    // calculate the expansion rate
    hydro_fields->expansion_rate[idx] = - D0u0 + D1u1 + D2u2 + D3u3 + u0/tau;

    // calculate the Du^mu, where D = u^nu partial_nu
    hydro_fields->Du_mu[0][idx] = - u0*D0u0 + u1*D1u0 + u2*D2u0 + u3*D3u0;
    hydro_fields->Du_mu[1][idx] = - u0*D0u1 + u1*D1u1 + u2*D2u1 + u3*D3u1;
    hydro_fields->Du_mu[2][idx] = - u0*D0u2 + u1*D1u2 + u2*D2u2 + u3*D3u2;
    hydro_fields->Du_mu[3][idx] = - u0*D0u3 + u1*D1u3 + u2*D2u3 + u3*D3u3;

    // calculate the velocity shear tensor sigma^\mu\nu
    // sigma^11
    hydro_fields->sigma_munu[4][idx] = (
        D1u1 - (1. + u1*u1)*hydro_fields->expansion_rate[idx]/3.
        + u1*hydro_fields->Du_mu[1][idx]);
    // sigma^12
    hydro_fields->sigma_munu[5][idx] = (
        (D2u1 + D1u2)/2. - (0. + u1*u2)*hydro_fields->expansion_rate[idx]/3.
        + (u1*hydro_fields->Du_mu[2][idx] + u2*hydro_fields->Du_mu[1][idx])/2.
    );
    // sigma^13
    hydro_fields->sigma_munu[6][idx] = (
        (D3u1 + D1u3)/2. - (0. + u1*u3)*hydro_fields->expansion_rate[idx]/3.
        + (u1*hydro_fields->Du_mu[3][idx] + u3*hydro_fields->Du_mu[1][idx])/2.
        + u3*u0/(2.*tau)*u1);
    // sigma^22
    hydro_fields->sigma_munu[7][idx] = (
        D2u2 - (1. + u2*u2)*hydro_fields->expansion_rate[idx]/3.
        + u2*hydro_fields->Du_mu[2][idx]);
    // sigma^23
    hydro_fields->sigma_munu[8][idx] = (
        (D2u3 + D3u2)/2. - (0. + u2*u3)*hydro_fields->expansion_rate[idx]/3.
        + (u2*hydro_fields->Du_mu[3][idx] + u3*hydro_fields->Du_mu[2][idx])/2.
        + u3*u0/(2.*tau)*u2);
    
    // make sigma^33 using traceless condition
    hydro_fields->sigma_munu[9][idx] = (
        (2.*(u1*u2*hydro_fields->sigma_munu[5][idx]
             + u1*u3*hydro_fields->sigma_munu[6][idx]
             + u2*u3*hydro_fields->sigma_munu[8][idx])
         - (u0*u0 - u1*u1)*hydro_fields->sigma_munu[4][idx]
         - (u0*u0 - u2*u2)*hydro_fields->sigma_munu[7][idx])/(u0*u0 - u3*u3));

    // make sigma^0i using transversality
    hydro_fields->sigma_munu[1][idx] = (
        (hydro_fields->sigma_munu[4][idx]*u1
         + hydro_fields->sigma_munu[5][idx]*u2
         + hydro_fields->sigma_munu[6][idx]*u3)/u0);
    hydro_fields->sigma_munu[2][idx] = (
        (hydro_fields->sigma_munu[5][idx]*u1
         + hydro_fields->sigma_munu[7][idx]*u2
         + hydro_fields->sigma_munu[8][idx]*u3)/u0);
    hydro_fields->sigma_munu[3][idx] = (
        (hydro_fields->sigma_munu[6][idx]*u1
         + hydro_fields->sigma_munu[8][idx]*u2
         + hydro_fields->sigma_munu[9][idx]*u3)/u0);
    hydro_fields->sigma_munu[0][idx] = (
        (hydro_fields->sigma_munu[1][idx]*u1
         + hydro_fields->sigma_munu[2][idx]*u2
         + hydro_fields->sigma_munu[3][idx]*u3)/u0);
}


//! This function computes D^\mu(mu_B/T)
void Advance::calculate_D_mu_muB_over_T(double tau, Field *hydro_fields,
                                        int ieta, int ix, int iy) {
    double rhob, eps;
    double f, fp1, fm1;

    int idx = get_indx(ieta, ix, iy);
    rhob = hydro_fields->rhob_rk0[idx];
    eps = hydro_fields->e_rk0[idx];
    f = get_mu(eps, rhob)/get_temperature(eps, rhob);

    // the tau-direction
    rhob = hydro_fields->rhob_prev[idx];
    eps = hydro_fields->e_prev[idx];
    fm1 = get_mu(eps, rhob)/get_temperature(eps, rhob);
    hydro_fields->D_mu_mu_B_over_T[1][idx] = - (f - fm1)/(DELTA_TAU);

    // the x-direction
    int idx_p_1 = get_indx(ieta, MIN(ix + 1, GRID_SIZE_X), iy);
    int idx_m_1 = get_indx(ieta, MAX(ix - 1, 0), iy);
    rhob = hydro_fields->rhob_rk0[idx_p_1];
    eps = hydro_fields->e_rk0[idx_p_1];
    fp1 = (get_mu(eps, rhob)/get_temperature(eps, rhob));
    rhob = hydro_fields->rhob_rk0[idx_m_1];
    eps = hydro_fields->e_rk0[idx_m_1];
    fm1 = (get_mu(eps, rhob)/get_temperature(eps, rhob));
    hydro_fields->D_mu_mu_B_over_T[1][idx] = minmod_dx(fp1, f, fm1)/(DELTA_X);
    
    // the y-direction
    idx_p_1 = get_indx(ieta, ix, MIN(iy + 1, GRID_SIZE_Y));
    idx_m_1 = get_indx(ieta, ix, MAX(iy - 1, 0));
    rhob = hydro_fields->rhob_rk0[idx_p_1];
    eps = hydro_fields->e_rk0[idx_p_1];
    fp1 = (get_mu(eps, rhob)/get_temperature(eps, rhob));
    rhob = hydro_fields->rhob_rk0[idx_m_1];
    eps = hydro_fields->e_rk0[idx_m_1];
    fm1 = (get_mu(eps, rhob)/get_temperature(eps, rhob));
    hydro_fields->D_mu_mu_B_over_T[2][idx] = minmod_dx(fp1, f, fm1)/(DELTA_Y);
    
    // the eta-direction
    idx_p_1 = get_indx(MIN(ieta + 1, GRID_SIZE_ETA - 1), ix, iy);
    idx_m_1 = get_indx(MAX(ieta - 1, 0), ix, iy);
    rhob = hydro_fields->rhob_rk0[idx_p_1];
    eps = hydro_fields->e_rk0[idx_p_1];
    fp1 = (get_mu(eps, rhob)/get_temperature(eps, rhob));
    rhob = hydro_fields->rhob_rk0[idx_m_1];
    eps = hydro_fields->e_rk0[idx_m_1];
    fm1 = (get_mu(eps, rhob)/get_temperature(eps, rhob));
    hydro_fields->D_mu_mu_B_over_T[3][idx] = (
                                minmod_dx(fp1, f, fm1)/(DELTA_ETA*tau));
}


void Advance::update_field_rk1_to_rk0(Field *hydro_fields, int indx) {
    hydro_fields->e_rk0[indx] = hydro_fields->e_rk1[indx];
    hydro_fields->rhob_rk0[indx] = hydro_fields->rhob_rk1[indx];
    for (int ii = 0; ii < 4; ii++) {
        hydro_fields->u_rk0[ii][indx] = hydro_fields->u_rk1[ii][indx];
    }
    for (int ii = 0; ii < 15; ii++) {
        hydro_fields->Wmunu_rk0[ii][indx] = hydro_fields->Wmunu_rk1[ii][indx];
    }
}


void Advance::update_field_rk0_to_prev(Field *hydro_fields, int indx) {
    hydro_fields->e_prev[indx] = hydro_fields->e_rk0[indx];
    hydro_fields->rhob_prev[indx] = hydro_fields->rhob_rk0[indx];
    for (int ii = 0; ii < 4; ii++) {
        hydro_fields->u_prev[ii][indx] = hydro_fields->u_rk0[ii][indx];
    }
    for (int ii = 0; ii < 15; ii++) {
        hydro_fields->Wmunu_prev[ii][indx] = hydro_fields->Wmunu_rk0[ii][indx];
    }
}
