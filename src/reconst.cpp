// Copyright 2011 @ Bjoern Schenke, Sangyong Jeon, and Charles Gale
#include <iostream>
#include "data.h"
#include "grid.h"
#include "eos.h"
#include "reconst.h"

using namespace std;

Reconst::Reconst(EOS *eosIn, InitData *DATA_in) {
    eos = eosIn;
    eos_eps_max = eos->get_eps_max();
    DATA_ptr = DATA_in;

    echo_level = DATA_ptr->echo_level;
    v_critical = 0.563624;
    LARGE = 1e20;

    max_iter = 100;
    rel_err = 1e-9;
    abs_err = 1e-10;
}

// destructor
Reconst::~Reconst() {
}

int Reconst::ReconstIt_shell(Grid *grid_p, double tau, double *uq,
                             Grid *grid_pt, int rk_flag) {
    int flag = 0;
    flag = ReconstIt_velocity_Newton(grid_p, tau, uq, grid_pt,
                                     rk_flag);
    if (flag < 0) {
        flag = ReconstIt_velocity_iteration(grid_p, tau, uq, grid_pt,
                                            rk_flag);
    }

    if (flag < 0) {
        flag = ReconstIt(grid_p, tau, uq, grid_pt, rk_flag);
    }

    if (flag == -1) {
        revert_grid(grid_p, grid_pt, rk_flag);
    } else if (flag == -2) {
        if (uq[0]/tau < abs_err) {
            regulate_grid(grid_p, abs_err, 0);
        } else {
            regulate_grid(grid_p, uq[0]/tau, 0);
        }
    }
    return(flag);
}

//! reconstruct TJb from q[0] - q[4] solve energy density first
int Reconst::ReconstIt(Grid *grid_p, double tau, double *uq,
                       Grid *grid_pt, int rk_flag) {
    double K00, T00, J0, u[4], epsilon, p, h, rhob;
    double epsilon_prev, rhob_prev, p_prev, p_guess, temperr;
    double epsilon_next, rhob_next, p_next, err, temph, cs2;
    double eps_guess, scalef;
    int iter, alpha;
    double q[5];
    const double RECONST_PRECISION = 1e-8;

    /* prepare for the iteration */
    /* uq = qiphL, qiphR, etc 
       qiphL[alpha] means, for instance, TJ[alpha][0] 
       in the cell at x+dx/2 calculated from the left 
       */

    /* uq are the conserved charges. That is, the ones appearing in
       d_tau (Ttautau/tau) + d_eta(Ttaueta/tau) + d_perp(Tperptau) = -Tetaeta
       d_tau (Ttaueta) + d_eta(Tetaeta) + d_v(tau Tveta) = -Ttaueta/tau 
       d_tau (Ttauv) + d_eta(Tetav) + d_w(tau Twv) = 0
       d_tau (Jtau) + d_eta Jeta + d_perp(tau Jperp) = 0
       */

    /* q[0] = Ttautau/tau, q[1] = Ttaux, q[2] = Ttauy, q[3] = Ttaueta
       q[4] = Jtau */
    /* uq = qiphL, qiphR, qimhL, qimhR, qirk */

    for (alpha=0; alpha<5; alpha++) {
        q[alpha] = uq[alpha];
    }

    K00 = q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    K00 /= (tau*tau);
 
    T00 = q[0]/tau;
    J0 = q[4]/tau;
 
    if ((T00 - K00/T00) < 0.0 || (T00 < (abs_err))) {
        // can't make Tmunu with this. restore the previous value
        // remember that uq are eigher halfway cells or the final q_next
        // at this point, the original values in grid_pt->TJb are not touched.
        return(-2);
    }

    /* Iteration scheme */
    double eps_init = grid_pt->epsilon;
    double rhob_init = grid_pt->rhob;
    cs2 = eos->p_e_func(eps_init, rhob_init);
    eps_guess = GuessEps(T00, K00, cs2);
    epsilon_next = eps_guess;
     
    if (isnan(epsilon_next)) {
        cout << "problem " << eps_guess << " T00=" << T00
                      << " K00=" << K00 << " cs2=" << cs2
                      << " q[0]=" << q[0] << " uq[0] ="
                      << uq[0] << " q[1]=" << q[1] << " q[2]=" << q[2];
    }
    p_guess = eos->get_pressure(epsilon_next, rhob_init);
    p_next = p_guess;

    if (J0 == 0.0) {
        rhob_next = 0.0;
    } else {
        rhob_next = J0*sqrt((eps_guess + p_guess)/(T00 + p_guess));
        if (rhob_next < 0) {
            rhob_next = 0.0;
        }
        if (rhob_next > LARGE) {
            cout << "rhob_next is crazy! rhob = " << rhob_next;
        }
    }
    err = 1.0;
    for (iter=0; iter<100; iter++) {
        if (err < (RECONST_PRECISION)*0.01) {
            if (isnan(epsilon_next)) {
                exit(-1);
            }
            p_next = eos->get_pressure(epsilon_next, rhob_next);
            break;
        } else {
            epsilon_prev = epsilon_next;
            rhob_prev = rhob_next;
        }
   
        if (isnan(epsilon_prev)) {
            exit(-1);
        }

        p_prev = eos->get_pressure(epsilon_prev, rhob_prev);
        epsilon_next = T00 - K00/(T00 + p_prev);
        err = 0.0;
   
        if (DATA_ptr->turn_on_rhob == 1) {
            rhob_next = J0*sqrt((epsilon_prev + p_prev)/(T00 + p_prev));
            temperr = fabs((rhob_next-rhob_prev)/(rhob_prev+abs_err));
            if (temperr > LARGE)
                err += temperr;
            else if (temperr > LARGE)
                err += 1000.0;  //big enough
            else if (isnan(temperr))
                err += 1000.0;
        } else {
            rhob_next = 0.0;
        }
   
        temperr = fabs((epsilon_next-epsilon_prev)/(epsilon_prev+abs_err));
        if (temperr > LARGE)
            err += temperr;
        else if (temperr > LARGE)
            err += 1000.0;  // big enough
        else if (isnan(temperr))
            err += 1000.0;
    }

    if (iter == 100) {
        return(-1);
    }  // if iteration is unsuccessful, revert

    // update
    epsilon = grid_p->epsilon = epsilon_next;
    p = grid_p->p = p_next;
    rhob = grid_p->rhob = rhob_next;
    h = p+epsilon;
    double pressure = p_next;

    /* q[0] = Ttautau/tau, q[1] = Ttaux, q[2] = Ttauy, q[3] = Ttaueta,
       q[4] = Jtau */

    u[0] = sqrt((q[0]/tau + p)/h);
    double check_u0_var = (fabs(u[0] - grid_pt->u[rk_flag][0])
                           /(grid_pt->u[rk_flag][0]));
    if (check_u0_var > 100.) {
        if (grid_pt->epsilon > 1e-6 || echo_level > 5) {
            cout << "u0 varies more than 100 times compared to "
                          << "its value at previous time step";
            cout << "e = " << grid_pt->epsilon
                          << ", u[0] = " << u[0]
                          << ", prev_u[0] = " << grid_pt->u[rk_flag][0];
        }
        return(-1);
    }
    
    if (epsilon > eos_eps_max) {
        if (echo_level > 5) {
            cout << "Reconst velocity: e = " << epsilon
                          << " > e_max in the EoS table.";
	        cout << "e_max = " << eos_eps_max << " [1/fm^4]";
	        cout << "previous epsilon = " << grid_pt->epsilon
                          << " [1/fm^4]";
        }
        return(-1);
    }

    double u_max = 242582597.70489514; // cosh(20)
    if (u[0] > u_max) {
        fprintf(stderr, "Reconst: u[0] = %e is too large.\n", u[0]);
        if (grid_pt->epsilon > 0.3) {
	        fprintf(stderr, "Reconst: u[0] = %e is too large.\n", u[0]);
	        fprintf(stderr, "epsilon = %e\n", grid_pt->epsilon);
	        fprintf(stderr, "Reverting to the previous TJb...\n"); 
	    }
        return(-1);
    } else {
        u[1] = q[1]/tau/h/u[0]; 
        u[2] = q[2]/tau/h/u[0]; 
        u[3] = q[3]/tau/h/u[0]; 
    }

    /* Correcting normalization of 4-velocity */
    temph = u[0]*u[0] - u[1]*u[1] - u[2]*u[2] - u[3]*u[3];
    // Correct velocity when unitarity is not satisfied to numerical accuracy
    if (fabs(temph - 1.0) > abs_err) {
        // If the deviation is too large, exit MUSIC
        if (fabs(temph - 1.0) > 0.1) {
            fprintf(stderr, "In Reconst, reconstructed : u2 = %e\n", temph);
            fprintf(stderr, "Can't happen.\n");
            exit(0);
        } else if(fabs(temph - 1.0) > sqrt(abs_err)) {
            // Warn only when the deviation from 1 is relatively large
            fprintf(stderr, "In Reconst, reconstructed : u2 = %e\n", temph);
            fprintf(stderr, "with u[0] = %e\n", u[0]);
            fprintf(stderr, "Correcting it...\n");
        }   

        // Rescaling spatial components of velocity so that unitarity 
        // is exactly satisfied (u[0] is not modified)
        scalef = sqrt((u[0]*u[0]-1.0)/(u[1]*u[1] + u[2]*u[2] + u[3]*u[3]));
        u[1] *= scalef;
        u[2] *= scalef;
        u[3] *= scalef;
    }/* if u^mu u_\mu != 1 */
    /* End: Correcting normalization of 4-velocity */

    for (int mu = 0; mu < 4; mu++) {
        grid_p->TJb[0][4][mu] = rhob*u[mu];
        grid_p->u[0][mu] = u[mu];
        double gfac = (mu == 0 ? -1.0 : 1.0);
        double tempf = (epsilon + pressure)*u[mu]*u[mu] + pressure*gfac;
        grid_p->TJb[0][mu][mu] = tempf;
    }
    for (int mu = 0; mu < 4; mu++) {
        double tempf = (epsilon + pressure)*u[mu];
        for (int nu = mu+1; nu < 4; nu++) {
            double Tmunu = tempf*u[nu];
            grid_p->TJb[0][mu][nu] = Tmunu;
            grid_p->TJb[0][nu][mu] = Tmunu;
        }
    }

    return(1);  // on successful execution
}

//! This function reverts the grid information back its values
//! at the previous time step
void Reconst::revert_grid(Grid *grid_current, Grid *grid_prev, int rk_flag) {
    grid_current->epsilon = grid_prev->epsilon;
    grid_current->rhob = grid_prev->rhob;
    grid_current->p = grid_prev->p;
    for (int mu = 0; mu < 4; mu++) {
        grid_current->TJb[0][4][mu] = grid_prev->TJb[rk_flag][4][mu];
        grid_current->u[0][mu] = grid_prev->u[rk_flag][mu];
 
        for (int nu = 0; nu < 4; nu++) {
            grid_current->TJb[0][nu][mu] = grid_prev->TJb[rk_flag][nu][mu];
        }
    }
}

int Reconst::ReconstIt_velocity_iteration(
    Grid *grid_p, double tau, double *uq, Grid *grid_pt, int rk_flag) {
    /* reconstruct TJb from q[0] - q[4] */
    /* reconstruct velocity first for finite mu_B case */
    /* use iteration to solve v and u0 */

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

    double q[5];
    for (int alpha = 0; alpha < 5; alpha++) {
        q[alpha] = uq[alpha]/tau;
    }

    double K00 = q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    double M = sqrt(K00);
    double T00 = q[0];
    double J0 = q[4];

    if ((T00 < abs_err) || ((T00 - K00/T00) < 0.0)) {
        // can't make Tmunu with this. restore the previous value 
        // remember that uq are eigher halfway cells or the final q_next 
        // at this point, the original values in grid_pt->TJb are not touched. 
        if (echo_level > 9) {
            cout << "T00 = " << T00 << ", K00 = " << K00;
        }
        return(-2);
    }/* if t00-k00/t00 < 0.0 */

    double u[4], epsilon, pressure, rhob;

    int v_status = 1;
    int iter = 0;
    double abs_error_v = 10.0;
    double rel_error_v = 10.0;
    double v_next = 1.0;
    double v_prev = 0.0;
    do {
        iter++;
        v_next = reconst_velocity_f(v_prev, T00, M, J0);
        abs_error_v = fabs(v_next - v_prev);
        rel_error_v = 2.*abs_error_v/(v_next + v_prev + 1e-15);
        v_prev = v_next;
        if (iter > max_iter) {
            v_status = 0;
            break;
        }
    } while (abs_error_v > abs_err && rel_error_v > rel_err);

    double v_solution;
    if (v_status == 1) {
        v_solution = v_next;
    } else {
        if (echo_level > 5) {
            cout << iter << "   [" << v_prev << ",  " << v_next
                          << "]  " << abs_error_v << "  " << rel_error_v;
        }
        return(-1);
    }/* if iteration is unsuccessful, revert */
   
    // for large velocity, solve u0
    double u0_solution = 1.0;
    if (v_solution > v_critical) {
        double u0_prev = 1./sqrt(1. - v_solution*v_solution);
        int u0_status = 1;
        iter = 0;
        double u0_next;
        double abs_error_u0 = 1.0;
        double rel_error_u0 = 1.0;
        do {
            iter++;
            u0_next = reconst_u0_f(u0_prev, T00, K00, M, J0);
            abs_error_u0 = fabs(u0_next - u0_prev);
            rel_error_u0 = 2.*abs_error_u0/(u0_next + u0_prev + 1e-15);
            u0_prev = u0_next;
            if (iter > max_iter) {
                u0_status = 0;
                break;
            }
        } while (abs_error_u0 > abs_err && rel_error_u0 > rel_err);

        if (u0_status == 1) {
            u0_solution = u0_next;
        } else {
            if (echo_level > 5) {
                cout << iter << "   [" << u0_prev << ",  " << u0_next
                              << "]  " << abs_error_u0 << "  " << rel_error_u0;
            }
            return(-1);
        } /* if iteration is unsuccessful, revert */
    }

    // successfully found velocity, now update everything else
    if (v_solution < v_critical) {
        u[0] = 1./(sqrt(1. - v_solution*v_solution) + v_solution*abs_err);
        epsilon = T00 - v_solution*sqrt(K00);
        rhob = J0/u[0];
    } else {
        u[0] = u0_solution;
        epsilon = T00 - sqrt((1. - 1./(u0_solution*u0_solution))*K00);
        rhob = J0/u0_solution;
    }
    
    double check_u0_var = (fabs(u[0] - grid_pt->u[rk_flag][0])
                           /(grid_pt->u[rk_flag][0]));
    if (check_u0_var > 100.) {
        if (grid_pt->epsilon > 1e-6 || echo_level > 5) {
            cout << "u0 varies more than 100 times compared to "
                          << "its value at previous time step";
            cout << "e = " << grid_pt->epsilon
                          << ", u[0] = " << u[0]
                          << ", prev_u[0] = " << grid_pt->u[rk_flag][0];
        }
        return(-1);
    }

    if (epsilon > eos_eps_max) {
        if (echo_level > 5) {
            cout << "Reconst velocity: e = " << epsilon
                          << " > e_max in the EoS table.";
	        cout << "e_max = " << eos_eps_max << " [1/fm^4]";
	        cout << "previous epsilon = " << grid_pt->epsilon
                          << " [1/fm^4]";
        }
        return(-1);
    }

    grid_p->epsilon = epsilon;
    grid_p->rhob = rhob;

    pressure = eos->get_pressure(epsilon, rhob);
    grid_p->p = pressure;

    double u_max = 242582597.70489514; // cosh(20)
    //remove if for speed
    if(u[0] > u_max) {
        // check whether velocity is too large
        if (echo_level > 5) {
            cout << "Reconst velocity: u[0] = " << u[0]
                          << " is too large!"
                          << "epsilon = " << grid_pt->epsilon;
        }
        return(-1);
    } else {
        // individual components of velocity
        double velocity_inverse_factor = u[0]/(T00 + pressure);
        u[1] = q[1]*velocity_inverse_factor; 
        u[2] = q[2]*velocity_inverse_factor; 
        u[3] = q[3]*velocity_inverse_factor; 
    }

    // Correcting normalization of 4-velocity
    double temp_usq = u[0]*u[0] - u[1]*u[1] - u[2]*u[2] - u[3]*u[3];
    // Correct velocity when unitarity is not satisfied to numerical accuracy
    if (fabs(temp_usq - 1.0) > abs_err) {
        // If the deviation is too large, exit MUSIC
        if (fabs(temp_usq - 1.0) > 0.1*u[0]) {
            cout << "In Reconst velocity, reconstructed: u^2 - 1 = "
                          << temp_usq - 1.0;
            cout << "u[0]=" << u[0] << ", u[1]=" << u[1]
                          << ", u[2]=" << u[2] << ", u[3]=" << u[3];
            cout << "e=" << epsilon << ", rhob=" << rhob
                          << ", p=" << pressure;
            cout << "with q1 = " << q[1] << ", q2 = " << q[2]
                          << ", q3 = " << q[3];
            exit(0);
        } else if (fabs(temp_usq - 1.0) > sqrt(abs_err)*u[0]
                                                && echo_level > 5) {
            // Warn only when the deviation from 1 is relatively large
            cout << "In Reconst velocity, reconstructed: u^2 - 1 = "
                          << temp_usq - 1.0;
            double f_res;
            if (v_solution < v_critical)
                f_res = fabs(v_solution
                             - reconst_velocity_f(v_solution, T00, M, J0));
            else
                f_res = fabs(u0_solution
                             - reconst_u0_f(u0_solution, T00, K00, M, J0));
            cout << "with v = " << v_solution << ", u[0] = " << u[0]
                          << ", res = " << f_res;
            cout << "with u[1] = " << u[1]
                          << "with u[2] = " << u[2]
                          << "with u[3] = " << u[3];
            cout << "with T00 = " << T00 << ", K = " << K00;
            cout << "with q1 = " << q[1] << ", q2 = " << q[2]
                          << ", q3 = " << q[3];
        }
        // Rescaling spatial components of velocity so that unitarity 
        // is exactly satisfied (u[0] is not modified)
        double scalef = sqrt((u[0]*u[0] - 1.0)
                             /(u[1]*u[1] + u[2]*u[2] + u[3]*u[3] + abs_err));
        u[1] *= scalef;
        u[2] *= scalef;
        u[3] *= scalef;
    }// if u^mu u_\mu != 1 
    // End: Correcting normalization of 4-velocity
   
    for (int mu = 0; mu < 4; mu++) {
        grid_p->TJb[0][4][mu] = rhob*u[mu];
        grid_p->u[0][mu] = u[mu];
        double gfac = (mu == 0 ? -1.0 : 1.0);
        double tempf = (epsilon + pressure)*u[mu]*u[mu] + pressure*gfac;
        grid_p->TJb[0][mu][mu] = tempf;
    }
    for (int mu = 0; mu < 4; mu++) {
        double tempf = (epsilon + pressure)*u[mu];
        for (int nu = mu+1; nu < 4; nu++) {
            double Tmunu = tempf*u[nu];
            grid_p->TJb[0][mu][nu] = Tmunu;
            grid_p->TJb[0][nu][mu] = Tmunu;
        }
    }

    return 1;  /* on successful execution */
}/* Reconst */


//! reconstruct TJb from q[0] - q[4]
//! reconstruct velocity first for finite mu_B case
//! use Newton's method to solve v and u0
int Reconst::ReconstIt_velocity_Newton(
    Grid *grid_p, double tau, double *uq, Grid *grid_pt, int rk_flag) {
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

    double q[5];
    for (int alpha = 0; alpha < 5; alpha++) {
        q[alpha] = uq[alpha]/tau;
    }

    double K00 = q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    double M = sqrt(K00);
    double T00 = q[0];
    double J0 = q[4];

    if ((T00 < abs_err) || ((T00 - K00/T00) < 0.0)) {
        // can't make Tmunu with this. restore the previous value 
        // remember that uq are eigher halfway cells or the final q_next 
        // at this point, the original values in grid_pt->TJb are not touched. 
        if (echo_level > 9) {
            cout << "T00 = " << T00 << ", K00 = " << K00;
        }
        return(-2);
    }/* if t00-k00/t00 < 0.0 */

    double u[4], epsilon, pressure, rhob;

    double u0_guess = grid_pt->u[rk_flag][0];
    double v_guess = sqrt(1. - 1./(u0_guess*u0_guess + 1e-15));
    if (isnan(v_guess)) {
        v_guess = 0.0;
    }
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
    } while (abs_error_v > abs_err && rel_error_v > rel_err);

    double v_solution;
    if (v_status == 1) {
        v_solution = v_next;
    } else {
        if (echo_level > 5) {
            cout << iter << "   [" << v_prev << ",  " << v_next
                          << "]  " << abs_error_v << "  " << rel_error_v;
        }
        return(-1);
    }/* if iteration is unsuccessful, revert */
   
    // for large velocity, solve u0
    double u0_solution = 1.0;
    if (v_solution > v_critical) {
        double u0_prev = 1./sqrt(1. - v_solution*v_solution);
        int u0_status = 1;
        iter = 0;
        double u0_next;
        double abs_error_u0 = reconst_u0_f_Newton(u0_prev, T00, K00, M, J0);
        double rel_error_u0 = 1.0;
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
        } while (abs_error_u0 > abs_err && rel_error_u0 > rel_err);

        if (u0_status == 1) {
            u0_solution = u0_next;
        } else {
            if (echo_level > 5) {
                cout << iter << "   [" << u0_prev << ",  " << u0_next
                              << "]  " << abs_error_u0 << "  " << rel_error_u0;
            }
            return(-1);
        }  // if iteration is unsuccessful, revert
    }

    // successfully found velocity, now update everything else
    if (v_solution < v_critical) {
        u[0] = 1./(sqrt(1. - v_solution*v_solution) + v_solution*abs_err);
        epsilon = T00 - v_solution*sqrt(K00);
        rhob = J0/u[0];
    } else {
        u[0] = u0_solution;
        epsilon = T00 - sqrt((1. - 1./(u0_solution*u0_solution))*K00);
        rhob = J0/u0_solution;
    }

    double check_u0_var = (fabs(u[0] - grid_pt->u[rk_flag][0])
                           /(grid_pt->u[rk_flag][0]));
    if (check_u0_var > 100.) {
        if (grid_pt->epsilon > 1e-6 || echo_level > 5) {
            cout << "u0 varies more than 100 times compared to "
                          << "its value at previous time step";
            cout << "e = " << grid_pt->epsilon
                          << ", u[0] = " << u[0]
                          << ", prev_u[0] = " << grid_pt->u[rk_flag][0];
        }
        return(-1);
    }

    if (epsilon > eos_eps_max) {
        if (echo_level > 5) {
            cout << "Reconst velocity: e = " << epsilon
                          << " > e_max in the EoS table.";
	        cout << "e_max = " << eos_eps_max << " [1/fm^4]";
	        cout << "previous epsilon = " << grid_pt->epsilon
                          << " [1/fm^4]";
        }
        return(-1);
    }

    grid_p->epsilon = epsilon;
    grid_p->rhob = rhob;

    pressure = eos->get_pressure(epsilon, rhob);
    grid_p->p = pressure;

    // individual components of velocity
    double velocity_inverse_factor = u[0]/(T00 + pressure);

    double u_max = 242582597.70489514; // cosh(20)
    //remove if for speed
    if(u[0] > u_max) {
        // check whether velocity is too large
        if (echo_level > 5) {
            cout << "Reconst velocity: u[0] = " << u[0]
                          << " is too large!"
                          << "epsilon = " << grid_pt->epsilon;
        }
        return(-1);
    } else {
        u[1] = q[1]*velocity_inverse_factor; 
        u[2] = q[2]*velocity_inverse_factor; 
        u[3] = q[3]*velocity_inverse_factor; 
    }

    // Correcting normalization of 4-velocity
    double temp_usq = u[0]*u[0] - u[1]*u[1] - u[2]*u[2] - u[3]*u[3];
    // Correct velocity when unitarity is not satisfied to numerical accuracy
    if (fabs(temp_usq - 1.0) > abs_err) {
        // If the deviation is too large, exit MUSIC
        if (fabs(temp_usq - 1.0) > 0.1*u[0]) {
            cout << "In Reconst velocity, reconstructed: u^2 - 1 = "
                          << temp_usq - 1.0;
            cout << "u[0]=" << u[0] << ", u[1]=" << u[1]
                          << ", u[2]=" << u[2] << ", u[3]=" << u[3];
            cout << "e=" << epsilon << ", rhob=" << rhob
                          << ", p=" << pressure;
            cout << "with q1 = " << q[1] << ", q2 = " << q[2]
                          << ", q3 = " << q[3];
            exit(0);
        } else if (fabs(temp_usq - 1.0) > sqrt(abs_err)*u[0]
                                                && echo_level > 5) {
            // Warn only when the deviation from 1 is relatively large
            cout << "In Reconst velocity, reconstructed: u^2 - 1 = "
                          << temp_usq - 1.0;
            double f_res;
            if (v_solution < v_critical)
                f_res = reconst_velocity_f_Newton(v_solution, T00, M, J0);
            else
                f_res = reconst_u0_f_Newton(u0_solution, T00, K00, M, J0);
            cout << "with v = " << v_solution << ", u[0] = " << u[0]
                          << ", res = " << f_res;
            cout << "with u[1] = " << u[1]
                          << "with u[2] = " << u[2]
                          << "with u[3] = " << u[3];
            cout << "with T00 = " << T00 << ", K = " << K00;
            cout << "with q1 = " << q[1] << ", q2 = " << q[2]
                          << ", q3 = " << q[3];
        }
        // Rescaling spatial components of velocity so that unitarity 
        // is exactly satisfied (u[0] is not modified)
        double scalef = sqrt((u[0]*u[0] - 1.0)
                             /(u[1]*u[1] + u[2]*u[2] + u[3]*u[3] + abs_err));
        u[1] *= scalef;
        u[2] *= scalef;
        u[3] *= scalef;
    }  // if u^mu u_\mu != 1 
    // End: Correcting normalization of 4-velocity
   
    for (int mu = 0; mu < 4; mu++) {
        grid_p->TJb[0][4][mu] = rhob*u[mu];
        grid_p->u[0][mu] = u[mu];
        double gfac = (mu == 0 ? -1.0 : 1.0);
        double tempf = (epsilon + pressure)*u[mu]*u[mu] + pressure*gfac;
        grid_p->TJb[0][mu][mu] = tempf;
    }
    for (int mu = 0; mu < 4; mu++) {
        double tempf = (epsilon + pressure)*u[mu];
        for (int nu = mu+1; nu < 4; nu++) {
            double Tmunu = tempf*u[nu];
            grid_p->TJb[0][mu][nu] = Tmunu;
            grid_p->TJb[0][nu][mu] = Tmunu;
        }
    }

    return 1;  /* on successful execution */
}/* Reconst */


double Reconst::GuessEps(double T00, double K00, double cs2) {
    double f;
 
    if (cs2 < abs_err) {
        f = ((-K00 + T00*T00)
             *(cs2*K00*T00*T00 + pow(T00, 4) 
               + cs2*cs2*K00*(2*K00 - T00*T00)))
            /pow(T00, 5);
    } else {
        f = ((-1.0 + cs2)*T00 
            + sqrt(-4.0*cs2*K00 + T00*T00
                   + 2.0*cs2*T00*T00
                   + cs2*cs2*T00*T00))/(2.0*cs2);
    }
    return(f);
}/*  GuessEps */

double Reconst::reconst_velocity_f(double v, double T00, double M,
                                   double J0) {
    // this function returns f(v) = M/(M0 + P)
    double epsilon = T00 - v*M;
    double rho = J0*sqrt(1 - v*v);
   
    double pressure = eos->get_pressure(epsilon, rho);
    double fv = M/(T00 + pressure);
    return(fv);
}

double Reconst::reconst_velocity_f_Newton(double v, double T00, double M,
                                          double J0) {
    double fv = v - reconst_velocity_f(v, T00, M, J0);
    return(fv);
}

double Reconst::reconst_velocity_df(double v, double T00, double M,
                                    double J0) {
    // this function returns df'(v)/dv where f' = v - f(v)
    double epsilon = T00 - v*M;
    double temp = sqrt(1. - v*v);
    double rho = J0*temp;
    double temp2 = v/temp;
   
    double pressure = eos->get_pressure(epsilon, rho);
    double dPde = eos->p_e_func(epsilon, rho);
    double dPdrho = eos->p_rho_func(epsilon, rho);
    
    double temp1 = T00 + pressure;

    double dfdv = 1. - M/(temp1*temp1)*(M*dPde + J0*temp2*dPdrho);
    return(dfdv);
}

double Reconst::reconst_u0_f(double u0, double T00, double K00, double M,
                             double J0) {
    // this function returns f(u0) = (M0+P)/sqrt((M0+P)^2 - M^2)
    double epsilon = T00 - sqrt(1. - 1./u0/u0)*M;
    double rho = J0/u0;
    
    double pressure = eos->get_pressure(epsilon, rho);
    double fu = (T00 + pressure)/sqrt((T00 + pressure)*(T00 + pressure) - K00);
    return(fu);
}

double Reconst::reconst_u0_f_Newton(double u0, double T00, double K00,
                                    double M, double J0) {
    // this function returns f(u0) = u0 - (M0+P)/sqrt((M0+P)^2 - M^2)
    double fu = u0 - reconst_u0_f(u0, T00, K00, M, J0);
    return(fu);
}

double Reconst::reconst_u0_df(double u0, double T00, double K00, double M,
                              double J0) {
    // this function returns df'/du0 where f'(u0) = u0 - f(u0)
    double v = sqrt(1. - 1./(u0*u0));
    double epsilon = T00 - v*M;
    double rho = J0/u0;
    double dedu0 = - M/(u0*u0*u0*v);
    double drhodu0 = - J0/(u0*u0);
    
    double pressure = eos->get_pressure(epsilon, rho);
    double dPde = eos->p_e_func(epsilon, rho);
    double dPdrho = eos->p_rho_func(epsilon, rho);

    double denorm = pow(((T00 + pressure)*(T00 + pressure) - K00), 1.5);
    double dfdu0 = 1. + (dedu0*dPde + drhodu0*dPdrho)*K00/denorm;
    return(dfdu0);
}


//! This function regulate the grid information
void Reconst::regulate_grid(Grid *grid_cell, double elocal, int rk_flag) {
    grid_cell->epsilon = elocal;
    grid_cell->rhob = 0.0;
    double plocal = eos->get_pressure(elocal, 0.0);
    grid_cell->p = plocal;

    grid_cell->u[rk_flag][0] = 1.0;
    grid_cell->u[rk_flag][1] = 0.0;
    grid_cell->u[rk_flag][2] = 0.0;
    grid_cell->u[rk_flag][3] = 0.0;
    
    grid_cell->TJb[rk_flag][0][0] = elocal;
    grid_cell->TJb[rk_flag][1][1] = plocal;
    grid_cell->TJb[rk_flag][2][2] = plocal;
    grid_cell->TJb[rk_flag][3][3] = plocal;

    for (int mu = 0; mu < 4; mu++) {
        grid_cell->TJb[rk_flag][4][mu] = 0.0;
        for (int nu = mu+1; nu < 4; nu++) {
            grid_cell->TJb[rk_flag][mu][nu] = 0.0;
            grid_cell->TJb[rk_flag][nu][mu] = 0.0;
        }
    }
}
