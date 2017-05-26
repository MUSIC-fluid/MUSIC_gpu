// Copyright 2011 @ Bjoern Schenke, Sangyong Jeon, and Charles Gale
#include <omp.h>
#include "./util.h"
#include "./data.h"
#include "./grid.h"
#include "./reconst.h"
#include "./eos.h"
#include "./evolve.h"
#include "./advance.h"

using namespace std;

Advance::Advance(EOS *eosIn, InitData *DATA_in) {
    DATA_ptr = DATA_in;
    eos = eosIn;
    util = new Util;
    reconst_ptr = new Reconst(eos, DATA_in);
    diss = new Diss(eosIn, DATA_in);
    minmod = new Minmod(DATA_in);

    grid_nx = DATA_in->nx;
    grid_ny = DATA_in->ny;
    grid_neta = DATA_in->neta;
    rk_order = DATA_in->rk_order;
}

// destructor
Advance::~Advance() {
    delete util;
    delete diss;
    delete reconst_ptr;
    delete minmod;
}


// evolve Runge-Kutta step in tau
int Advance::AdvanceIt(double tau, InitData *DATA, Grid ***arena,
                       int rk_flag) {
    int ieta;
    for (ieta=0; ieta < grid_neta; ieta++) {
        int ix;
        #pragma omp parallel private(ix)
        {
            #pragma omp for
            for (ix = 0; ix <= grid_nx; ix++) {
	            AdvanceLocalT(tau, DATA, ieta, ix, arena, rk_flag);
            }
	    }
        #pragma omp barrier
    }
    return 1;
}/* AdvanceIt */

/* %%%%%%%%%%%%%%%%%%%%%%%%%%%  Advance Local T %%%%%%%%%%%%%%%%%% */
//! this function advances the ideal part in the transverse plane
int Advance::AdvanceLocalT(double tau, InitData *DATA, int ieta, int ix,
                           Grid ***arena, int rk_flag) {
    for (int iy = 0; iy <= grid_ny; iy++) {
        FirstRKStepT(tau, DATA, &(arena[ieta][ix][iy]), rk_flag);
        if (DATA_ptr->viscosity_flag == 1) {
            FirstRKStepW(tau, DATA, &(arena[ieta][ix][iy]), rk_flag);
        }
    }
    return(1); /* if successful */
}/* AdvanceLocalT */


/* %%%%%%%%%%%%%%%%%%%%%% First steps begins here %%%%%%%%%%%%%%%%%% */
int Advance::FirstRKStepT(double tau, InitData *DATA, Grid *grid_pt,
                          int rk_flag) {
    // this advances the ideal part
    double tau_now = tau;
    double tau_next = tau + (DATA_ptr->delta_tau);
    double tau_rk;
    if (rk_flag == 0) {
        tau_rk = tau_now;
    } else if (rk_flag == 1) {
        tau_rk = tau_next;
    } else {
        fprintf(stderr,"rk_flag = %d out of range.\n", rk_flag);
        exit(0);
    }

    double *qirk = new double[5];
    double *rhs = new double[5];
    
    // Solve partial_a T^{a mu} = -partial_a W^{a mu}
    // Update T^{mu nu}

    // MakeDelatQI gets
    //   qi = q0 if rk_flag = 0 or
    //   qi = q0 + k1 if rk_flag = 1
    // rhs[alpha] is what MakeDeltaQI outputs. 
    // It is the spatial derivative part of partial_a T^{a mu}
    // (including geometric terms)
    double *qi = new double[5];
    MakeDeltaQI(tau_rk, grid_pt, qi, rhs, rk_flag);

    for (int alpha = 0; alpha < 5; alpha++) {
        qirk[alpha] = qi[alpha] + rhs[alpha];
        // now MakeWSource returns partial_a W^{a mu}
        // (including geometric terms) 
        double dwmn = diss->MakeWSource(tau_rk, alpha, grid_pt, DATA, rk_flag);
        /* dwmn is the only one with the minus sign */
        qirk[alpha] -= dwmn*(DATA->delta_tau);

        // set baryon density back to zero if viscous correction made it
        // non-zero remove/modify if rho_b!=0 
        // - this is only to remove the viscous correction that 
        // can make rho_b negative which we do not want.
        if (DATA->turn_on_rhob == 0) {
            if (alpha == 4 && fabs(qirk[alpha]) > 1e-12)
                qirk[alpha] = 0.;
        }

        /* if rk_flag > 0, we now have q0 + k1 + k2. 
         * So add q0 and multiply by 1/2 */
        if (rk_flag > 0) {
            qirk[alpha] += get_TJb(grid_pt, 0, alpha, 0)*tau_now;
            qirk[alpha] *= 0.5;
        }
    }

    int flag = 0;
    Grid grid_rk_t;
    grid_rk_t.u = util->mtx_malloc(rk_order, 4);
    flag = reconst_ptr->ReconstIt_shell(&grid_rk_t, tau_next, qirk, grid_pt,
                                        rk_flag); 

    delete[] qi;
    delete[] qirk;
    delete[] rhs;

    if (flag != 0) {
        UpdateTJbRK(&grid_rk_t, grid_pt, rk_flag); 
        /* TJb[rk_flag+1] is filled */
    }
    util->mtx_free(grid_rk_t.u, rk_order, 4);
    return(flag);
}


/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */
/*
   Done with T 
   Start W
*/
/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */

int Advance::FirstRKStepW(double tau, InitData *DATA, Grid *grid_pt,
                          int rk_flag) {
    double tau_now = tau;
    double tau_next = tau + (DATA->delta_tau);
  
    double **w_rhs;
    w_rhs = new double* [5];
    for (int i = 0; i < 5; i++) {
        w_rhs[i] = new double[4];
        for (int j = 0; j < 4; j++) {
            w_rhs[i][j] = 0.;
        }
    }

    // Sangyong Nov 18 2014 implemented mu_max
    int mu_max;
    if (DATA->turn_on_rhob == 1)
        mu_max = 4;
    else 
        mu_max = 3;
 
    // Solve partial_a (u^a W^{mu nu}) = 0
    // Update W^{mu nu}
    // mu = 4 is the baryon current qmu

    // calculate delta uWmunu
    // need to use u[0][mu], remember rk_flag = 0 here
    // with the KT flux 
    // solve partial_tau (u^0 W^{kl}) = -partial_i (u^i W^{kl}
 
    /* Advance uWmunu */
    double tempf, temps;
    if (rk_flag == 0) {
        diss->Make_uWRHS(tau_now, grid_pt, w_rhs, DATA, rk_flag);
        for (int mu = 1; mu < 4; mu++) {
            for (int nu = mu; nu < 4; nu++) {
                int idx_1d = util->map_2d_idx_to_1d(mu, nu);
                tempf = ((grid_pt->Wmunu[rk_flag][idx_1d])
                         *(grid_pt->u[rk_flag][0]));
                temps = diss->Make_uWSource(tau_now, grid_pt, mu, nu, DATA,
                                            rk_flag); 
                tempf += temps*(DATA->delta_tau);
                tempf += w_rhs[mu][nu];
                grid_pt->Wmunu[rk_flag+1][idx_1d] = (
                                    tempf/(grid_pt->u[rk_flag+1][0]));
            }
        }
    } else if (rk_flag > 0) {
        diss->Make_uWRHS(tau_next, grid_pt, w_rhs, DATA, rk_flag);
        for (int mu = 1; mu < 4; mu++) {
            for (int nu = mu; nu < 4; nu++) {
                int idx_1d = util->map_2d_idx_to_1d(mu, nu);
                tempf = (grid_pt->Wmunu[0][idx_1d])*(grid_pt->u[0][0]);
                temps = diss->Make_uWSource(tau_next, grid_pt, mu, nu, DATA,
                                            rk_flag); 
                tempf += temps*(DATA->delta_tau);
                tempf += w_rhs[mu][nu];

                tempf += ((grid_pt->Wmunu[rk_flag][idx_1d])
                          *(grid_pt->u[rk_flag][0]));
                tempf *= 0.5;
       
                grid_pt->Wmunu[rk_flag+1][idx_1d] = (
                                            tempf/(grid_pt->u[rk_flag+1][0]));
            }
        }
    } /* rk_flag > 0 */

    if (DATA->turn_on_bulk == 1) {
        /* calculate delta u pi */
        double p_rhs;
        if (rk_flag == 0) {
            /* calculate delta u^0 pi */
            diss->Make_uPRHS(tau_now, grid_pt, &p_rhs, DATA, rk_flag);
   
            tempf = (grid_pt->pi_b[rk_flag])*(grid_pt->u[rk_flag][0]);
            temps = diss->Make_uPiSource(tau_now, grid_pt, DATA, rk_flag);
            tempf += temps*(DATA->delta_tau);
            tempf += p_rhs;
   
            grid_pt->pi_b[rk_flag+1] = tempf/(grid_pt->u[rk_flag+1][0]);
        } else if (rk_flag > 0) {
            /* calculate delta u^0 pi */
            diss->Make_uPRHS(tau_next, grid_pt, &p_rhs, DATA, rk_flag);
   
            tempf = (grid_pt->pi_b[0])*(grid_pt->u[0][0]);
            temps = diss->Make_uPiSource(tau_next, grid_pt, DATA, rk_flag);
            tempf += temps*(DATA->delta_tau);
            tempf += p_rhs;
  
            tempf += (grid_pt->pi_b[1])*(grid_pt->u[0][0]);
            tempf *= 0.5;

            grid_pt->pi_b[rk_flag+1] = tempf/(grid_pt->u[rk_flag+1][0]);
        }
    } else {
            grid_pt->pi_b[rk_flag+1] = 0.0;
    }

    // CShen: add source term for baryon diffusion
    if (DATA->turn_on_diff == 1) {
        if (rk_flag == 0) {
            diss->Make_uqRHS(tau_now, grid_pt, w_rhs, DATA, rk_flag);
            int mu = 4;
            for (int nu = 1; nu < 4; nu++) {
                int idx_1d = util->map_2d_idx_to_1d(mu, nu);
                tempf = ((grid_pt->Wmunu[rk_flag][idx_1d])
                         *(grid_pt->u[rk_flag][0]));
                temps = diss->Make_uqSource(tau_now, grid_pt, nu, DATA,
                                            rk_flag); 
                tempf += temps*(DATA->delta_tau);
                tempf += w_rhs[mu][nu];

                grid_pt->Wmunu[rk_flag+1][idx_1d] = (
                                            tempf/(grid_pt->u[rk_flag+1][0]));
            }
        } else if (rk_flag > 0) {
            diss->Make_uqRHS(tau_next, grid_pt, w_rhs, DATA, rk_flag);
            int mu = 4;
            for (int nu = 1; nu < 4; nu++) {
                int idx_1d = util->map_2d_idx_to_1d(mu, nu);
                tempf = (grid_pt->Wmunu[0][idx_1d])*(grid_pt->u[0][0]);
                temps = diss->Make_uqSource(tau_next, grid_pt, nu, DATA,
                                            rk_flag); 
                tempf += temps*(DATA->delta_tau);
                tempf += w_rhs[mu][nu];

                tempf += ((grid_pt->Wmunu[rk_flag][idx_1d])
                          *(grid_pt->u[rk_flag][0]));
                tempf *= 0.5;
       
                grid_pt->Wmunu[rk_flag+1][idx_1d] = (
                                        tempf/(grid_pt->u[rk_flag+1][0]));
            }
        } /* rk_flag > 0 */
    } else {
        for (int nu = 0; nu < 4; nu++) {
            int idx_1d = util->map_2d_idx_to_1d(4, nu);
            grid_pt->Wmunu[rk_flag+1][idx_1d] = 0.0;
        }
    }
   
    // re-make Wmunu[3][3] so that Wmunu[mu][nu] is traceless
    grid_pt->Wmunu[rk_flag+1][9] = (
            (2.*(grid_pt->u[rk_flag+1][1]*grid_pt->u[rk_flag+1][2]
                *grid_pt->Wmunu[rk_flag+1][5]
                + grid_pt->u[rk_flag+1][1]*grid_pt->u[rk_flag+1][3]
                  *grid_pt->Wmunu[rk_flag+1][6]
                + grid_pt->u[rk_flag+1][2]*grid_pt->u[rk_flag+1][3]
                  *grid_pt->Wmunu[rk_flag+1][8])
                - (grid_pt->u[rk_flag+1][0]*grid_pt->u[rk_flag+1][0] 
                   - grid_pt->u[rk_flag+1][1]*grid_pt->u[rk_flag+1][1])
                   *grid_pt->Wmunu[rk_flag+1][4] 
                - (grid_pt->u[rk_flag+1][0]*grid_pt->u[rk_flag+1][0] 
                   - grid_pt->u[rk_flag+1][2]*grid_pt->u[rk_flag+1][2])
                  *grid_pt->Wmunu[rk_flag+1][7])
            /(grid_pt->u[rk_flag+1][0]*grid_pt->u[rk_flag+1][0] 
              - grid_pt->u[rk_flag+1][3]*grid_pt->u[rk_flag+1][3]));

    // make Wmunu[i][0] using the transversality
    for (int mu = 1; mu < 4; mu++) {
        tempf = 0.0;
        for (int nu = 1; nu < 4; nu++) {
            int idx_1d = util->map_2d_idx_to_1d(mu, nu);
            tempf += (
                grid_pt->Wmunu[rk_flag+1][idx_1d]*grid_pt->u[rk_flag+1][nu]);
        }
        grid_pt->Wmunu[rk_flag+1][mu] = tempf/(grid_pt->u[rk_flag+1][0]);
    }

    // make Wmunu[0][0]
    tempf = 0.0;
    for (int nu=1; nu<4; nu++)
        tempf += grid_pt->Wmunu[rk_flag+1][nu]*grid_pt->u[rk_flag+1][nu]; 
    grid_pt->Wmunu[rk_flag+1][0] = tempf/(grid_pt->u[rk_flag+1][0]);
 
    if (DATA->turn_on_diff == 1) {
        // make qmu[0] using transversality
        for (int mu = 4; mu < mu_max + 1; mu++) {
            tempf = 0.0;
            for (int nu = 1; nu < 4; nu++) {
                int idx_1d = util->map_2d_idx_to_1d(mu, nu);
                tempf += (grid_pt->Wmunu[rk_flag+1][idx_1d]
                          *grid_pt->u[rk_flag+1][nu]);
            }
            grid_pt->Wmunu[rk_flag+1][10] = (
                                        tempf/(grid_pt->u[rk_flag+1][0]));
        }
    } else {
        grid_pt->Wmunu[rk_flag+1][10] = 0.0;
    }

    // If the energy density of the fluid element is smaller than 0.01GeV
    // reduce Wmunu using the QuestRevert algorithm
    int revert_flag = 0;
    int revert_q_flag = 0;
    if (DATA->Initial_profile != 0) {
        revert_flag = QuestRevert(tau, grid_pt, rk_flag, DATA);
        if (DATA->turn_on_diff == 1) {
            revert_q_flag = QuestRevert_qmu(tau, grid_pt, rk_flag, DATA);
        }
    }

    for (int i = 0; i < 5; i++) {
        delete[] w_rhs[i];
    }
    delete[] w_rhs;

    if (revert_flag == 1 || revert_q_flag == 1)
        return(-1);
    else
        return(1);
}/* FirstRKStepW */

// update results after RK evolution to grid_pt
void Advance::UpdateTJbRK(Grid *grid_rk, Grid *grid_pt, int rk_flag) {
    int trk_flag = rk_flag+1;

    grid_pt->epsilon_t = grid_rk->epsilon;
    grid_pt->p_t = grid_rk->p;
    grid_pt->rhob_t = grid_rk->rhob;
    
    // reconstructed grid_rk uses rk_flag 0 only
    for (int mu=0; mu<4; mu++) {
        grid_pt->u[trk_flag][mu] = grid_rk->u[0][mu];
    }
}/* UpdateTJbRK */

//! this function reduce the size of shear stress tensor and bulk pressure
//! in the dilute region to stablize numerical simulations
int Advance::QuestRevert(double tau, Grid *grid_pt, int rk_flag,
                         InitData *DATA) {
    int revert_flag = 0;
    const double energy_density_warning = 0.01;  // GeV/fm^3, T~100 MeV

    double eps_scale = 1.0;  // 1/fm^4
    double factor = 300.*tanh(grid_pt->epsilon/eps_scale);

    double pi_00 = grid_pt->Wmunu[rk_flag+1][0];
    double pi_01 = grid_pt->Wmunu[rk_flag+1][1];
    double pi_02 = grid_pt->Wmunu[rk_flag+1][2];
    double pi_03 = grid_pt->Wmunu[rk_flag+1][3];
    double pi_11 = grid_pt->Wmunu[rk_flag+1][4];
    double pi_12 = grid_pt->Wmunu[rk_flag+1][5];
    double pi_13 = grid_pt->Wmunu[rk_flag+1][6];
    double pi_22 = grid_pt->Wmunu[rk_flag+1][7];
    double pi_23 = grid_pt->Wmunu[rk_flag+1][8];
    double pi_33 = grid_pt->Wmunu[rk_flag+1][9];

    double pisize = (pi_00*pi_00 + pi_11*pi_11 + pi_22*pi_22 + pi_33*pi_33
                     - 2.*(pi_01*pi_01 + pi_02*pi_02 + pi_03*pi_03)
                     + 2.*(pi_12*pi_12 + pi_13*pi_13 + pi_23*pi_23));
  
    double pi_local = grid_pt->pi_b[rk_flag+1];
    double bulksize = 3.*pi_local*pi_local;

    double e_local = grid_pt->epsilon;
    double p_local = grid_pt->p;
    double eq_size = e_local*e_local + 3.*p_local*p_local;
       
    double rho_shear = sqrt(pisize/eq_size)/factor; 
    double rho_bulk  = sqrt(bulksize/eq_size)/factor;
 
    // Reducing the shear stress tensor 
    double rho_shear_max = 0.1;
    if (rho_shear > rho_shear_max) {
        if (e_local*hbarc > energy_density_warning) {
            printf("energy density = %lf -- |pi/(epsilon+3*P)| = %lf\n",
                   e_local*hbarc, rho_shear);
        }
        for (int mu = 0; mu < 4; mu++) {
            for (int nu = mu; nu < 4; nu++) {
                int idx_1d = util->map_2d_idx_to_1d(mu, nu);
                grid_pt->Wmunu[rk_flag+1][idx_1d] = (
                    (rho_shear_max/rho_shear)
                    *grid_pt->Wmunu[rk_flag+1][idx_1d]);
            }
        }
        revert_flag = 1;
    }
   
    // Reducing bulk viscous pressure 
    double rho_bulk_max = 0.1;
    if (rho_bulk > rho_bulk_max) {
        if (e_local*hbarc > energy_density_warning) {
            printf("energy density = %lf --  |Pi/(epsilon+3*P)| = %lf\n",
                   e_local*hbarc, rho_bulk);
        }
        grid_pt->pi_b[rk_flag+1] = (
                (rho_bulk_max/rho_bulk)*grid_pt->pi_b[rk_flag+1]);
        revert_flag = 1;
    }

    return(revert_flag);
}/* QuestRevert */


//! this function reduce the size of net baryon diffusion current
//! in the dilute region to stablize numerical simulations
int Advance::QuestRevert_qmu(double tau, Grid *grid_pt, int rk_flag,
                             InitData *DATA) {
    int revert_flag = 0;
    const double energy_density_warning = 0.01;  // GeV/fm^3, T~100 MeV
    double eps_scale = 1.0;   // in 1/fm^4
    double factor = 300.*tanh(grid_pt->epsilon/eps_scale);

    double q_mu_local[4];
    for (int i = 0; i < 4; i++) {
        // copy the value from the grid
        int idx_1d = util->map_2d_idx_to_1d(4, i);
        q_mu_local[i] = grid_pt->Wmunu[rk_flag+1][idx_1d];
    }

    // calculate the size of q^\mu
    double q_size = 0.0;
    for (int i = 0; i < 4; i++) {
        double gfac = (i == 0 ? -1.0 : 1.0);
        q_size += gfac*q_mu_local[i]*q_mu_local[i];
    }

    // first check the positivity of q^mu q_mu 
    // (in the conversion of gmn = diag(-+++))
    if (q_size < 0.0) {
        cout << "Advance::QuestRevert_qmu: q^mu q_mu = " << q_size << " < 0!"
             << endl;
        cout << "Reset it to zero!!!!" << endl;
        for (int i = 0; i < 4; i++) {
            int idx_1d = util->map_2d_idx_to_1d(4, i);
            grid_pt->Wmunu[rk_flag+1][idx_1d] = 0.0;
        }
        revert_flag = 1;
    }

    // reduce the size of q^mu according to rhoB
    double e_local = grid_pt->epsilon;
    double rhob_local = grid_pt->rhob;
    double rho_q = sqrt(q_size/(rhob_local*rhob_local))/factor;
    double rho_q_max = 0.1;
    if (rho_q > rho_q_max) {
        if (e_local*hbarc > energy_density_warning) {
            printf("energy density = %lf, rhob = %lf -- |q/rhob| = %lf\n",
                   e_local*hbarc, rhob_local, rho_q);
        }
        for (int i = 0; i < 4; i++) {
            int idx_1d = util->map_2d_idx_to_1d(4, i);
            grid_pt->Wmunu[rk_flag+1][idx_1d] =
                                (rho_q_max/rho_q)*q_mu_local[i];
        }
        revert_flag = 1;
    }
    return(revert_flag);
}


//! This function computes the rhs array. It computes the spatial
//! derivatives of T^\mu\nu using the KT algorithm
void Advance::MakeDeltaQI(double tau, Grid *grid_pt, double *qi, double *rhs,
                          int rk_flag) {
    double delta[4];
    delta[1] = DATA_ptr->delta_x;
    delta[2] = DATA_ptr->delta_y;
    delta[3] = DATA_ptr->delta_eta;

    /* \partial_tau (tau Ttautau) + \partial_eta Tetatau 
            + \partial_x (tau Txtau) + \partial_y (tau Tytau) + Tetaeta = 0 */
    /* \partial_tau (tau Ttaueta) + \partial_eta Teteta 
            + \partial_x (tau Txeta) + \partial_y (tau Txeta) + Tetatau = 0 */
    /* \partial_tau (tau Txtau) + \partial_eta Tetax + \partial_x tau T_xx
            + \partial_y tau Tyx = 0 */

    // tau*Tmu0
    for (int alpha = 0; alpha < 5; alpha++) {
        qi[alpha] = get_TJb(grid_pt, rk_flag, alpha, 0)*tau;
    }/* get qi first */

    double *qiphL = new double[5];
    double *qiphR = new double[5];
    double *qimhL = new double[5];
    double *qimhR = new double[5];
    double **DFmmp = util->mtx_malloc(5, 4);
    
    Grid grid_phL, grid_phR, grid_mhL, grid_mhR;
    grid_phL.u = util->mtx_malloc(1, 4);
    grid_phR.u = util->mtx_malloc(1, 4);
    grid_mhL.u = util->mtx_malloc(1, 4);
    grid_mhR.u = util->mtx_malloc(1, 4);

    // implement Kurganov-Tadmor scheme
    // here computes the half way T^\tau\mu currents
    for (int direc = 1; direc < 4; direc++) {
        double tau_fac = tau;
        if (direc == 3) {
            tau_fac = 1.0;
        }
        for (int alpha = 0; alpha < 5; alpha++) {
            double gphL = qi[alpha];
            double gphR = (
                    tau*get_TJb(grid_pt->nbr_p_1[direc], rk_flag, alpha, 0));
            double gmhL = (
                    tau*get_TJb(grid_pt->nbr_m_1[direc], rk_flag, alpha, 0));
            double gmhR = qi[alpha];
            double fphL = 0.5*minmod->minmod_dx(gphR, qi[alpha], gmhL);
            double fphR = -0.5*minmod->minmod_dx(
                    tau*get_TJb(grid_pt->nbr_p_2[direc], rk_flag, alpha, 0),
                    gphR, qi[alpha]);
            double fmhL = 0.5*minmod->minmod_dx(qi[alpha], gmhL,
                    tau*get_TJb(grid_pt->nbr_m_2[direc], rk_flag, alpha, 0));
            double fmhR = -0.5*minmod->minmod_dx(gphR, qi[alpha], gmhL);
            qiphL[alpha] = gphL + fphL;
            qiphR[alpha] = gphR + fphR;
            qimhL[alpha] = gmhL + fmhL;
            qimhR[alpha] = gmhR + fmhR;
        }
        // for each direction, reconstruct half-way cells
        // reconstruct e, rhob, and u[4] for half way cells
        int flag = reconst_ptr->ReconstIt_shell(
                                    &grid_phL, tau, qiphL, grid_pt, 0);
        flag *= reconst_ptr->ReconstIt_shell(
                                    &grid_phR, tau, qiphR, grid_pt, 0); 
        flag *= reconst_ptr->ReconstIt_shell(
                                    &grid_mhL, tau, qimhL, grid_pt, 0);
        flag *= reconst_ptr->ReconstIt_shell(
                                    &grid_mhR, tau, qimhR, grid_pt, 0);
        double aiphL = MaxSpeed(tau, direc, &grid_phL);
        double aiphR = MaxSpeed(tau, direc, &grid_phR);
        double aimhL = MaxSpeed(tau, direc, &grid_mhL);
        double aimhR = MaxSpeed(tau, direc, &grid_mhR);
        
        double aiph = maxi(aiphL, aiphR);
        double aimh = maxi(aimhL, aimhR);

        double FiphL[5], FiphR[5], FimhL[5], FimhR[5];
        double Fiph[5], Fimh[5];
        for (int alpha = 0; alpha < 5; alpha++) {
            FiphL[alpha] = get_TJb(&grid_phL, 0, alpha, direc)*tau_fac;
            FiphR[alpha] = get_TJb(&grid_phR, 0, alpha, direc)*tau_fac;
            FimhL[alpha] = get_TJb(&grid_mhL, 0, alpha, direc)*tau_fac;
            FimhR[alpha] = get_TJb(&grid_mhR, 0, alpha, direc)*tau_fac;
            
            // KT: H_{j+1/2} = (f(u^+_{j+1/2}) + f(u^-_{j+1/2})/2
            //                  - a_{j+1/2}(u_{j+1/2}^+ - u^-_{j+1/2})/2
            Fiph[alpha] = 0.5*(FiphL[alpha] + FiphR[alpha]);
            Fiph[alpha] -= 0.5*aiph*(qiphR[alpha] - qiphL[alpha]);
            Fimh[alpha] = 0.5*(FimhL[alpha] + FimhR[alpha]);
            Fimh[alpha] -= 0.5*aimh*(qimhR[alpha] - qimhL[alpha]);
            
            DFmmp[alpha][direc] = Fimh[alpha] - Fiph[alpha];
        }
    }
    
    delete[] qiphL;
    delete[] qiphR;
    delete[] qimhL;
    delete[] qimhR;
    
    util->mtx_free(grid_phL.u, 1, 4);
    util->mtx_free(grid_phR.u, 1, 4);
    util->mtx_free(grid_mhL.u, 1, 4);
    util->mtx_free(grid_mhR.u, 1, 4);
   
    for (int alpha = 0; alpha < 5; alpha++) {
        double sumf = 0.0; 
        for (int i = 1; i <= 3; i++) {
            sumf += DFmmp[alpha][i]/delta[i];
        } /* i */
        if (alpha == 0) {
            sumf -= get_TJb(grid_pt, rk_flag, 3, 3);
        } else if(alpha==3) {
            sumf -= get_TJb(grid_pt, rk_flag, 3, 0);
        }
        rhs[alpha] = sumf*(DATA_ptr->delta_tau);
    }/* alpha */
    
    util->mtx_free(DFmmp, 5, 4);

    return;
}/* MakeDeltaQI */


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

// determine the maximum signal propagation speed at the given direction
double Advance::MaxSpeed(double tau, int direc, Grid *grid_p) {
    //grid_p = grid_p_h_L, grid_p_h_R, grid_m_h_L, grid_m_h_R
    //these are reconstructed by Reconst which only uses u[0] and TJb[0]
    double utau = (grid_p->u[0][0]);
    double utau2 = utau*utau;
    double ux = fabs((grid_p->u[0][direc]));
    double ux2 = ux*ux;
    double ut2mux2 = utau2 - ux2;
  
    double eps = grid_p->epsilon;
    double rhob = grid_p->rhob;
  
    double vs2 = eos->get_cs2(eps, rhob);

    double den = utau2*(1. - vs2) + vs2;
    double num_temp_sqrt = (ut2mux2 - (ut2mux2 - 1.)*vs2)*vs2;
    double num;
    if (num_temp_sqrt >= 0) {
        num = utau*ux*(1. - vs2) + sqrt(num_temp_sqrt);
    } else {
        double dpde = eos->p_e_func(eps, rhob);
        double p = grid_p->p;
        double h = p+eps;
        if (dpde < 0.001) {
            num = (sqrt(-(h*dpde*h*(dpde*(-1.0 + ut2mux2) - ut2mux2))) 
                   - h*(-1.0 + dpde)*utau*ux);
        } else {
            fprintf(stderr,"WARNING: in MaxSpeed. \n");
            fprintf(stderr, "Expression under sqrt in num=%lf. \n",
                    num_temp_sqrt);
            fprintf(stderr,"at value e=%lf. \n",eps);
            fprintf(stderr,"at value p=%lf. \n",p);
            fprintf(stderr,"at value h=%lf. \n",h);
            fprintf(stderr,"at value rhob=%lf. \n",rhob);
            fprintf(stderr,"at value utau=%lf. \n", utau);
            fprintf(stderr,"at value uk=%lf. \n", ux);
            fprintf(stderr,"at value vs^2=%lf. \n", vs2);
            fprintf(stderr,"at value dpde=%lf. \n", eos->p_e_func(eps, rhob));
            fprintf(stderr,"at value dpdrhob=%lf. \n",
                    eos->p_rho_func(eps, rhob));
            fprintf(stderr, "MaxSpeed: exiting.\n");
            exit(1);
        }
    }
    
    double f = num/(den + 1e-15);
    if (f < 0.0) {
        fprintf(stderr, "SpeedMax = %e\n is negative.\n", f);
        fprintf(stderr, "Can't happen.\n");
        exit(0);
    } else if (f <  ux/utau) {
        if (num != 0.0) {
            if (fabs(f-ux/utau)<0.0001) {
                f = ux/utau;
            } else {
	            fprintf(stderr, "SpeedMax-v = %lf\n", f-ux/utau);
	            fprintf(stderr, "SpeedMax = %e\n is smaller than v = %e.\n",
                        f, ux/utau);
	            fprintf(stderr, "Can't happen.\n");
	            exit(0);
            }
        }
    } else if (f >  1.0) {
        fprintf(stderr, "SpeedMax = %e\n is bigger than 1.\n", f);
        fprintf(stderr, "Can't happen.\n");
        fprintf(stderr, "SpeedMax = num/den, num = %e, den = %e \n", num, den);
        fprintf(stderr, "cs2 = %e \n", vs2);
        f =1.;
        exit(1);
    }
    if (direc == 3)
        f /= tau;

    return f;
}/* MaxSpeed */

double Advance::get_TJb(Grid *grid_p, int rk_flag, int mu, int nu) {
    double rhob = grid_p->rhob;
    if (rk_flag == 1) {
        rhob = grid_p->rhob_t;
    }
    double u_nu = grid_p->u[rk_flag][nu];
    if (mu == 4) {
        double J_nu = rhob*u_nu;
        return(J_nu);
    } else if (mu < 4) {
        double e = grid_p->epsilon;
        if (rk_flag == 1) {
            e = grid_p->epsilon_t;
        }
        double gfac = 0.0;
        double u_mu = 0.0;
        if (mu == nu) {
            u_mu = u_nu;
            if (mu == 0) {
                gfac = -1.0;
            } else {
                gfac = 1.0;
            }
        } else {
            u_mu = grid_p->u[rk_flag][mu];
        }
        double pressure = eos->get_pressure(e, rhob);
        double T_munu = (e + pressure)*u_mu*u_nu + pressure*gfac;
        return(T_munu);
    } else {
        return(0.0);
    }
}
