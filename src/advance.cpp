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
    u_derivative_ptr = new U_derivative(eosIn, DATA_in);

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
    delete u_derivative_ptr;
}


void Advance::prepare_qi_array(
        double tau, Grid ***arena, int rk_flag, int ieta, int ix, int iy,
        int n_cell_eta, int n_cell_x, double **qi_array, double **qi_nbr_x,
        double **qi_nbr_y, double **qi_nbr_eta,
        double **qi_rk0, double **grid_array) {

    double tau_rk;
    if (rk_flag == 0) {
        tau_rk = tau;
    } else {
        tau_rk = tau + DATA_ptr->delta_tau;
    }

    // first build qi cube n_cell_x*n_cell_x*n_cell_eta
    for (int k = 0; k < n_cell_eta; k++) {
        int idx_ieta = min(ieta + k, DATA_ptr->neta - 1);
        for (int i = 0; i < n_cell_x; i++) {
            int idx_ix = min(ix + i, DATA_ptr->nx);
            for (int j = 0; j < n_cell_x; j++) {
                int idx_iy = min(iy + j, DATA_ptr->nx);
                int idx = j + n_cell_x*i + n_cell_x*n_cell_x*k;
                update_grid_array_from_grid_cell(
                                    &arena[idx_ieta][idx_ix][idx_iy],
                                    grid_array[idx], rk_flag);
                get_qmu_from_grid_array(tau_rk, qi_array[idx],
                                        grid_array[idx]);
            }
        }
    }

    double *grid_array_temp = new double[5];
    if (rk_flag == 1) {
        for (int k = 0; k < n_cell_eta; k++) {
            int idx_ieta = min(ieta + k, DATA_ptr->neta - 1);
            for (int i = 0; i < n_cell_x; i++) {
                int idx_ix = min(ix + i, DATA_ptr->nx);
                for (int j = 0; j < n_cell_x; j++) {
                    int idx_iy = min(iy + j, DATA_ptr->nx);
                    int idx = j + n_cell_x*i + n_cell_x*n_cell_x*k;
                    update_grid_array_from_grid_cell(
                                        &arena[idx_ieta][idx_ix][idx_iy],
                                        grid_array_temp, 0);
                    get_qmu_from_grid_array(tau, qi_rk0[idx],
                                            grid_array_temp);
                }
            }
        }
    }

    // now build neighbouring cells
    // x-direction
    for (int k = 0; k < n_cell_eta; k++) {
        int idx_ieta = min(ieta + k, DATA_ptr->neta - 1);
        for (int i = 0; i < n_cell_x; i++) {
            int idx_iy = min(iy + i, DATA_ptr->nx);
            int idx = 4*i + 4*n_cell_x*k;

            int idx_m_2 = max(0, ix - 2);
            int idx_m_1 = max(0, ix - 1);
            int idx_p_1 = min(ix + n_cell_x, DATA_ptr->nx);
            int idx_p_2 = min(ix + n_cell_x + 1, DATA_ptr->nx);

            update_grid_array_from_grid_cell(&arena[idx_ieta][idx_m_2][idx_iy],
                                             grid_array_temp, rk_flag);
            get_qmu_from_grid_array(tau_rk, qi_nbr_x[idx], grid_array_temp);
            update_grid_array_from_grid_cell(&arena[idx_ieta][idx_m_1][idx_iy],
                                             grid_array_temp, rk_flag);
            get_qmu_from_grid_array(tau_rk, qi_nbr_x[idx+1], grid_array_temp);
            update_grid_array_from_grid_cell(&arena[idx_ieta][idx_p_1][idx_iy],
                                             grid_array_temp, rk_flag);
            get_qmu_from_grid_array(tau_rk, qi_nbr_x[idx+2], grid_array_temp);
            update_grid_array_from_grid_cell(&arena[idx_ieta][idx_p_2][idx_iy],
                                             grid_array_temp, rk_flag);
            get_qmu_from_grid_array(tau_rk, qi_nbr_x[idx+3], grid_array_temp);
        }
    }

    // y-direction
    for (int k = 0; k < n_cell_eta; k++) {
        int idx_ieta = min(ieta + k, DATA_ptr->neta - 1);
        for (int i = 0; i < n_cell_x; i++) {
            int idx_ix = min(ix + i, DATA_ptr->nx);
            int idx = 4*i + 4*n_cell_x*k;

            int idx_m_2 = max(0, iy - 2);
            int idx_m_1 = max(0, iy - 1);
            int idx_p_1 = min(iy + n_cell_x, DATA_ptr->nx);
            int idx_p_2 = min(iy + n_cell_x + 1, DATA_ptr->nx);

            update_grid_array_from_grid_cell(&arena[idx_ieta][idx_ix][idx_m_2],
                                             grid_array_temp, rk_flag);
            get_qmu_from_grid_array(tau_rk, qi_nbr_y[idx], grid_array_temp);
            update_grid_array_from_grid_cell(&arena[idx_ieta][idx_ix][idx_m_1],
                                             grid_array_temp, rk_flag);
            get_qmu_from_grid_array(tau_rk, qi_nbr_y[idx+1], grid_array_temp);
            update_grid_array_from_grid_cell(&arena[idx_ieta][idx_ix][idx_p_1],
                                             grid_array_temp, rk_flag);
            get_qmu_from_grid_array(tau_rk, qi_nbr_y[idx+2], grid_array_temp);
            update_grid_array_from_grid_cell(&arena[idx_ieta][idx_ix][idx_p_2],
                                             grid_array_temp, rk_flag);
            get_qmu_from_grid_array(tau_rk, qi_nbr_y[idx+3], grid_array_temp);
        }
    }

    // eta-direction
    for (int k = 0; k < n_cell_x; k++) {
        int idx_iy = min(iy + k, DATA_ptr->nx);
        for (int i = 0; i < n_cell_x; i++) {
            int idx_ix = min(ix + i, DATA_ptr->nx);
            int idx = 4*i + 4*n_cell_x*k;

            int idx_m_2 = max(0, ieta - 2);
            int idx_m_1 = max(0, ieta - 1);
            int idx_p_1 = min(ieta + n_cell_eta, DATA_ptr->neta-1);
            int idx_p_2 = min(ieta + n_cell_eta + 1, DATA_ptr->neta-1);

            update_grid_array_from_grid_cell(&arena[idx_m_2][idx_ix][idx_iy],
                                             grid_array_temp, rk_flag);
            get_qmu_from_grid_array(tau_rk, qi_nbr_eta[idx], grid_array_temp);
            update_grid_array_from_grid_cell(&arena[idx_m_1][idx_ix][idx_iy],
                                             grid_array_temp, rk_flag);
            get_qmu_from_grid_array(tau_rk, qi_nbr_eta[idx+1], grid_array_temp);
            update_grid_array_from_grid_cell(&arena[idx_p_1][idx_ix][idx_iy],
                                             grid_array_temp, rk_flag);
            get_qmu_from_grid_array(tau_rk, qi_nbr_eta[idx+2], grid_array_temp);
            update_grid_array_from_grid_cell(&arena[idx_p_2][idx_ix][idx_iy],
                                             grid_array_temp, rk_flag);
            get_qmu_from_grid_array(tau_rk, qi_nbr_eta[idx+3], grid_array_temp);
        }
    }
    delete[] grid_array_temp;
}

void Advance::prepare_vis_array(
        Grid ***arena, int rk_flag, int ieta, int ix, int iy,
        int n_cell_eta, int n_cell_x, double **vis_array, double **vis_nbr_tau,
        double **vis_nbr_x, double **vis_nbr_y, double **vis_nbr_eta) {

    // first build qi cube n_cell_x*n_cell_x*n_cell_eta
    for (int k = 0; k < n_cell_eta; k++) {
        int idx_ieta = min(ieta + k, DATA_ptr->neta - 1);
        for (int i = 0; i < n_cell_x; i++) {
            int idx_ix = min(ix + i, DATA_ptr->nx);
            for (int j = 0; j < n_cell_x; j++) {
                int idx_iy = min(iy + j, DATA_ptr->nx);
                int idx = j + n_cell_x*i + n_cell_x*n_cell_x*k;
                update_vis_array_from_grid_cell(
                    &arena[idx_ieta][idx_ix][idx_iy], vis_array[idx], rk_flag);
                update_vis_prev_tau_from_grid_cell(
                    &arena[idx_ieta][idx_ix][idx_iy], vis_nbr_tau[idx],
                    rk_flag);
            }
        }
    }

    // now build neighbouring cells
    // x-direction
    for (int k = 0; k < n_cell_eta; k++) {
        int idx_ieta = min(ieta + k, DATA_ptr->neta - 1);
        for (int i = 0; i < n_cell_x; i++) {
            int idx_iy = min(iy + i, DATA_ptr->nx);
            int idx = 4*i + 4*n_cell_x*k;

            int idx_m_2 = max(0, ix - 2);
            int idx_m_1 = max(0, ix - 1);
            int idx_p_1 = min(ix + n_cell_x, DATA_ptr->nx);
            int idx_p_2 = min(ix + n_cell_x + 1, DATA_ptr->nx);

            update_vis_array_from_grid_cell(&arena[idx_ieta][idx_m_2][idx_iy],
                                            vis_nbr_x[idx], rk_flag);
            update_vis_array_from_grid_cell(&arena[idx_ieta][idx_m_1][idx_iy],
                                            vis_nbr_x[idx+1], rk_flag);
            update_vis_array_from_grid_cell(&arena[idx_ieta][idx_p_1][idx_iy],
                                            vis_nbr_x[idx+2], rk_flag);
            update_vis_array_from_grid_cell(&arena[idx_ieta][idx_p_2][idx_iy],
                                            vis_nbr_x[idx+3], rk_flag);
        }
    }
    
    // y-direction
    for (int k = 0; k < n_cell_eta; k++) {
        int idx_ieta = min(ieta + k, DATA_ptr->neta - 1);
        for (int i = 0; i < n_cell_x; i++) {
            int idx_ix = min(ix + i, DATA_ptr->nx);
            int idx = 4*i + 4*n_cell_x*k;

            int idx_m_2 = max(0, iy - 2);
            int idx_m_1 = max(0, iy - 1);
            int idx_p_1 = min(iy + n_cell_x, DATA_ptr->nx);
            int idx_p_2 = min(iy + n_cell_x + 1, DATA_ptr->nx);

            update_vis_array_from_grid_cell(&arena[idx_ieta][idx_ix][idx_m_2],
                                            vis_nbr_y[idx], rk_flag);
            update_vis_array_from_grid_cell(&arena[idx_ieta][idx_ix][idx_m_1],
                                            vis_nbr_y[idx+1], rk_flag);
            update_vis_array_from_grid_cell(&arena[idx_ieta][idx_ix][idx_p_1],
                                            vis_nbr_y[idx+2], rk_flag);
            update_vis_array_from_grid_cell(&arena[idx_ieta][idx_ix][idx_p_2],
                                            vis_nbr_y[idx+3], rk_flag);
        }
    }

    // eta-direction
    for (int k = 0; k < n_cell_x; k++) {
        int idx_iy = min(iy + k, DATA_ptr->nx);
        for (int i = 0; i < n_cell_x; i++) {
            int idx_ix = min(ix + i, DATA_ptr->nx);
            int idx = 4*i + 4*n_cell_x*k;

            int idx_m_2 = max(0, ieta - 2);
            int idx_m_1 = max(0, ieta - 1);
            int idx_p_1 = min(ieta + n_cell_eta, DATA_ptr->neta - 1);
            int idx_p_2 = min(ieta + n_cell_eta + 1, DATA_ptr->neta - 1);

            update_vis_array_from_grid_cell(&arena[idx_m_2][idx_ix][idx_iy],
                                            vis_nbr_eta[idx], rk_flag);
            update_vis_array_from_grid_cell(&arena[idx_m_1][idx_ix][idx_iy],
                                            vis_nbr_eta[idx+1], rk_flag);
            update_vis_array_from_grid_cell(&arena[idx_p_1][idx_ix][idx_iy],
                                            vis_nbr_eta[idx+2], rk_flag);
            update_vis_array_from_grid_cell(&arena[idx_p_2][idx_ix][idx_iy],
                                            vis_nbr_eta[idx+3], rk_flag);
        }
    }
}
                        
void Advance::prepare_velocity_array(double tau_rk, Grid ***arena,
                                     int ieta, int ix, int iy, int rk_flag,
                                     int n_cell_eta, int n_cell_x,
                                     double **velocity_array, 
                                     double **grid_array) {
    int trk_flag = rk_flag + 1;
    if (rk_flag == 1) {
        trk_flag = 0;
    }
    double theta_local;
    double *a_local = new double[5];
    double *sigma_local = new double[10];
    for (int k = 0; k < n_cell_eta; k++) {
        int idx_ieta = min(ieta + k, DATA_ptr->neta - 1);
        for (int i = 0; i < n_cell_x; i++) {
            int idx_ix = min(ix + i, DATA_ptr->nx);
            for (int j = 0; j < n_cell_x; j++) {
                int idx_iy = min(iy + j, DATA_ptr->nx);
                int idx = j + n_cell_x*i + n_cell_x*n_cell_x*k;
                update_grid_array_from_grid_cell(
                                    &arena[idx_ieta][idx_ix][idx_iy],
                                    grid_array[idx], trk_flag);
                theta_local = u_derivative_ptr->calculate_expansion_rate(
                            tau_rk, arena, idx_ieta, idx_ix, idx_iy, rk_flag);
                u_derivative_ptr->calculate_Du_supmu(tau_rk, arena, idx_ieta,
                                                     idx_ix, idx_iy, rk_flag,
                                                     a_local);
                u_derivative_ptr->calculate_velocity_shear_tensor(
                            tau_rk, arena, idx_ieta, idx_ix, idx_iy, rk_flag,
                            a_local, sigma_local);
                velocity_array[idx][0] = theta_local;
                for (int alpha = 0; alpha < 5; alpha++) {
                    velocity_array[idx][1+alpha] = a_local[alpha];
                }
                for (int alpha = 0; alpha < 10; alpha++) {
                    velocity_array[idx][6+alpha] = sigma_local[alpha];
                }
                for (int alpha = 0; alpha < 4; alpha++) {
                    velocity_array[idx][16+alpha] = (
                        arena[idx_ieta][idx_ix][idx_iy].dUsup[0][4][alpha]);
                }
            }
        }
    }
    delete[] a_local;
    delete[] sigma_local;
}


// evolve Runge-Kutta step in tau
int Advance::AdvanceIt(double tau, InitData *DATA, Grid ***arena,
                       int rk_flag) {
    int n_cell_eta = 1;
    int n_cell_x = 1;
    int ieta;
    for (ieta = 0; ieta < grid_neta; ieta += n_cell_eta) {
        int ix;
        #pragma omp parallel private(ix)
        {
            #pragma omp for
            for (ix = 0; ix <= grid_nx; ix += n_cell_x) {

                double **qi_array = new double* [n_cell_x*n_cell_x*n_cell_eta];
                double **qi_rk0 = new double* [n_cell_x*n_cell_x*n_cell_eta];
                double **grid_array = (
                                new double* [n_cell_x*n_cell_x*n_cell_eta]);
                for (int i = 0; i < n_cell_x*n_cell_x*n_cell_eta; i++) {
                    qi_array[i] = new double[5];
                    qi_rk0[i] = new double[5];
                    grid_array[i] = new double[5];
                }
                double **qi_nbr_x = new double* [4*n_cell_x*n_cell_eta];
                double **qi_nbr_y = new double* [4*n_cell_x*n_cell_eta];
                for (int i = 0; i < 4*n_cell_x*n_cell_eta; i++) {
                    qi_nbr_x[i] = new double[5];
                    qi_nbr_y[i] = new double[5];
                }
                double **qi_nbr_eta = new double* [4*n_cell_x*n_cell_x];
                for (int i = 0; i < 4*n_cell_x*n_cell_x; i++) {
                    qi_nbr_eta[i] = new double[5];
                }
                double **vis_array =
                                new double* [n_cell_x*n_cell_x*n_cell_eta];
                double **vis_array_new =
                                new double* [n_cell_x*n_cell_x*n_cell_eta];
                double **vis_nbr_tau =
                                new double* [n_cell_x*n_cell_x*n_cell_eta];
                double **velocity_array =
                                new double* [n_cell_x*n_cell_x*n_cell_eta];
                for (int i = 0; i < n_cell_x*n_cell_x*n_cell_eta; i++) {
                    vis_array[i] = new double[19];
                    vis_array_new[i] = new double[19];
                    vis_nbr_tau[i] = new double[19];
                    velocity_array[i] = new double[20];
                }
                double **vis_nbr_x = new double* [4*n_cell_x*n_cell_eta];
                double **vis_nbr_y = new double* [4*n_cell_x*n_cell_eta];
                for (int i = 0; i < 4*n_cell_x*n_cell_eta; i++) {
                    vis_nbr_x[i] = new double[19];
                    vis_nbr_y[i] = new double[19];
                }
                double **vis_nbr_eta = new double* [4*n_cell_x*n_cell_x];
                for (int i = 0; i < 4*n_cell_x*n_cell_x; i++) {
                    vis_nbr_eta[i] = new double[19];
                }

                for (int iy = 0; iy <= grid_ny; iy += n_cell_x) {
                    prepare_qi_array(tau, arena, rk_flag, ieta, ix, iy,
                                     n_cell_eta, n_cell_x, qi_array,
                                     qi_nbr_x, qi_nbr_y, qi_nbr_eta,
                                     qi_rk0, grid_array);
                    prepare_vis_array(arena, rk_flag, ieta, ix, iy,
                                      n_cell_eta, n_cell_x, vis_array,
                                      vis_nbr_tau, vis_nbr_x, vis_nbr_y,
                                      vis_nbr_eta);
                    FirstRKStepT(tau, &(arena[ieta][ix][iy]), rk_flag,
                                 qi_array, qi_nbr_x, qi_nbr_y, qi_nbr_eta,
                                 n_cell_eta, n_cell_x, vis_array, vis_nbr_tau,
                                 vis_nbr_x, vis_nbr_y, vis_nbr_eta,
                                 qi_rk0, grid_array);

                    update_grid_cell(grid_array, arena, rk_flag, ieta, ix, iy,
                                     n_cell_eta, n_cell_x);

                    if (DATA_ptr->viscosity_flag == 1) {
                        double tau_rk = tau;
                        if (rk_flag == 1) {
                            tau_rk = tau + DATA_ptr->delta_tau;
                        }

                        double theta_local = (
                                u_derivative_ptr->calculate_expansion_rate(
                                    tau_rk, arena, ieta, ix, iy, rk_flag));
                        double *a_local = new double[5];
                        double *sigma_local = new double[10];
                        u_derivative_ptr->calculate_Du_supmu(
                                tau_rk, arena, ieta, ix, iy, rk_flag, a_local);
                        u_derivative_ptr->calculate_velocity_shear_tensor(
                                tau_rk, arena, ieta, ix, iy, rk_flag, a_local,
                                sigma_local);

                        prepare_velocity_array(tau_rk, arena, ieta, ix, iy,
                                               rk_flag, n_cell_eta, n_cell_x,
                                               velocity_array, grid_array);

                        FirstRKStepW(tau, DATA, &(arena[ieta][ix][iy]),
                                     rk_flag, theta_local, a_local,
                                     sigma_local, vis_array, vis_nbr_tau,
                                     vis_nbr_x, vis_nbr_y, vis_nbr_eta,
                                     velocity_array, grid_array,
                                     vis_array_new);

                        delete[] a_local;
                        delete[] sigma_local;
                    }
                }

                //clean up
                for (int i = 0; i < n_cell_x*n_cell_x*n_cell_eta; i++) {
                    delete[] qi_array[i];
                    delete[] qi_rk0[i];
                    delete[] grid_array[i];
                    delete[] vis_array[i];
                    delete[] vis_nbr_tau[i];
                    delete[] velocity_array[i];
                }
                delete[] qi_array;
                delete[] qi_rk0;
                delete[] grid_array;
                delete[] vis_array;
                delete[] vis_nbr_tau;
                delete[] velocity_array;
                for (int i = 0; i < 4*n_cell_x*n_cell_eta; i++) {
                    delete[] qi_nbr_x[i];
                    delete[] qi_nbr_y[i];
                    delete[] vis_nbr_x[i];
                    delete[] vis_nbr_y[i];
                }
                delete[] qi_nbr_x;
                delete[] qi_nbr_y;
                delete[] vis_nbr_x;
                delete[] vis_nbr_y;
                for (int i = 0; i < 4*n_cell_x*n_cell_x; i++) {
                    delete[] qi_nbr_eta[i];
                    delete[] vis_nbr_eta[i];
                }
                delete[] qi_nbr_eta;
                delete[] vis_nbr_eta;
            }
	    }
        #pragma omp barrier
    }


    return(1);
}/* AdvanceIt */


/* %%%%%%%%%%%%%%%%%%%%%% First steps begins here %%%%%%%%%%%%%%%%%% */
int Advance::FirstRKStepT(double tau, Grid *grid_pt, int rk_flag,
                          double **qi_array, double **qi_nbr_x,
                          double **qi_nbr_y, double **qi_nbr_eta,
                          int n_cell_eta, int n_cell_x, double **vis_array,
                          double **vis_nbr_tau, double **vis_nbr_x,
                          double **vis_nbr_y, double **vis_nbr_eta,
                          double **qi_rk0, double **grid_array) {

    // this advances the ideal part
    double tau_next = tau + (DATA_ptr->delta_tau);
    double tau_rk;
    if (rk_flag == 0) {
        tau_rk = tau;
    } else {
        tau_rk = tau_next;
    }

    // Solve partial_a T^{a mu} = -partial_a W^{a mu}
    // Update T^{mu nu}

    // MakeDelatQI gets
    //   qi = q0 if rk_flag = 0 or
    //   qi = q0 + k1 if rk_flag = 1
    // rhs[alpha] is what MakeDeltaQI outputs. 
    // It is the spatial derivative part of partial_a T^{a mu}
    // (including geometric terms)
    MakeDeltaQI(tau_rk, qi_array, qi_nbr_x, qi_nbr_y, qi_nbr_eta,
                n_cell_eta, n_cell_x, grid_array);

    // now MakeWSource returns partial_a W^{a mu}
    // (including geometric terms) 
    diss->MakeWSource(tau_rk, qi_array, n_cell_eta, n_cell_x,
                      vis_array, vis_nbr_tau, vis_nbr_x, vis_nbr_y,
                      vis_nbr_eta);

    if (rk_flag == 1) {
        // if rk_flag == 1, we now have q0 + k1 + k2. 
        // So add q0 and multiply by 1/2
        for (int k = 0; k < n_cell_eta; k++) {
            for (int i = 0; i < n_cell_x; i++) {
                for (int j = 0; j < n_cell_x; j++) {
                    int idx = j + i*n_cell_x + k*n_cell_x*n_cell_x;
                    for (int alpha = 0; alpha < 5; alpha++) {
                        qi_array[idx][alpha] += qi_rk0[idx][alpha];
                        qi_array[idx][alpha] *= 0.5;
                    }
                }
            }
        }
    }

    double *grid_array_t = new double[5];
    for (int k = 0; k < n_cell_eta; k++) {
        for (int i = 0; i < n_cell_x; i++) {
            for (int j = 0; j < n_cell_x; j++) {
                int idx = j + i*n_cell_x + k*n_cell_x*n_cell_x;
                reconst_ptr->ReconstIt_shell(grid_array_t, tau_next,
                                             qi_array[idx], grid_array[idx]);

                // update the grid_array
                for (int alpha = 0; alpha < 5; alpha++) {
                    grid_array[idx][alpha] = grid_array_t[alpha];
                }
            }
        }
    }

    // clean up
    delete[] grid_array_t;
    return(0);
}


/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */
/*
   Done with T 
   Start W
*/
/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */

int Advance::FirstRKStepW(double tau, InitData *DATA, Grid *grid_pt,
                          int rk_flag, double theta_local, double* a_local,
                          double *sigma_local, double **vis_array,
                          double **vis_nbr_tau, double **vis_nbr_x,
                          double **vis_nbr_y, double **vis_nbr_eta,
                          double **velocity_array, double **grid_array,
                          double **vis_array_new) {

    double tau_now = tau;
    double tau_next = tau + (DATA_ptr->delta_tau);

    int trk_flag = rk_flag + 1;
    if (rk_flag == 1) {
        trk_flag = 0;
    }
  
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
    if (DATA_ptr->turn_on_rhob == 1)
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
    int idx = 0;
    double u_new[4];
    u_new[0] = 1./sqrt(1. - grid_array[idx][1]*grid_array[idx][1]
                          - grid_array[idx][2]*grid_array[idx][2]
                          - grid_array[idx][3]*grid_array[idx][3]);
    u_new[1] = grid_array[idx][1]*u_new[0];
    u_new[2] = grid_array[idx][2]*u_new[0];
    u_new[3] = grid_array[idx][3]*u_new[0];

    if (rk_flag == 0) {
        diss->Make_uWRHS(tau_now, w_rhs,
                         vis_array, vis_nbr_x, vis_nbr_y, vis_nbr_eta,
                         velocity_array);
        for (int mu = 1; mu < 4; mu++) {
            for (int nu = mu; nu < 4; nu++) {
                int idx_1d = util->map_2d_idx_to_1d(mu, nu);
                //tempf = ((grid_pt->Wmunu[rk_flag][idx_1d])
                //         *(grid_pt->u[rk_flag][0]));
                tempf = vis_array[idx][idx_1d]*vis_array[idx][15];
                temps = diss->Make_uWSource(tau_now, mu, nu,
                                            vis_array, velocity_array,
                                            grid_array); 
                tempf += temps*(DATA_ptr->delta_tau);
                tempf += w_rhs[mu][nu];
                grid_pt->Wmunu[trk_flag][idx_1d] = tempf/u_new[0];
                vis_array_new[idx][idx_1d] = tempf/u_new[0];
            }
        }
    } else if (rk_flag > 0) {
        diss->Make_uWRHS(tau_next, w_rhs,
                         vis_array, vis_nbr_x, vis_nbr_y, vis_nbr_eta,
                         velocity_array);
        for (int mu = 1; mu < 4; mu++) {
            for (int nu = mu; nu < 4; nu++) {
                int idx_1d = util->map_2d_idx_to_1d(mu, nu);
                //tempf = (grid_pt->Wmunu[0][idx_1d])*(grid_pt->prev_u[0][0]);
                tempf = vis_nbr_tau[idx][idx_1d]*vis_nbr_tau[idx][15];
                temps = diss->Make_uWSource(tau_next, mu, nu,
                                            vis_array, velocity_array,
                                            grid_array); 
                tempf += temps*(DATA_ptr->delta_tau);
                tempf += w_rhs[mu][nu];

                tempf += vis_array[idx][idx_1d]*vis_array[idx][15];
                tempf *= 0.5;
       
                grid_pt->Wmunu[trk_flag][idx_1d] = tempf/u_new[0];
                vis_array_new[idx][idx_1d] = tempf/u_new[0];
            }
        }
    } /* rk_flag > 0 */

    if (DATA->turn_on_bulk == 1) {
        /* calculate delta u pi */
        double p_rhs;
        if (rk_flag == 0) {
            /* calculate delta u^0 pi */
            diss->Make_uPRHS(tau_now, &p_rhs, vis_array,
                             vis_nbr_x, vis_nbr_y, vis_nbr_eta,
                             velocity_array);
   
            //tempf = (grid_pt->pi_b[rk_flag])*(grid_pt->u[rk_flag][0]);
            tempf = vis_array[idx][14]*vis_array[idx][15];
            temps = diss->Make_uPiSource(tau_now, vis_array,
                                         velocity_array, grid_array);
            tempf += temps*(DATA_ptr->delta_tau);
            tempf += p_rhs;
   
            //grid_pt->pi_b[trk_flag] = tempf/(grid_pt->u[trk_flag][0]);
            grid_pt->pi_b[trk_flag] = tempf/u_new[0];
            vis_array_new[idx][14] = tempf/u_new[0];
        } else if (rk_flag > 0) {
            /* calculate delta u^0 pi */
            diss->Make_uPRHS(tau_next, &p_rhs, vis_array,
                             vis_nbr_x, vis_nbr_y, vis_nbr_eta,
                             velocity_array);
   
            //tempf = (grid_pt->pi_b[0])*(grid_pt->prev_u[0][0]);
            tempf = vis_nbr_tau[idx][14]*vis_nbr_tau[idx][15];
            temps = diss->Make_uPiSource(tau_next, vis_array,
                                         velocity_array, grid_array);
            tempf += temps*(DATA_ptr->delta_tau);
            tempf += p_rhs;
  
            //tempf += (grid_pt->pi_b[1])*(grid_pt->u[0][0]);
            tempf += vis_array[idx][14]*vis_array[idx][15];
            tempf *= 0.5;

            //grid_pt->pi_b[trk_flag] = tempf/(grid_pt->u[trk_flag][0]);
            grid_pt->pi_b[trk_flag] = tempf/u_new[0];
            vis_array_new[idx][14] = tempf/u_new[0];
        }
    } else {
        grid_pt->pi_b[trk_flag] = 0.0;
        vis_array_new[idx][14] = 0.0;
    }

    // CShen: add source term for baryon diffusion
    if (DATA->turn_on_diff == 1) {
        if (rk_flag == 0) {
            diss->Make_uqRHS(tau_now, w_rhs, vis_array, vis_nbr_x,
                             vis_nbr_y, vis_nbr_eta);
            int mu = 4;
            for (int nu = 1; nu < 4; nu++) {
                int idx_1d = util->map_2d_idx_to_1d(mu, nu);
                //tempf = ((grid_pt->Wmunu[rk_flag][idx_1d])
                //         *(grid_pt->u[rk_flag][0]));
                tempf = vis_array[idx][idx_1d]*vis_array[idx][15];
                temps = diss->Make_uqSource(tau_now, nu, vis_array,
                                            velocity_array, grid_array);
                tempf += temps*(DATA_ptr->delta_tau);
                tempf += w_rhs[mu][nu];

                grid_pt->Wmunu[trk_flag][idx_1d] = tempf/u_new[0];
                                            //tempf/(grid_pt->u[trk_flag][0]));
                vis_array_new[idx][idx_1d] = tempf/u_new[0];
            }
        } else if (rk_flag > 0) {
            diss->Make_uqRHS(tau_next, w_rhs, vis_array, vis_nbr_x, vis_nbr_y,
                             vis_nbr_eta);
            int mu = 4;
            for (int nu = 1; nu < 4; nu++) {
                int idx_1d = util->map_2d_idx_to_1d(mu, nu);
                //tempf = (grid_pt->Wmunu[0][idx_1d])*(grid_pt->prev_u[0][0]);
                tempf = vis_nbr_tau[idx][idx_1d]*vis_nbr_tau[idx][15];
                temps = diss->Make_uqSource(tau_next, nu, vis_array,
                                            velocity_array, grid_array);
                tempf += temps*(DATA_ptr->delta_tau);
                tempf += w_rhs[mu][nu];

                //tempf += ((grid_pt->Wmunu[rk_flag][idx_1d])
                //          *(grid_pt->u[rk_flag][0]));
                tempf += vis_array[idx][idx_1d]*vis_array[idx][15];
                tempf *= 0.5;
       
                grid_pt->Wmunu[trk_flag][idx_1d] = tempf/u_new[0];
                                        //tempf/(grid_pt->u[trk_flag][0]));
                vis_array_new[idx][idx_1d] = tempf/u_new[0];
            }
        }
    } else {
        for (int nu = 0; nu < 4; nu++) {
            int idx_1d = util->map_2d_idx_to_1d(4, nu);
            grid_pt->Wmunu[trk_flag][idx_1d] = 0.0;
            vis_array_new[idx][idx_1d] = 0.0;
        }
    }
   
    // re-make Wmunu[3][3] so that Wmunu[mu][nu] is traceless
    //grid_pt->Wmunu[trk_flag][9] = (
    //        (2.*(grid_pt->u[trk_flag][1]*grid_pt->u[trk_flag][2]
    //            *grid_pt->Wmunu[trk_flag][5]
    //            + grid_pt->u[trk_flag][1]*grid_pt->u[trk_flag][3]
    //              *grid_pt->Wmunu[trk_flag][6]
    //            + grid_pt->u[trk_flag][2]*grid_pt->u[trk_flag][3]
    //              *grid_pt->Wmunu[trk_flag][8])
    //            - (grid_pt->u[trk_flag][0]*grid_pt->u[trk_flag][0] 
    //               - grid_pt->u[trk_flag][1]*grid_pt->u[trk_flag][1])
    //               *grid_pt->Wmunu[trk_flag][4] 
    //            - (grid_pt->u[trk_flag][0]*grid_pt->u[trk_flag][0] 
    //               - grid_pt->u[trk_flag][2]*grid_pt->u[trk_flag][2])
    //              *grid_pt->Wmunu[trk_flag][7])
    //        /(grid_pt->u[trk_flag][0]*grid_pt->u[trk_flag][0] 
    //          - grid_pt->u[trk_flag][3]*grid_pt->u[trk_flag][3]));
    vis_array_new[idx][9] = (
            (2.*(u_new[1]*u_new[2]*vis_array_new[idx][5]
                 + u_new[1]*u_new[3]*vis_array_new[idx][6]
                 + u_new[2]*u_new[3]*vis_array_new[idx][8])
                - (u_new[0]*u_new[0] - u_new[1]*u_new[1])*vis_array_new[idx][4] 
                - (u_new[0]*u_new[0] - u_new[2]*u_new[2])
                  *vis_array_new[idx][7])
            /(u_new[0]*u_new[0] - u_new[3]*u_new[3]));
    grid_pt->Wmunu[trk_flag][9] = vis_array_new[idx][9];

    // make Wmunu[i][0] using the transversality
    for (int mu = 1; mu < 4; mu++) {
        tempf = 0.0;
        for (int nu = 1; nu < 4; nu++) {
            int idx_1d = util->map_2d_idx_to_1d(mu, nu);
            //tempf += (
            //    grid_pt->Wmunu[trk_flag][idx_1d]*grid_pt->u[trk_flag][nu]);
            tempf += vis_array_new[idx][idx_1d]*u_new[nu];
        }
        //grid_pt->Wmunu[trk_flag][mu] = tempf/u_new[0];
        vis_array_new[idx][mu] = tempf/u_new[0];
        grid_pt->Wmunu[trk_flag][mu] = vis_array_new[idx][mu];
    }

    // make Wmunu[0][0]
    tempf = 0.0;
    for (int nu = 1; nu < 4; nu++) {
        //tempf += grid_pt->Wmunu[trk_flag][nu]*grid_pt->u[trk_flag][nu];
        tempf += vis_array_new[idx][nu]*u_new[nu];
    }
    //grid_pt->Wmunu[trk_flag][0] = tempf/u_new[0];
    vis_array_new[idx][0] = tempf/u_new[0];
    grid_pt->Wmunu[trk_flag][0] = vis_array_new[idx][0];
 
    if (DATA_ptr->turn_on_diff == 1) {
        // make qmu[0] using transversality
        for (int mu = 4; mu < mu_max + 1; mu++) {
            tempf = 0.0;
            for (int nu = 1; nu < 4; nu++) {
                int idx_1d = util->map_2d_idx_to_1d(mu, nu);
                //tempf += (grid_pt->Wmunu[trk_flag][idx_1d]
                //          *grid_pt->u[trk_flag][nu]);
                tempf += (vis_array_new[idx][idx_1d]*u_new[nu]);
            }
            //grid_pt->Wmunu[trk_flag][10] = tempf/u_new[0];
            vis_array_new[idx][10] = tempf/u_new[0];
            grid_pt->Wmunu[trk_flag][10] = vis_array_new[idx][10];
        }
    } else {
        grid_pt->Wmunu[trk_flag][10] = 0.0;
        vis_array_new[idx][10] = 0.0;
    }

    // If the energy density of the fluid element is smaller than 0.01GeV
    // reduce Wmunu using the QuestRevert algorithm
    int revert_flag = 0;
    int revert_q_flag = 0;
    if (DATA->Initial_profile != 0) {
        revert_flag = QuestRevert(tau, grid_pt, rk_flag,
                                  vis_array_new, grid_array);
        if (DATA->turn_on_diff == 1) {
            revert_q_flag = QuestRevert_qmu(tau, grid_pt, rk_flag,
                                            vis_array_new, grid_array);
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
}


void Advance::update_grid_array_from_grid_cell(
                Grid *grid_p, double *grid_array, int rk_flag) {
    if (rk_flag == 0) {
        grid_array[0] = grid_p->epsilon;
        grid_array[4] = grid_p->rhob;
    } else {
        grid_array[0] = grid_p->epsilon_t;
        grid_array[4] = grid_p->rhob_t;
    }
    grid_array[1] = grid_p->u[rk_flag][1]/grid_p->u[rk_flag][0];
    grid_array[2] = grid_p->u[rk_flag][2]/grid_p->u[rk_flag][0];
    grid_array[3] = grid_p->u[rk_flag][3]/grid_p->u[rk_flag][0];
}


void Advance::update_vis_array_from_grid_cell(Grid *grid_p, double *vis_array,
                                              int rk_flag) {
    for (int i = 0; i < 14; i++) {
        vis_array[i] = grid_p->Wmunu[rk_flag][i];
    }
    vis_array[14] = grid_p->pi_b[rk_flag];
    for (int i = 0; i < 4; i++) {
        vis_array[15+i] = grid_p->u[rk_flag][i];
    }
}

void Advance::update_vis_prev_tau_from_grid_cell(
                    Grid *grid_p, double *vis_array,int rk_flag) {
    for (int i = 0; i < 14; i++) {
        if (rk_flag == 0) {
            vis_array[i] = grid_p->prevWmunu[0][i];
        } else {
            vis_array[i] = grid_p->Wmunu[0][i];
        }
    }
    if (rk_flag == 0) {
        vis_array[14] = grid_p->prev_pi_b[0];
    } else {
        vis_array[14] = grid_p->pi_b[0];
    }
    for (int i = 0; i < 4; i++) {
        if (rk_flag == 0) {
            vis_array[15+i] = grid_p->prev_u[0][i];
        } else {
            vis_array[15+i] = grid_p->u[0][i];
        }
    }
}

void Advance::update_grid_cell_from_grid_array(Grid *grid_p,
                                               double *grid_array) {
    grid_p->epsilon = grid_array[0];
    grid_p->rhob = grid_array[4];
    grid_p->u[0][0] = 1./sqrt(1. - grid_array[1]*grid_array[1]
                                 - grid_array[2]*grid_array[2]
                                 - grid_array[3]*grid_array[3]);
    grid_p->u[0][1] = grid_p->u[0][0]*grid_array[1];
    grid_p->u[0][2] = grid_p->u[0][0]*grid_array[2];
    grid_p->u[0][3] = grid_p->u[0][0]*grid_array[3];
}

//! update results after RK evolution to grid_pt
void Advance::UpdateTJbRK(double *grid_array, Grid *grid_pt, int rk_flag) {
    int trk_flag = rk_flag + 1;
    if (rk_flag == 1) {
        trk_flag = 0;
    }

    if (rk_flag == 0) {
        grid_pt->epsilon_t = grid_array[0];
        grid_pt->rhob_t = grid_array[4];
    } else {
        grid_pt->epsilon = grid_array[0];
        grid_pt->rhob = grid_array[4];
    }

    grid_pt->u[trk_flag][0] = 1./sqrt(1. - grid_array[1]*grid_array[1]
                                         - grid_array[2]*grid_array[2]
                                         - grid_array[3]*grid_array[3]);
    grid_pt->u[trk_flag][1] = grid_pt->u[trk_flag][0]*grid_array[1];
    grid_pt->u[trk_flag][2] = grid_pt->u[trk_flag][0]*grid_array[2];
    grid_pt->u[trk_flag][3] = grid_pt->u[trk_flag][0]*grid_array[3];
}/* UpdateTJbRK */

//! this function reduce the size of shear stress tensor and bulk pressure
//! in the dilute region to stablize numerical simulations
int Advance::QuestRevert(double tau, Grid *grid_pt, int rk_flag,
                         double **vis_array, double **grid_array) {
    int idx = 0;
    int revert_flag = 0;
    const double energy_density_warning = 0.01;  // GeV/fm^3, T~100 MeV

    int trk_flag = rk_flag + 1;
    if (rk_flag == 1) {
        trk_flag = 0;
    }

    double eps_scale = 1.0;  // 1/fm^4
    double e_local = grid_array[idx][0];
    double factor = 300.*tanh(e_local/eps_scale);

    //double pi_00 = grid_pt->Wmunu[trk_flag][0];
    //double pi_01 = grid_pt->Wmunu[trk_flag][1];
    //double pi_02 = grid_pt->Wmunu[trk_flag][2];
    //double pi_03 = grid_pt->Wmunu[trk_flag][3];
    //double pi_11 = grid_pt->Wmunu[trk_flag][4];
    //double pi_12 = grid_pt->Wmunu[trk_flag][5];
    //double pi_13 = grid_pt->Wmunu[trk_flag][6];
    //double pi_22 = grid_pt->Wmunu[trk_flag][7];
    //double pi_23 = grid_pt->Wmunu[trk_flag][8];
    //double pi_33 = grid_pt->Wmunu[trk_flag][9];
    
    double pi_00 = vis_array[idx][0];
    double pi_01 = vis_array[idx][1];
    double pi_02 = vis_array[idx][2];
    double pi_03 = vis_array[idx][3];
    double pi_11 = vis_array[idx][4];
    double pi_12 = vis_array[idx][5];
    double pi_13 = vis_array[idx][6];
    double pi_22 = vis_array[idx][7];
    double pi_23 = vis_array[idx][8];
    double pi_33 = vis_array[idx][9];

    double pisize = (pi_00*pi_00 + pi_11*pi_11 + pi_22*pi_22 + pi_33*pi_33
                     - 2.*(pi_01*pi_01 + pi_02*pi_02 + pi_03*pi_03)
                     + 2.*(pi_12*pi_12 + pi_13*pi_13 + pi_23*pi_23));
  
    //double pi_local = grid_pt->pi_b[trk_flag];
    double pi_local = vis_array[idx][14];
    double bulksize = 3.*pi_local*pi_local;

    //double rhob_local = grid_pt->rhob;
    double rhob_local = grid_array[idx][4];
    double p_local = eos->get_pressure(e_local, rhob_local);
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
                //grid_pt->Wmunu[trk_flag][idx_1d] = (
                //    (rho_shear_max/rho_shear)
                //    *grid_pt->Wmunu[trk_flag][idx_1d]);
                vis_array[idx][idx_1d] = ((rho_shear_max/rho_shear)
                                          *vis_array[idx][idx_1d]);
                grid_pt->Wmunu[trk_flag][idx_1d] = vis_array[idx][idx_1d];
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
        //grid_pt->pi_b[trk_flag] = (
        //        (rho_bulk_max/rho_bulk)*grid_pt->pi_b[trk_flag]);
        vis_array[idx][14] = (rho_bulk_max/rho_bulk)*vis_array[idx][14];
        grid_pt->pi_b[trk_flag] = vis_array[idx][14];
        revert_flag = 1;
    }

    return(revert_flag);
}/* QuestRevert */


//! this function reduce the size of net baryon diffusion current
//! in the dilute region to stablize numerical simulations
int Advance::QuestRevert_qmu(double tau, Grid *grid_pt, int rk_flag,
                             double **vis_array, double **grid_array) {

    int idx = 0;
    int trk_flag = rk_flag + 1;
    if (rk_flag == 1) {
        trk_flag = 0;
    }
    int revert_flag = 0;
    const double energy_density_warning = 0.01;  // GeV/fm^3, T~100 MeV
    double eps_scale = 1.0;   // in 1/fm^4
    double e_local = grid_array[idx][0];
    double factor = 300.*tanh(e_local/eps_scale);

    double q_mu_local[4];
    for (int i = 0; i < 4; i++) {
        // copy the value from the grid
        int idx_1d = util->map_2d_idx_to_1d(4, i);
        //q_mu_local[i] = grid_pt->Wmunu[trk_flag][idx_1d];
        q_mu_local[i] = vis_array[idx][idx_1d];
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
            //grid_pt->Wmunu[trk_flag][idx_1d] = 0.0;
            vis_array[idx][idx_1d] = 0.0;
            grid_pt->Wmunu[trk_flag][idx_1d] = vis_array[idx][idx_1d];
        }
        revert_flag = 1;
    }

    // reduce the size of q^mu according to rhoB
    double rhob_local = grid_array[idx][4];
    double rho_q = sqrt(q_size/(rhob_local*rhob_local))/factor;
    double rho_q_max = 0.1;
    if (rho_q > rho_q_max) {
        if (e_local*hbarc > energy_density_warning) {
            printf("energy density = %lf, rhob = %lf -- |q/rhob| = %lf\n",
                   e_local*hbarc, rhob_local, rho_q);
        }
        for (int i = 0; i < 4; i++) {
            int idx_1d = util->map_2d_idx_to_1d(4, i);
            //grid_pt->Wmunu[trk_flag][idx_1d] =
            //                    (rho_q_max/rho_q)*q_mu_local[i];
            vis_array[idx][idx_1d] =rho_q_max/rho_q*q_mu_local[i];
            grid_pt->Wmunu[trk_flag][idx_1d] = vis_array[idx][idx_1d];
        }
        revert_flag = 1;
    }
    return(revert_flag);
}


//! This function computes the rhs array. It computes the spatial
//! derivatives of T^\mu\nu using the KT algorithm
void Advance::MakeDeltaQI(double tau, double **qi_array, double **qi_nbr_x,
                          double **qi_nbr_y, double **qi_nbr_eta,
                          int n_cell_eta, int n_cell_x, double **grid_array) {
    /* \partial_tau (tau Ttautau) + \partial_eta Tetatau 
            + \partial_x (tau Txtau) + \partial_y (tau Tytau) + Tetaeta = 0 */
    /* \partial_tau (tau Ttaueta) + \partial_eta Teteta 
            + \partial_x (tau Txeta) + \partial_y (tau Txeta) + Tetatau = 0 */
    /* \partial_tau (tau Txtau) + \partial_eta Tetax + \partial_x tau T_xx
            + \partial_y tau Tyx = 0 */
    
    //double check0 = 0.0;
    //for (int i = 0; i < 5; i++) {
    //    check0 += fabs(qi_array[0][i] - qi[i]);
    //}
    //cout << check0 << endl;

    // tau*Tmu0
    double rhs[5];
    for (int alpha = 0; alpha < 5; alpha++) {
        rhs[alpha] = 0.0;
    }

    double *qiphL = new double[5];
    double *qiphR = new double[5];
    double *qimhL = new double[5];
    double *qimhR = new double[5];
    
    double *grid_array_hL = new double[5];
    double *grid_array_hR = new double[5];
    
    for (int k = 0; k < n_cell_eta; k++) {
        for (int i = 0; i < n_cell_x; i++) {
            for (int j = 0; j < n_cell_x; j++) {
                int idx = j + i*n_cell_x + k*n_cell_x*n_cell_x;

                // implement Kurganov-Tadmor scheme
                // here computes the half way T^\tau\mu currents
                // x-direction
                int direc = 1;
                double tau_fac = tau;
                for (int alpha = 0; alpha < 5; alpha++) {
                    double gp = qi_array[idx][alpha];
                    double gphL = qi_array[idx][alpha];
                    double gmhR = qi_array[idx][alpha];
                    
                    double gphR, gmhL, gphR2, gmhL2;
                    if (i + 1 < n_cell_x) {
                        int idx_p_1 = j + (i+1)*n_cell_x + k*n_cell_x*n_cell_x;
                        gphR = qi_array[idx_p_1][alpha];
                    } else {
                        int idx_p_1 = 4*j + k*4*n_cell_x + 2;
                        gphR = qi_nbr_x[idx_p_1][alpha];
                    }
                    if (i - 1 > 0) {
                        int idx_m_1 = j + (i-1)*n_cell_x + k*n_cell_x*n_cell_x;
                        gmhL = qi_array[idx_m_1][alpha];
                    } else {
                        int idx_m_1 = 4*j + k*4*n_cell_x + 1;
                        gmhL = qi_nbr_x[idx_m_1][alpha];
                    }
                    if (i + 2 < n_cell_x) {
                        int idx_p_2 = j + (i+2)*n_cell_x + k*n_cell_x*n_cell_x;
                        gphR2 = qi_array[idx_p_2][alpha];
                    } else {
                        int idx_p_2 = 4*j + k*4*n_cell_x + 3;
                        gphR2 = qi_nbr_x[idx_p_2][alpha];
                    }
                    if (i - 2 > 0) {
                        int idx_m_2 = j + (i-2)*n_cell_x + k*n_cell_x*n_cell_x;
                        gmhL2 = qi_array[idx_m_2][alpha];
                    } else {
                        int idx_m_2 = 4*j + k*4*n_cell_x;
                        gmhL2 = qi_nbr_x[idx_m_2][alpha];
                    }

                    double fphL = 0.5*minmod->minmod_dx(gphR, gp, gmhL);
                    double fphR = -0.5*minmod->minmod_dx(gphR2, gphR, gp);
                    double fmhL = 0.5*minmod->minmod_dx(gp, gmhL, gmhL2);
                    double fmhR = -0.5*minmod->minmod_dx(gphR, gp, gmhL);
                    qiphL[alpha] = gphL + fphL;
                    qiphR[alpha] = gphR + fphR;
                    qimhL[alpha] = gmhL + fmhL;
                    qimhR[alpha] = gmhR + fmhR;
                }
                // for each direction, reconstruct half-way cells
                // reconstruct e, rhob, and u[4] for half way cells
                int flag = reconst_ptr->ReconstIt_shell(
                                grid_array_hL, tau, qiphL, grid_array[idx]);
                double aiphL = MaxSpeed(tau, direc, grid_array_hL);

                flag *= reconst_ptr->ReconstIt_shell(
                                grid_array_hR, tau, qiphR, grid_array[idx]);
                double aiphR = MaxSpeed(tau, direc, grid_array_hR);
                double aiph = maxi(aiphL, aiphR);
                for (int alpha = 0; alpha < 5; alpha++) {
                    double FiphL = tau_fac*get_TJb_new(grid_array_hL,
                                                       alpha, direc);
                    double FiphR = tau_fac*get_TJb_new(grid_array_hR,
                                                        alpha, direc);
                    // KT: H_{j+1/2} = (f(u^+_{j+1/2}) + f(u^-_{j+1/2})/2
                    //              - a_{j+1/2}(u_{j+1/2}^+ - u^-_{j+1/2})/2
                    double Fiph = 0.5*((FiphL + FiphR)
                                        - aiph*(qiphR[alpha] - qiphL[alpha]));

                    rhs[alpha] = -Fiph/DATA_ptr->delta_x*DATA_ptr->delta_tau;
                }

                flag *= reconst_ptr->ReconstIt_shell(grid_array_hL, tau,
                                                     qimhL, grid_array[idx]);
                double aimhL = MaxSpeed(tau, direc, grid_array_hL);

                flag *= reconst_ptr->ReconstIt_shell(grid_array_hR, tau,
                                                     qimhR, grid_array[idx]);
                double aimhR = MaxSpeed(tau, direc, grid_array_hR);
                double aimh = maxi(aimhL, aimhR);

                for (int alpha = 0; alpha < 5; alpha++) {
                    double FimhL = tau_fac*get_TJb_new(grid_array_hL,
                                                       alpha, direc);
                    double FimhR = tau_fac*get_TJb_new(grid_array_hR,
                                                       alpha, direc);
                    // KT: H_{j+1/2} = (f(u^+_{j+1/2}) + f(u^-_{j+1/2})/2
                    //              - a_{j+1/2}(u_{j+1/2}^+ - u^-_{j+1/2})/2
                    double Fimh = 0.5*((FimhL + FimhR)
                                        - aimh*(qimhR[alpha] - qimhL[alpha]));
                    rhs[alpha] += Fimh/DATA_ptr->delta_x*DATA_ptr->delta_tau;
                }
                
                // y-direction
                direc = 2;
                tau_fac = tau;
                for (int alpha = 0; alpha < 5; alpha++) {
                    double gp = qi_array[idx][alpha];
                    double gphL = qi_array[idx][alpha];
                    double gmhR = qi_array[idx][alpha];

                    double gphR, gmhL, gphR2, gmhL2;
                    if (j + 1 < n_cell_x) {
                        int idx_p_1 = j + 1 + i*n_cell_x + k*n_cell_x*n_cell_x;
                        gphR = qi_array[idx_p_1][alpha];
                    } else {
                        int idx_p_1 = 4*i + 4*k*n_cell_x + 2;
                        gphR = qi_nbr_y[idx_p_1][alpha];
                    }
                    if (j - 1 > 0) {
                        int idx_m_1 = j - 1 + i*n_cell_x + k*n_cell_x*n_cell_x;
                        gmhL = qi_array[idx_m_1][alpha];
                    } else {
                        int idx_m_1 = 4*i + 4*k*n_cell_x + 1;
                        gmhL = qi_nbr_y[idx_m_1][alpha];
                    }
                    if (j + 2 < n_cell_x) {
                        int idx_p_2 = j + 2 + i*n_cell_x + k*n_cell_x*n_cell_x;
                        gphR2 = qi_array[idx_p_2][alpha];
                    } else {
                        int idx_p_2 = 4*i + 4*k*n_cell_x + 3;
                        gphR2 = qi_nbr_y[idx_p_2][alpha];
                    }
                    if (j - 2 > 0) {
                        int idx_m_2 = j - 2 + i*n_cell_x + k*n_cell_x*n_cell_x;
                        gmhL2 = qi_array[idx_m_2][alpha];
                    } else {
                        int idx_m_2 = 4*i + 4*k*n_cell_x;
                        gmhL2 = qi_nbr_y[idx_m_2][alpha];
                    }

                    double fphL = 0.5*minmod->minmod_dx(gphR, gp, gmhL);
                    double fphR = -0.5*minmod->minmod_dx(gphR2, gphR, gp);
                    double fmhL = 0.5*minmod->minmod_dx(gp, gmhL, gmhL2);
                    double fmhR = -0.5*minmod->minmod_dx(gphR, gp, gmhL);
                    qiphL[alpha] = gphL + fphL;
                    qiphR[alpha] = gphR + fphR;
                    qimhL[alpha] = gmhL + fmhL;
                    qimhR[alpha] = gmhR + fmhR;
                }
                // for each direction, reconstruct half-way cells
                // reconstruct e, rhob, and u[4] for half way cells
                flag = reconst_ptr->ReconstIt_shell(
                                grid_array_hL, tau, qiphL, grid_array[idx]);
                aiphL = MaxSpeed(tau, direc, grid_array_hL);

                flag *= reconst_ptr->ReconstIt_shell(
                                grid_array_hR, tau, qiphR, grid_array[idx]);
                aiphR = MaxSpeed(tau, direc, grid_array_hR);
                aiph = maxi(aiphL, aiphR);
                for (int alpha = 0; alpha < 5; alpha++) {
                    double FiphL = tau_fac*get_TJb_new(grid_array_hL,
                                                       alpha, direc);
                    double FiphR = tau_fac*get_TJb_new(grid_array_hR,
                                                        alpha, direc);
                    // KT: H_{j+1/2} = (f(u^+_{j+1/2}) + f(u^-_{j+1/2})/2
                    //              - a_{j+1/2}(u_{j+1/2}^+ - u^-_{j+1/2})/2
                    double Fiph = 0.5*((FiphL + FiphR)
                                        - aiph*(qiphR[alpha] - qiphL[alpha]));

                    rhs[alpha] -= Fiph/DATA_ptr->delta_y*DATA_ptr->delta_tau;
                }

                flag *= reconst_ptr->ReconstIt_shell(grid_array_hL, tau,
                                                     qimhL, grid_array[idx]);
                aimhL = MaxSpeed(tau, direc, grid_array_hL);

                flag *= reconst_ptr->ReconstIt_shell(grid_array_hR, tau,
                                                     qimhR, grid_array[idx]);
                aimhR = MaxSpeed(tau, direc, grid_array_hR);
                aimh = maxi(aimhL, aimhR);

                for (int alpha = 0; alpha < 5; alpha++) {
                    double FimhL = tau_fac*get_TJb_new(grid_array_hL,
                                                       alpha, direc);
                    double FimhR = tau_fac*get_TJb_new(grid_array_hR,
                                                       alpha, direc);
                    // KT: H_{j+1/2} = (f(u^+_{j+1/2}) + f(u^-_{j+1/2})/2
                    //              - a_{j+1/2}(u_{j+1/2}^+ - u^-_{j+1/2})/2
                    double Fimh = 0.5*((FimhL + FimhR)
                                        - aimh*(qimhR[alpha] - qimhL[alpha]));
                    rhs[alpha] += Fimh/DATA_ptr->delta_y*DATA_ptr->delta_tau;
                }
                
                // eta-direction
                direc = 3;
                tau_fac = 1.0;
                for (int alpha = 0; alpha < 5; alpha++) {
                    double gp = qi_array[idx][alpha];
                    double gphL = qi_array[idx][alpha];
                    double gmhR = qi_array[idx][alpha];

                    double gphR, gmhL, gphR2, gmhL2;
                    if (k + 1 < n_cell_eta) {
                        int idx_p_1 = j + i*n_cell_x + (k+1)*n_cell_x*n_cell_x;
                        gphR = qi_array[idx_p_1][alpha];
                    } else {
                        int idx_p_1 = 4*i + 4*j*n_cell_x + 2;
                        gphR = qi_nbr_eta[idx_p_1][alpha];
                    }
                    if (k - 1 > 0) {
                        int idx_m_1 = j + i*n_cell_x + (k-1)*n_cell_x*n_cell_x;
                        gmhL = qi_array[idx_m_1][alpha];
                    } else {
                        int idx_m_1 = 4*i + 4*j*n_cell_x + 1;
                        gmhL = qi_nbr_eta[idx_m_1][alpha];
                    }
                    if (k + 2 < n_cell_eta) {
                        int idx_p_2 = j + i*n_cell_x + (k+2)*n_cell_x*n_cell_x;
                        gphR2 = qi_array[idx_p_2][alpha];
                    } else {
                        int idx_p_2 = 4*i + 4*j*n_cell_x + 3;
                        gphR2 = qi_nbr_eta[idx_p_2][alpha];
                    }
                    if (k - 2 > 0) {
                        int idx_m_2 = j + i*n_cell_x + (k-2)*n_cell_x*n_cell_x;
                        gmhL2 = qi_array[idx_m_2][alpha];
                    } else {
                        int idx_m_2 = 4*i + 4*j*n_cell_x;
                        gmhL2 = qi_nbr_eta[idx_m_2][alpha];
                    }

                    double fphL = 0.5*minmod->minmod_dx(gphR, gp, gmhL);
                    double fphR = -0.5*minmod->minmod_dx(gphR2, gphR, gp);
                    double fmhL = 0.5*minmod->minmod_dx(gp, gmhL, gmhL2);
                    double fmhR = -0.5*minmod->minmod_dx(gphR, gp, gmhL);
                    qiphL[alpha] = gphL + fphL;
                    qiphR[alpha] = gphR + fphR;
                    qimhL[alpha] = gmhL + fmhL;
                    qimhR[alpha] = gmhR + fmhR;
                }
                // for each direction, reconstruct half-way cells
                // reconstruct e, rhob, and u[4] for half way cells
                flag = reconst_ptr->ReconstIt_shell(
                                grid_array_hL, tau, qiphL, grid_array[idx]);
                aiphL = MaxSpeed(tau, direc, grid_array_hL);

                flag *= reconst_ptr->ReconstIt_shell(
                                grid_array_hR, tau, qiphR, grid_array[idx]);
                aiphR = MaxSpeed(tau, direc, grid_array_hR);
                aiph = maxi(aiphL, aiphR);
                for (int alpha = 0; alpha < 5; alpha++) {
                    double FiphL = tau_fac*get_TJb_new(grid_array_hL,
                                                       alpha, direc);
                    double FiphR = tau_fac*get_TJb_new(grid_array_hR,
                                                        alpha, direc);
                    // KT: H_{j+1/2} = (f(u^+_{j+1/2}) + f(u^-_{j+1/2})/2
                    //              - a_{j+1/2}(u_{j+1/2}^+ - u^-_{j+1/2})/2
                    double Fiph = 0.5*((FiphL + FiphR)
                                        - aiph*(qiphR[alpha] - qiphL[alpha]));

                    rhs[alpha] -= Fiph/DATA_ptr->delta_eta*DATA_ptr->delta_tau;
                }

                flag *= reconst_ptr->ReconstIt_shell(grid_array_hL, tau,
                                                     qimhL, grid_array[idx]);
                aimhL = MaxSpeed(tau, direc, grid_array_hL);

                flag *= reconst_ptr->ReconstIt_shell(grid_array_hR, tau,
                                                     qimhR, grid_array[idx]);
                aimhR = MaxSpeed(tau, direc, grid_array_hR);
                aimh = maxi(aimhL, aimhR);

                for (int alpha = 0; alpha < 5; alpha++) {
                    double FimhL = tau_fac*get_TJb_new(grid_array_hL,
                                                       alpha, direc);
                    double FimhR = tau_fac*get_TJb_new(grid_array_hR,
                                                       alpha, direc);
                    // KT: H_{j+1/2} = (f(u^+_{j+1/2}) + f(u^-_{j+1/2})/2
                    //              - a_{j+1/2}(u_{j+1/2}^+ - u^-_{j+1/2})/2
                    double Fimh = 0.5*((FimhL + FimhR)
                                        - aimh*(qimhR[alpha] - qimhL[alpha]));
                    rhs[alpha] += Fimh/DATA_ptr->delta_eta*DATA_ptr->delta_tau;
                }

                // geometric terms
                rhs[0] -= (get_TJb_new(grid_array[idx], 3, 3)
                           *DATA_ptr->delta_tau);
                rhs[3] -= (get_TJb_new(grid_array[idx], 3, 0)
                           *DATA_ptr->delta_tau);
                
                for (int i = 0; i < 5; i++) {
                    qi_array[idx][i] += rhs[i];
                }
            }
        }
    }

    // clean up
    delete[] qiphL;
    delete[] qiphR;
    delete[] qimhL;
    delete[] qimhR;

    delete[] grid_array_hL;
    delete[] grid_array_hR;
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
  
    double vs2 = eos->get_cs2(eps, rhob);

    double den = utau2*(1. - vs2) + vs2;
    double num_temp_sqrt = (ut2mux2 - (ut2mux2 - 1.)*vs2)*vs2;
    double num;
    if (num_temp_sqrt >= 0) {
        num = utau*ux*(1. - vs2) + sqrt(num_temp_sqrt);
    } else {
        double dpde = eos->p_e_func(eps, rhob);
        double p = eos->get_pressure(eps, rhob);
        double h = p + eps;
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

    return(f);
}

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
        double pressure = eos->get_pressure(e, rhob);
        double T_munu = (e + pressure)*u_mu*u_nu + pressure*gfac;
        return(T_munu);
    } else {
        return(0.0);
    }
}


//! This function computes the vector [T^\tau\mu, J^\tau] from the
//! grid_array [e, v^i, rhob]
void Advance::get_qmu_from_grid_array(double tau, double *qi,
                                      double *grid_array) {
    double rhob = grid_array[4];
    double e = grid_array[0];
    double pressure = eos->get_pressure(e, rhob);
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

void Advance::update_grid_cell(double **grid_array, Grid ***arena, int rk_flag,
                               int ieta, int ix, int iy,
                               int n_cell_eta, int n_cell_x) {
    for (int k = 0; k < n_cell_eta; k++) {
        for (int i = 0; i < n_cell_x; i++) {
            for (int j = 0; j < n_cell_x; j++) {
                int idx = j + i*n_cell_x + k*n_cell_x*n_cell_x;
                UpdateTJbRK(grid_array[idx], &arena[ieta+k][ix+i][iy+j],
                            rk_flag);
            }
        }
    }
}           
