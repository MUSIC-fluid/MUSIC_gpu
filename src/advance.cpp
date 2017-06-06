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
        double tau, Field *hydro_fields, int rk_flag, int ieta, int ix, int iy,
        int n_cell_eta, int n_cell_x, int n_cell_y,
        double **qi_array, double **qi_nbr_x,
        double **qi_nbr_y, double **qi_nbr_eta,
        double **qi_rk0, double **grid_array, double *grid_array_temp) {

    double tau_rk;
    if (rk_flag == 0) {
        tau_rk = tau;
    } else {
        tau_rk = tau + DATA_ptr->delta_tau;
    }

    int field_idx;
    int field_ny = DATA_ptr->ny + 1;
    int field_nperp = (DATA_ptr->ny + 1)*(DATA_ptr->nx + 1);
    // first build qi cube n_cell_x*n_cell_x*n_cell_eta
    for (int k = 0; k < n_cell_eta; k++) {
        int idx_ieta = min(ieta + k, DATA_ptr->neta - 1);
        for (int i = 0; i < n_cell_x; i++) {
            int idx_ix = min(ix + i, DATA_ptr->nx);
            for (int j = 0; j < n_cell_y; j++) {
                int idx_iy = min(iy + j, DATA_ptr->ny);
                int idx = j + n_cell_y*i + n_cell_x*n_cell_y*k;
                field_idx = (idx_iy + idx_ix*field_ny + idx_ieta*field_nperp);
                update_grid_array_from_field(hydro_fields, field_idx,
                                             grid_array[idx], rk_flag);
                get_qmu_from_grid_array(tau_rk, qi_array[idx],
                                        grid_array[idx]);
            }
        }
    }

    if (rk_flag == 1) {
        for (int k = 0; k < n_cell_eta; k++) {
            int idx_ieta = min(ieta + k, DATA_ptr->neta - 1);
            for (int i = 0; i < n_cell_x; i++) {
                int idx_ix = min(ix + i, DATA_ptr->nx);
                for (int j = 0; j < n_cell_y; j++) {
                    int idx_iy = min(iy + j, DATA_ptr->ny);
                    int idx = j + n_cell_y*i + n_cell_x*n_cell_y*k;
                    field_idx = (idx_iy + idx_ix*field_ny
                                 + idx_ieta*field_nperp);
                    update_grid_array_from_field(hydro_fields, field_idx,
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
        for (int i = 0; i < n_cell_y; i++) {
            int idx_iy = min(iy + i, DATA_ptr->ny);
            int idx = 4*i + 4*n_cell_y*k;

            int idx_m_2 = max(0, ix - 2);
            int idx_m_1 = max(0, ix - 1);
            int idx_p_1 = min(ix + n_cell_x, DATA_ptr->nx);
            int idx_p_2 = min(ix + n_cell_x + 1, DATA_ptr->nx);

            field_idx = (idx_iy + idx_m_2*field_ny + idx_ieta*field_nperp);
            update_grid_array_from_field(hydro_fields, field_idx,
                                         grid_array_temp, rk_flag);
            get_qmu_from_grid_array(tau_rk, qi_nbr_x[idx], grid_array_temp);
            field_idx = (idx_iy + idx_m_1*field_ny + idx_ieta*field_nperp);
            update_grid_array_from_field(hydro_fields, field_idx,
                                         grid_array_temp, rk_flag);
            get_qmu_from_grid_array(tau_rk, qi_nbr_x[idx+1], grid_array_temp);
            field_idx = (idx_iy + idx_p_1*field_ny + idx_ieta*field_nperp);
            update_grid_array_from_field(hydro_fields, field_idx,
                                         grid_array_temp, rk_flag);
            get_qmu_from_grid_array(tau_rk, qi_nbr_x[idx+2], grid_array_temp);
            field_idx = (idx_iy + idx_p_2*field_ny + idx_ieta*field_nperp);
            update_grid_array_from_field(hydro_fields, field_idx,
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
            int idx_p_1 = min(iy + n_cell_y, DATA_ptr->ny);
            int idx_p_2 = min(iy + n_cell_y + 1, DATA_ptr->ny);

            field_idx = (idx_m_2 + idx_ix*field_ny + idx_ieta*field_nperp);
            update_grid_array_from_field(hydro_fields, field_idx,
                                         grid_array_temp, rk_flag);
            get_qmu_from_grid_array(tau_rk, qi_nbr_y[idx], grid_array_temp);
            field_idx = (idx_m_1 + idx_ix*field_ny + idx_ieta*field_nperp);
            update_grid_array_from_field(hydro_fields, field_idx,
                                         grid_array_temp, rk_flag);
            get_qmu_from_grid_array(tau_rk, qi_nbr_y[idx+1], grid_array_temp);
            field_idx = (idx_p_1 + idx_ix*field_ny + idx_ieta*field_nperp);
            update_grid_array_from_field(hydro_fields, field_idx,
                                         grid_array_temp, rk_flag);
            get_qmu_from_grid_array(tau_rk, qi_nbr_y[idx+2], grid_array_temp);
            field_idx = (idx_p_2 + idx_ix*field_ny + idx_ieta*field_nperp);
            update_grid_array_from_field(hydro_fields, field_idx,
                                         grid_array_temp, rk_flag);
            get_qmu_from_grid_array(tau_rk, qi_nbr_y[idx+3], grid_array_temp);
        }
    }

    // eta-direction
    for (int i = 0; i < n_cell_x; i++) {
        int idx_ix = min(ix + i, DATA_ptr->nx);
        for (int k = 0; k < n_cell_y; k++) {
            int idx_iy = min(iy + k, DATA_ptr->ny);
            int idx = 4*k + 4*n_cell_y*i;

            int idx_m_2 = max(0, ieta - 2);
            int idx_m_1 = max(0, ieta - 1);
            int idx_p_1 = min(ieta + n_cell_eta, DATA_ptr->neta-1);
            int idx_p_2 = min(ieta + n_cell_eta + 1, DATA_ptr->neta-1);

            field_idx = (idx_iy + idx_ix*field_ny + idx_m_2*field_nperp);
            update_grid_array_from_field(hydro_fields, field_idx,
                                         grid_array_temp, rk_flag);
            get_qmu_from_grid_array(tau_rk, qi_nbr_eta[idx], grid_array_temp);
            field_idx = (idx_iy + idx_ix*field_ny + idx_m_1*field_nperp);
            update_grid_array_from_field(hydro_fields, field_idx,
                                         grid_array_temp, rk_flag);
            get_qmu_from_grid_array(tau_rk, qi_nbr_eta[idx+1], grid_array_temp);
            field_idx = (idx_iy + idx_ix*field_ny + idx_p_1*field_nperp);
            update_grid_array_from_field(hydro_fields, field_idx,
                                         grid_array_temp, rk_flag);
            get_qmu_from_grid_array(tau_rk, qi_nbr_eta[idx+2], grid_array_temp);
            field_idx = (idx_iy + idx_ix*field_ny + idx_p_2*field_nperp);
            update_grid_array_from_field(hydro_fields, field_idx,
                                         grid_array_temp, rk_flag);
            get_qmu_from_grid_array(tau_rk, qi_nbr_eta[idx+3], grid_array_temp);
        }
    }
}

void Advance::prepare_vis_array(
        Field *hydro_fields, int rk_flag, int ieta, int ix, int iy,
        int n_cell_eta, int n_cell_x, int n_cell_y,
        double **vis_array, double **vis_nbr_tau,
        double **vis_nbr_x, double **vis_nbr_y, double **vis_nbr_eta) {

    int field_idx;
    int field_ny = DATA_ptr->ny + 1;
    int field_nperp = (DATA_ptr->ny + 1)*(DATA_ptr->nx + 1);

    // first build qi cube n_cell_x*n_cell_x*n_cell_eta
    for (int k = 0; k < n_cell_eta; k++) {
        int idx_ieta = min(ieta + k, DATA_ptr->neta - 1);
        for (int i = 0; i < n_cell_x; i++) {
            int idx_ix = min(ix + i, DATA_ptr->nx);
            for (int j = 0; j < n_cell_y; j++) {
                int idx_iy = min(iy + j, DATA_ptr->ny);
                int idx = j + n_cell_y*i + n_cell_x*n_cell_y*k;

                field_idx = (idx_iy + idx_ix*field_ny + idx_ieta*field_nperp);
                update_vis_array_from_field(hydro_fields, field_idx,
                                            vis_array[idx], rk_flag);
                update_vis_prev_tau_from_field(hydro_fields, field_idx,
                                               vis_nbr_tau[idx], rk_flag);
            }
        }
    }

    // now build neighbouring cells
    // x-direction
    for (int k = 0; k < n_cell_eta; k++) {
        int idx_ieta = min(ieta + k, DATA_ptr->neta - 1);
        for (int i = 0; i < n_cell_y; i++) {
            int idx_iy = min(iy + i, DATA_ptr->ny);
            int idx = 4*i + 4*n_cell_y*k;

            int idx_m_2 = max(0, ix - 2);
            int idx_m_1 = max(0, ix - 1);
            int idx_p_1 = min(ix + n_cell_x, DATA_ptr->nx);
            int idx_p_2 = min(ix + n_cell_x + 1, DATA_ptr->nx);

            field_idx = (idx_iy + idx_m_2*field_ny + idx_ieta*field_nperp);
            update_vis_array_from_field(hydro_fields, field_idx,
                                        vis_nbr_x[idx], rk_flag);
            field_idx = (idx_iy + idx_m_1*field_ny + idx_ieta*field_nperp);
            update_vis_array_from_field(hydro_fields, field_idx,
                                        vis_nbr_x[idx+1], rk_flag);
            field_idx = (idx_iy + idx_p_1*field_ny + idx_ieta*field_nperp);
            update_vis_array_from_field(hydro_fields, field_idx,
                                        vis_nbr_x[idx+2], rk_flag);
            field_idx = (idx_iy + idx_p_2*field_ny + idx_ieta*field_nperp);
            update_vis_array_from_field(hydro_fields, field_idx,
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
            int idx_p_1 = min(iy + n_cell_y, DATA_ptr->ny);
            int idx_p_2 = min(iy + n_cell_y + 1, DATA_ptr->ny);

            field_idx = (idx_m_2 + idx_ix*field_ny + idx_ieta*field_nperp);
            update_vis_array_from_field(hydro_fields, field_idx,
                                        vis_nbr_y[idx], rk_flag);
            field_idx = (idx_m_1 + idx_ix*field_ny + idx_ieta*field_nperp);
            update_vis_array_from_field(hydro_fields, field_idx,
                                        vis_nbr_y[idx+1], rk_flag);
            field_idx = (idx_p_1 + idx_ix*field_ny + idx_ieta*field_nperp);
            update_vis_array_from_field(hydro_fields, field_idx,
                                        vis_nbr_y[idx+2], rk_flag);
            field_idx = (idx_p_2 + idx_ix*field_ny + idx_ieta*field_nperp);
            update_vis_array_from_field(hydro_fields, field_idx,
                                        vis_nbr_y[idx+3], rk_flag);
        }
    }

    // eta-direction
    for (int i = 0; i < n_cell_x; i++) {
        int idx_ix = min(ix + i, DATA_ptr->nx);
        for (int k = 0; k < n_cell_y; k++) {
            int idx_iy = min(iy + k, DATA_ptr->ny);
            int idx = 4*k + 4*n_cell_y*i;

            int idx_m_2 = max(0, ieta - 2);
            int idx_m_1 = max(0, ieta - 1);
            int idx_p_1 = min(ieta + n_cell_eta, DATA_ptr->neta - 1);
            int idx_p_2 = min(ieta + n_cell_eta + 1, DATA_ptr->neta - 1);

            field_idx = (idx_iy + idx_ix*field_ny + idx_m_2*field_nperp);
            update_vis_array_from_field(hydro_fields, field_idx,
                                        vis_nbr_eta[idx], rk_flag);
            field_idx = (idx_iy + idx_ix*field_ny + idx_m_1*field_nperp);
            update_vis_array_from_field(hydro_fields, field_idx,
                                        vis_nbr_eta[idx+1], rk_flag);
            field_idx = (idx_iy + idx_ix*field_ny + idx_p_1*field_nperp);
            update_vis_array_from_field(hydro_fields, field_idx,
                                        vis_nbr_eta[idx+2], rk_flag);
            field_idx = (idx_iy + idx_ix*field_ny + idx_p_2*field_nperp);
            update_vis_array_from_field(hydro_fields, field_idx,
                                        vis_nbr_eta[idx+3], rk_flag);
        }
    }
}
                        
void Advance::prepare_velocity_array(double tau_rk, Field *hydro_fields,
                                     int ieta, int ix, int iy, int rk_flag,
                                     int n_cell_eta, int n_cell_x,
                                     int n_cell_y, double **velocity_array, 
                                     double **grid_array,
                                     double **vis_array_new,
                                     double *grid_array_temp) {
    int trk_flag = 1;
    if (rk_flag == 1) {
        trk_flag = 0;
    }

    int field_idx;
    int field_ny = DATA_ptr->ny + 1;
    int field_nperp = (DATA_ptr->ny + 1)*(DATA_ptr->nx + 1);

    for (int k = 0; k < n_cell_eta; k++) {
        int idx_ieta = min(ieta + k, DATA_ptr->neta - 1);
        for (int i = 0; i < n_cell_x; i++) {
            int idx_ix = min(ix + i, DATA_ptr->nx);
            for (int j = 0; j < n_cell_y; j++) {
                int idx_iy = min(iy + j, DATA_ptr->ny);
                int idx = j + n_cell_y*i + n_cell_x*n_cell_y*k;

                field_idx = (idx_iy + idx_ix*field_ny + idx_ieta*field_nperp);
                update_grid_array_from_field(hydro_fields, field_idx,
                                             grid_array[idx], rk_flag);

                update_grid_array_from_field(hydro_fields, field_idx,
                                             grid_array_temp, trk_flag);
                double u0 = 1./sqrt(1. - grid_array_temp[1]*grid_array_temp[1]
                                       - grid_array_temp[2]*grid_array_temp[2]
                                       - grid_array_temp[3]*grid_array_temp[3]
                                    );
                for (int alpha = 0; alpha < 15; alpha++) {
                    vis_array_new[idx][alpha] = 0.0;
                }
                vis_array_new[idx][15] = u0;
                vis_array_new[idx][16] = u0*grid_array_temp[1];
                vis_array_new[idx][17] = u0*grid_array_temp[2];
                vis_array_new[idx][18] = u0*grid_array_temp[3];

                velocity_array[idx][0] = (
                        u_derivative_ptr->calculate_expansion_rate_1(
                                    tau_rk, hydro_fields, field_idx, rk_flag));
                u_derivative_ptr->calculate_Du_supmu_1(
                        tau_rk, hydro_fields, field_idx, rk_flag,
                        velocity_array[idx]);
                u_derivative_ptr->calculate_velocity_shear_tensor_2(
                            tau_rk, hydro_fields, field_idx, rk_flag,
                            velocity_array[idx]);
                for (int alpha = 0; alpha < 4; alpha++) {
                    velocity_array[idx][16+alpha] = (
                                    hydro_fields->dUsup[field_idx][16+alpha]);
                }
            }
        }
    }
}


// evolve Runge-Kutta step in tau
int Advance::AdvanceIt(double tau, InitData *DATA, Field *hydro_fields,
                       int rk_flag) {
    int n_cell_eta = 1;
    int n_cell_x = 1;
    int n_cell_y = 1;
    int ieta;
    for (ieta = 0; ieta < grid_neta; ieta += n_cell_eta) {
        int ix;
        #pragma omp parallel private(ix)
        {
            #pragma omp for
            for (ix = 0; ix <= grid_nx; ix += n_cell_x) {
                double **qi_array = new double* [n_cell_x*n_cell_y*n_cell_eta];
                double **qi_array_new = new double* [n_cell_x*n_cell_y*n_cell_eta];
                double **qi_rk0 = new double* [n_cell_x*n_cell_y*n_cell_eta];
                double **grid_array = (
                                new double* [n_cell_x*n_cell_y*n_cell_eta]);
                for (int i = 0; i < n_cell_x*n_cell_y*n_cell_eta; i++) {
                    qi_array[i] = new double[5];
                    qi_array_new[i] = new double[5];
                    qi_rk0[i] = new double[5];
                    grid_array[i] = new double[5];
                }
                double **qi_nbr_x = new double* [4*n_cell_y*n_cell_eta];
                double **qi_nbr_y = new double* [4*n_cell_x*n_cell_eta];
                for (int i = 0; i < 4*n_cell_x*n_cell_eta; i++) {
                    qi_nbr_y[i] = new double[5];
                }
                for (int i = 0; i < 4*n_cell_y*n_cell_eta; i++) {
                    qi_nbr_x[i] = new double[5];
                }
                double **qi_nbr_eta = new double* [4*n_cell_x*n_cell_y];
                for (int i = 0; i < 4*n_cell_x*n_cell_y; i++) {
                    qi_nbr_eta[i] = new double[5];
                }
                double **vis_array =
                                new double* [n_cell_x*n_cell_y*n_cell_eta];
                double **vis_array_new =
                                new double* [n_cell_x*n_cell_y*n_cell_eta];
                double **vis_nbr_tau =
                                new double* [n_cell_x*n_cell_y*n_cell_eta];
                double **velocity_array =
                                new double* [n_cell_x*n_cell_y*n_cell_eta];
                for (int i = 0; i < n_cell_x*n_cell_y*n_cell_eta; i++) {
                    vis_array[i] = new double[19];
                    vis_array_new[i] = new double[19];
                    vis_nbr_tau[i] = new double[19];
                    velocity_array[i] = new double[20];
                }
                double **vis_nbr_x = new double* [4*n_cell_y*n_cell_eta];
                double **vis_nbr_y = new double* [4*n_cell_x*n_cell_eta];
                for (int i = 0; i < 4*n_cell_x*n_cell_eta; i++) {
                    vis_nbr_y[i] = new double[19];
                }
                for (int i = 0; i < 4*n_cell_y*n_cell_eta; i++) {
                    vis_nbr_x[i] = new double[19];
                }
                double **vis_nbr_eta = new double* [4*n_cell_x*n_cell_y];
                for (int i = 0; i < 4*n_cell_x*n_cell_y; i++) {
                    vis_nbr_eta[i] = new double[19];
                }

                double *grid_array_temp = new double[5];
                double *rhs = new double[5];
                double *qiphL = new double[5];
                double *qiphR = new double[5];
                double *qimhL = new double[5];
                double *qimhR = new double[5];
                double *grid_array_hL = new double[5];
                double *grid_array_hR = new double[5];

                for (int iy = 0; iy <= grid_ny; iy += n_cell_y) {
                    prepare_qi_array(tau, hydro_fields, rk_flag, ieta, ix, iy,
                                     n_cell_eta, n_cell_x, n_cell_y, qi_array,
                                     qi_nbr_x, qi_nbr_y, qi_nbr_eta,
                                     qi_rk0, grid_array, grid_array_temp);
                    // viscous source terms
                    prepare_vis_array(hydro_fields, rk_flag, ieta, ix, iy,
                                      n_cell_eta, n_cell_x, n_cell_y,
                                      vis_array, vis_nbr_tau, vis_nbr_x,
                                      vis_nbr_y, vis_nbr_eta);

                    FirstRKStepT(tau, rk_flag,
                                 qi_array, qi_nbr_x, qi_nbr_y, qi_nbr_eta,
                                 n_cell_eta, n_cell_x, n_cell_y,
                                 vis_array, vis_nbr_tau,
                                 vis_nbr_x, vis_nbr_y, vis_nbr_eta,
                                 qi_rk0, qi_array_new, grid_array,
                                 rhs, qiphL, qiphR, qimhL, qimhR,
                                 grid_array_hL, grid_array_hR);

                    update_grid_cell(grid_array, hydro_fields, rk_flag, ieta, ix, iy,
                                     n_cell_eta, n_cell_x, n_cell_y);

                    if (DATA_ptr->viscosity_flag == 1) {
                        double tau_rk = tau;
                        if (rk_flag == 1) {
                            tau_rk = tau + DATA_ptr->delta_tau;
                        }

                        prepare_velocity_array(tau_rk, hydro_fields,
                                               ieta, ix, iy,
                                               rk_flag, n_cell_eta, n_cell_x,
                                               n_cell_y, velocity_array,
                                               grid_array, vis_array_new,
                                               grid_array_temp);

                        FirstRKStepW(tau, rk_flag, n_cell_eta, n_cell_x,
                                     n_cell_y, vis_array, vis_nbr_tau,
                                     vis_nbr_x, vis_nbr_y, vis_nbr_eta,
                                     velocity_array, grid_array,
                                     vis_array_new);

                        update_grid_cell_viscous(vis_array_new, hydro_fields, rk_flag,
                                                 ieta, ix, iy, n_cell_eta,
                                                 n_cell_x, n_cell_y);
                    }
                }

                //clean up
                delete[] grid_array_temp;
                delete[] rhs;
                delete[] qiphL;
                delete[] qiphR;
                delete[] qimhL;
                delete[] qimhR;
                delete[] grid_array_hL;
                delete[] grid_array_hR;
                for (int i = 0; i < n_cell_x*n_cell_y*n_cell_eta; i++) {
                    delete[] qi_array[i];
                    delete[] qi_array_new[i];
                    delete[] qi_rk0[i];
                    delete[] grid_array[i];
                    delete[] vis_array[i];
                    delete[] vis_nbr_tau[i];
                    delete[] velocity_array[i];
                    delete[] vis_array_new[i];
                }
                delete[] qi_array;
                delete[] qi_array_new;
                delete[] qi_rk0;
                delete[] grid_array;
                delete[] vis_array;
                delete[] vis_nbr_tau;
                delete[] velocity_array;
                delete[] vis_array_new;
                for (int i = 0; i < 4*n_cell_x*n_cell_eta; i++) {
                    delete[] qi_nbr_y[i];
                    delete[] vis_nbr_y[i];
                }
                for (int i = 0; i < 4*n_cell_y*n_cell_eta; i++) {
                    delete[] qi_nbr_x[i];
                    delete[] vis_nbr_x[i];
                }
                delete[] qi_nbr_x;
                delete[] qi_nbr_y;
                delete[] vis_nbr_x;
                delete[] vis_nbr_y;
                for (int i = 0; i < 4*n_cell_x*n_cell_y; i++) {
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
int Advance::FirstRKStepT(double tau, int rk_flag,
                          double **qi_array, double **qi_nbr_x,
                          double **qi_nbr_y, double **qi_nbr_eta,
                          int n_cell_eta, int n_cell_x, int n_cell_y,
                          double **vis_array, double **vis_nbr_tau,
                          double **vis_nbr_x, double **vis_nbr_y,
                          double **vis_nbr_eta, double **qi_rk0,
                          double **qi_array_new, double **grid_array,
                          double *rhs, double *qiphL, double *qiphR,
                          double *qimhL, double *qimhR,
                          double *grid_array_hL, double *grid_array_hR) {

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
                n_cell_eta, n_cell_x, n_cell_y, qi_array_new, grid_array,
                rhs, qiphL, qiphR, qimhL, qimhR, grid_array_hL, grid_array_hR);

    // now MakeWSource returns partial_a W^{a mu}
    // (including geometric terms) 
    diss->MakeWSource(tau_rk, qi_array, n_cell_eta, n_cell_x, n_cell_y,
                      vis_array, vis_nbr_tau, vis_nbr_x, vis_nbr_y,
                      vis_nbr_eta, qi_array_new);
    
    if (rk_flag == 1) {
        // if rk_flag == 1, we now have q0 + k1 + k2. 
        // So add q0 and multiply by 1/2
        for (int k = 0; k < n_cell_eta; k++) {
            for (int i = 0; i < n_cell_x; i++) {
                for (int j = 0; j < n_cell_y; j++) {
                    int idx = j + i*n_cell_y + k*n_cell_y*n_cell_x;
                    for (int alpha = 0; alpha < 5; alpha++) {
                        qi_array_new[idx][alpha] += qi_rk0[idx][alpha];
                        qi_array_new[idx][alpha] *= 0.5;
                    }
                }
            }
        }
    }

    for (int k = 0; k < n_cell_eta; k++) {
        for (int i = 0; i < n_cell_x; i++) {
            for (int j = 0; j < n_cell_y; j++) {
                int idx = j + i*n_cell_y + k*n_cell_x*n_cell_y;
                reconst_ptr->ReconstIt_shell(grid_array[idx], tau_next,
                                             qi_array_new[idx],
                                             grid_array[idx]);
            }
        }
    }

    // clean up
    return(0);
}


/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */
/*
   Done with T 
   Start W
*/
/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */

int Advance::FirstRKStepW(double tau, int rk_flag, int n_cell_eta,
                          int n_cell_x, int n_cell_y, double **vis_array,
                          double **vis_nbr_tau, double **vis_nbr_x,
                          double **vis_nbr_y, double **vis_nbr_eta,
                          double **velocity_array, double **grid_array,
                          double **vis_array_new) {

    double tau_now = tau;
    double tau_next = tau + (DATA_ptr->delta_tau);
    double tau_rk = tau_now;
    if (rk_flag == 1) {
        tau_rk = tau_next;
    }

    //int mu_max;
    //if (DATA_ptr->turn_on_rhob == 1)
    //    mu_max = 4;
    //else 
    //    mu_max = 3;
 
    // Solve partial_a (u^a W^{mu nu}) = 0
    // Update W^{mu nu}
    // mu = 4 is the baryon current qmu

    // calculate delta uWmunu
    // need to use u[0][mu], remember rk_flag = 0 here
    // with the KT flux 
    // solve partial_tau (u^0 W^{kl}) = -partial_i (u^i W^{kl}
 
    /* Advance uWmunu */
    // add partial_\mu uW^\mu\nu terms using KT
    diss->Make_uWRHS(tau_rk, n_cell_eta, n_cell_x, n_cell_y,
                     vis_array, vis_nbr_x, vis_nbr_y, vis_nbr_eta,
                     velocity_array, vis_array_new);
    // add source terms
    diss->Make_uWSource(tau_rk, n_cell_eta, n_cell_x, n_cell_y, vis_array,
                        velocity_array, grid_array, vis_array_new);
    if (DATA_ptr->turn_on_bulk == 1) {
        diss->Make_uPiSource(tau_rk, n_cell_eta, n_cell_x, n_cell_y, vis_array,
                             velocity_array, grid_array, vis_array_new);
    }
    
    // add source term for baryon diffusion
    if (DATA_ptr->turn_on_diff == 1) {
        diss->Make_uqSource(tau_rk, n_cell_eta, n_cell_x, n_cell_y, vis_array,
                            velocity_array, grid_array, vis_array_new);
    }

    for (int k = 0; k < n_cell_eta; k++) {
        for (int i = 0; i < n_cell_x; i++) {
            for (int j = 0; j < n_cell_y; j++) {
                int idx = j + i*n_cell_y + k*n_cell_x*n_cell_y;
                if (rk_flag == 0) {
                    for (int alpha = 0; alpha < 15; alpha++) {
                            vis_array_new[idx][alpha] /= (
                                                    vis_array_new[idx][15]);
                    }
                } else {
                    for (int alpha = 0; alpha < 15; alpha++) {
                            double rk0 = (vis_nbr_tau[idx][alpha]
                                           *vis_nbr_tau[idx][15]);
                            vis_array_new[idx][alpha] += rk0;
                            vis_array_new[idx][alpha] *= 0.5;
                            vis_array_new[idx][alpha] /= (
                                                    vis_array_new[idx][15]);
                    }
                }
            }
        }
    }

    // reconstruct other components
    double u0, u1, u2, u3;
    for (int k = 0; k < n_cell_eta; k++) {
        for (int i = 0; i < n_cell_x; i++) {
            for (int j = 0; j < n_cell_y; j++) {
                int idx = j + i*n_cell_y + k*n_cell_x*n_cell_y;
                u0 = vis_array_new[idx][15];
                u1 = vis_array_new[idx][16];
                u2 = vis_array_new[idx][17]; 
                u3 = vis_array_new[idx][18]; 

                // re-make Wmunu[3][3] so that Wmunu[mu][nu] is traceless
                vis_array_new[idx][9] = (
                        (2.*(u1*u2*vis_array_new[idx][5]
                             + u1*u3*vis_array_new[idx][6]
                             + u2*u3*vis_array_new[idx][8])
                            - (u0*u0 - u1*u1)*vis_array_new[idx][4] 
                            - (u0*u0 - u2*u2)*vis_array_new[idx][7])
                        /(u0*u0 - u3*u3));

                // make Wmunu^0i using the transversality
                vis_array_new[idx][1] = (vis_array_new[idx][4]*u1
                                         + vis_array_new[idx][5]*u2
                                         + vis_array_new[idx][6]*u3)/u0;
                vis_array_new[idx][2] = (vis_array_new[idx][5]*u1
                                         + vis_array_new[idx][7]*u2
                                         + vis_array_new[idx][8]*u3)/u0;
                vis_array_new[idx][3] = (vis_array_new[idx][6]*u1
                                         + vis_array_new[idx][8]*u2
                                         + vis_array_new[idx][9]*u3)/u0;

                // make Wmunu^00
                vis_array_new[idx][0] = (vis_array_new[idx][1]*u1
                                         + vis_array_new[idx][2]*u2
                                         + vis_array_new[idx][3]*u3)/u0;

                if (DATA_ptr->turn_on_diff == 1) {
                    // make qmu[0] using transversality
                    vis_array_new[idx][10] = (vis_array_new[idx][11]*u1
                                              + vis_array_new[idx][12]*u2
                                              + vis_array_new[idx][13]*u3)/u0;
                } else {
                    vis_array_new[idx][10] = 0.0;
                }

                // If the energy density of the fluid element is smaller
                // than 0.01GeV reduce Wmunu using the QuestRevert algorithm
                if (DATA_ptr->Initial_profile != 0) {
                    QuestRevert(tau, vis_array_new[idx], grid_array[idx]);
                    if (DATA_ptr->turn_on_diff == 1) {
                        QuestRevert_qmu(tau, vis_array_new[idx],
                                        grid_array[idx]);
                    }
                }
            }
        }
    }

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

void Advance::update_grid_array_from_field(
                Field *hydro_fields, int idx, double *grid_array, int rk_flag) {
    if (rk_flag == 0) {
        grid_array[0] = hydro_fields->e_rk0[idx];
        grid_array[4] = hydro_fields->rhob_rk0[idx];
        for (int i = 1; i < 4; i++) {
            grid_array[i] = (hydro_fields->u_rk0[idx][i]
                             /hydro_fields->u_rk0[idx][0]);
        }
    } else {
        grid_array[0] = hydro_fields->e_rk1[idx];
        grid_array[4] = hydro_fields->rhob_rk1[idx];
        for (int i = 1; i < 4; i++) {
            grid_array[i] = (hydro_fields->u_rk1[idx][i]
                             /hydro_fields->u_rk1[idx][0]);
        }
    }
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

void Advance::update_vis_array_from_field(Field *hydro_fields, int idx,
                                          double *vis_array, int rk_flag) {
    if (rk_flag == 0) {
        for (int i = 0; i < 14; i++) {
            vis_array[i] = hydro_fields->Wmunu_rk0[idx][i];
        }
        vis_array[14] = hydro_fields->pi_b_rk0[idx];
        for (int i = 0; i < 4; i++) {
            vis_array[15+i] = hydro_fields->u_rk0[idx][i];
        }
    } else {
        for (int i = 0; i < 14; i++) {
            vis_array[i] = hydro_fields->Wmunu_rk1[idx][i];
        }
        vis_array[14] = hydro_fields->pi_b_rk1[idx];
        for (int i = 0; i < 4; i++) {
            vis_array[15+i] = hydro_fields->u_rk1[idx][i];
        }
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

void Advance::update_vis_prev_tau_from_field(Field *hydro_fields, int idx,
                                             double *vis_array,int rk_flag) {
    if (rk_flag == 0) {
        for (int i = 0; i < 14; i++) {
            vis_array[i] = hydro_fields->Wmunu_prev[idx][i];
        }
        vis_array[14] = hydro_fields->pi_b_prev[idx];
        for (int i = 0; i < 4; i++) {
            vis_array[15+i] = hydro_fields->u_prev[idx][i];
        }
    } else {
        for (int i = 0; i < 14; i++) {
            vis_array[i] = hydro_fields->Wmunu_rk0[idx][i];
        }
        vis_array[14] = hydro_fields->pi_b_rk0[idx];
        for (int i = 0; i < 4; i++) {
            vis_array[15+i] = hydro_fields->u_rk0[idx][i];
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
}

void Advance::update_grid_array_to_hydro_fields(
            double *grid_array, Field *hydro_fields, int idx, int rk_flag) {
    double gamma = 1./sqrt(1. - grid_array[1]*grid_array[1]
                              - grid_array[2]*grid_array[2]
                              - grid_array[3]*grid_array[3]);
    if (rk_flag == 0) {
        hydro_fields->e_rk1[idx] = grid_array[0];
        hydro_fields->rhob_rk1[idx] = grid_array[4];
        hydro_fields->u_rk1[idx][0] = gamma;
        for (int i = 1; i < 4; i++) {
            hydro_fields->u_rk1[idx][i] = gamma*grid_array[i];
        }
    } else {
        hydro_fields->e_rk0[idx] = grid_array[0];
        hydro_fields->rhob_rk0[idx] = grid_array[4];
        hydro_fields->u_rk0[idx][0] = gamma;
        for (int i = 1; i < 4; i++) {
            hydro_fields->u_rk0[idx][i] = gamma*grid_array[i];
        }
    }
}

//! this function reduce the size of shear stress tensor and bulk pressure
//! in the dilute region to stablize numerical simulations
int Advance::QuestRevert(double tau, double *vis_array, double *grid_array) {
    int revert_flag = 0;
    //const double energy_density_warning = 0.01;  // GeV/fm^3, T~100 MeV

    double eps_scale = 1.0;  // 1/fm^4
    double e_local = grid_array[0];
    double factor = 300.*tanh(e_local/eps_scale);

    //double pi_00 = vis_array[0];
    //double pi_01 = vis_array[1];
    //double pi_02 = vis_array[2];
    //double pi_03 = vis_array[3];
    //double pi_11 = vis_array[4];
    //double pi_12 = vis_array[5];
    //double pi_13 = vis_array[6];
    //double pi_22 = vis_array[7];
    //double pi_23 = vis_array[8];
    //double pi_33 = vis_array[9];

    double pisize = (vis_array[0]*vis_array[0]
                     + vis_array[4]*vis_array[4]
                     + vis_array[7]*vis_array[7]
                     + vis_array[9]*vis_array[9]
                     - 2.*(vis_array[1]*vis_array[1]
                           + vis_array[2]*vis_array[2]
                           + vis_array[3]*vis_array[3])
                     + 2.*(vis_array[5]*vis_array[5]
                           + vis_array[6]*vis_array[6]
                           + vis_array[8]*vis_array[8]));
  
    //double pi_local = grid_pt->pi_b[trk_flag];
    //double pi_local = vis_array[14];
    double bulksize = 3.*vis_array[14]*vis_array[14];

    //double rhob_local = grid_pt->rhob;
    //double rhob_local = grid_array[4];
    double p_local = eos->get_pressure(e_local, grid_array[4]);
    double eq_size = e_local*e_local + 3.*p_local*p_local;
       
    double rho_shear = sqrt(pisize/eq_size)/factor; 
    double rho_bulk  = sqrt(bulksize/eq_size)/factor;
 
    // Reducing the shear stress tensor 
    double rho_shear_max = 0.1;
    if (rho_shear > rho_shear_max) {
        //if (e_local*hbarc > energy_density_warning) {
        //    printf("energy density = %lf -- |pi/(epsilon+3*P)| = %lf\n",
        //           e_local*hbarc, rho_shear);
        //}
        double ratio = rho_shear_max/rho_shear;
        for (int mu = 0; mu < 10; mu++) {
            vis_array[mu] *= ratio;
        }
        revert_flag = 1;
    }
   
    // Reducing bulk viscous pressure 
    double rho_bulk_max = 0.1;
    if (rho_bulk > rho_bulk_max) {
        //if (e_local*hbarc > energy_density_warning) {
        //    printf("energy density = %lf --  |Pi/(epsilon+3*P)| = %lf\n",
        //           e_local*hbarc, rho_bulk);
        //}
        vis_array[14] = (rho_bulk_max/rho_bulk)*vis_array[14];
        revert_flag = 1;
    }
    return(revert_flag);
}


//! this function reduce the size of net baryon diffusion current
//! in the dilute region to stablize numerical simulations
int Advance::QuestRevert_qmu(double tau, double *vis_array,
                             double *grid_array) {
    int revert_flag = 0;
    //const double energy_density_warning = 0.01;  // GeV/fm^3, T~100 MeV
    double eps_scale = 1.0;   // in 1/fm^4
    //double e_local = grid_array[0];
    double factor = 300.*tanh(grid_array[0]/eps_scale);

    // calculate the size of q^\mu
    double q_size = (- vis_array[10]*vis_array[10]
                     + vis_array[11]*vis_array[11]
                     + vis_array[12]*vis_array[12]
                     + vis_array[13]*vis_array[13]);

    // first check the positivity of q^mu q_mu 
    // (in the conversion of gmn = diag(-+++))
    if (q_size < 0.0) {
        //cout << "Advance::QuestRevert_qmu: q^mu q_mu = " << q_size << " < 0!"
        //     << endl;
        //cout << "Reset it to zero!!!!" << endl;
        vis_array[10] = 0.0;
        vis_array[11] = 0.0;
        vis_array[12] = 0.0;
        vis_array[13] = 0.0;
        revert_flag = 1;
        return(revert_flag);
    }

    // reduce the size of q^mu according to rhoB
    //double rhob_local = grid_array[4];
    double rho_q = sqrt(q_size/(grid_array[4]*grid_array[4]))/factor;
    double rho_q_max = 0.1;
    if (rho_q > rho_q_max) {
        //if (e_local*hbarc > energy_density_warning) {
        //    printf("energy density = %lf, rhob = %lf -- |q/rhob| = %lf\n",
        //           e_local*hbarc, rhob_local, rho_q);
        //}
        double ratio = rho_q_max/rho_q;
        vis_array[10] *= ratio;
        vis_array[11] *= ratio;
        vis_array[12] *= ratio;
        vis_array[13] *= ratio;
        revert_flag = 1;
    }
    return(revert_flag);
}


//! This function computes the rhs array. It computes the spatial
//! derivatives of T^\mu\nu using the KT algorithm
void Advance::MakeDeltaQI(double tau, double **qi_array, double **qi_nbr_x,
                          double **qi_nbr_y, double **qi_nbr_eta,
                          int n_cell_eta, int n_cell_x, int n_cell_y,
                          double **qi_array_new, double **grid_array,
                          double *rhs, double *qiphL, double *qiphR,
                          double *qimhL, double *qimhR,
                          double *grid_array_hL, double *grid_array_hR) {
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
    //double rhs[5];
    for (int alpha = 0; alpha < 5; alpha++) {
        rhs[alpha] = 0.0;
    }

    //double *qiphL = new double[5];
    //double *qiphR = new double[5];
    //double *qimhL = new double[5];
    //double *qimhR = new double[5];
    //
    //double *grid_array_hL = new double[5];
    //double *grid_array_hR = new double[5];
    
    for (int k = 0; k < n_cell_eta; k++) {
        for (int i = 0; i < n_cell_x; i++) {
            for (int j = 0; j < n_cell_y; j++) {
                int idx = j + i*n_cell_y + k*n_cell_x*n_cell_y;

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
                        int idx_p_1 = j + (i+1)*n_cell_y + k*n_cell_x*n_cell_y;
                        gphR = qi_array[idx_p_1][alpha];
                    } else {
                        int idx_p_1 = 4*j + k*4*n_cell_y + 2;
                        gphR = qi_nbr_x[idx_p_1][alpha];
                    }
                    if (i - 1 >= 0) {
                        int idx_m_1 = j + (i-1)*n_cell_y + k*n_cell_x*n_cell_y;
                        gmhL = qi_array[idx_m_1][alpha];
                    } else {
                        int idx_m_1 = 4*j + k*4*n_cell_y + 1;
                        gmhL = qi_nbr_x[idx_m_1][alpha];
                    }
                    if (i + 2 < n_cell_x) {
                        int idx_p_2 = j + (i+2)*n_cell_y + k*n_cell_x*n_cell_y;
                        gphR2 = qi_array[idx_p_2][alpha];
                    } else {
                        int idx_p_2 = 4*j + k*4*n_cell_y + 4 + i - n_cell_x;
                        gphR2 = qi_nbr_x[idx_p_2][alpha];
                    }
                    if (i - 2 >= 0) {
                        int idx_m_2 = j + (i-2)*n_cell_y + k*n_cell_x*n_cell_y;
                        gmhL2 = qi_array[idx_m_2][alpha];
                    } else {
                        int idx_m_2 = 4*j + k*4*n_cell_y + i;
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
                //cout << "x-direction" << endl;
                
                // y-direction
                direc = 2;
                tau_fac = tau;
                for (int alpha = 0; alpha < 5; alpha++) {
                    double gp = qi_array[idx][alpha];
                    double gphL = qi_array[idx][alpha];
                    double gmhR = qi_array[idx][alpha];

                    double gphR, gmhL, gphR2, gmhL2;
                    if (j + 1 < n_cell_y) {
                        int idx_p_1 = j + 1 + i*n_cell_y + k*n_cell_x*n_cell_y;
                        gphR = qi_array[idx_p_1][alpha];
                    } else {
                        int idx_p_1 = 4*i + 4*k*n_cell_x + 2;
                        gphR = qi_nbr_y[idx_p_1][alpha];
                    }
                    if (j - 1 >= 0) {
                        int idx_m_1 = j - 1 + i*n_cell_y + k*n_cell_x*n_cell_y;
                        gmhL = qi_array[idx_m_1][alpha];
                    } else {
                        int idx_m_1 = 4*i + 4*k*n_cell_x + 1;
                        gmhL = qi_nbr_y[idx_m_1][alpha];
                    }
                    if (j + 2 < n_cell_y) {
                        int idx_p_2 = j + 2 + i*n_cell_y + k*n_cell_x*n_cell_y;
                        gphR2 = qi_array[idx_p_2][alpha];
                    } else {
                        int idx_p_2 = 4*i + 4*k*n_cell_x + 4 + j - n_cell_y;
                        gphR2 = qi_nbr_y[idx_p_2][alpha];
                    }
                    if (j - 2 >= 0) {
                        int idx_m_2 = j - 2 + i*n_cell_y + k*n_cell_x*n_cell_y;
                        gmhL2 = qi_array[idx_m_2][alpha];
                    } else {
                        int idx_m_2 = 4*i + 4*k*n_cell_x + j;
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
                //cout << "y-direction" << endl;
                
                // eta-direction
                direc = 3;
                tau_fac = 1.0;
                for (int alpha = 0; alpha < 5; alpha++) {
                    double gp = qi_array[idx][alpha];
                    double gphL = qi_array[idx][alpha];
                    double gmhR = qi_array[idx][alpha];

                    double gphR, gmhL, gphR2, gmhL2;
                    if (k + 1 < n_cell_eta) {
                        int idx_p_1 = j + i*n_cell_y + (k+1)*n_cell_x*n_cell_y;
                        gphR = qi_array[idx_p_1][alpha];
                    } else {
                        int idx_p_1 = 4*j + 4*i*n_cell_y + 2;
                        gphR = qi_nbr_eta[idx_p_1][alpha];
                    }
                    if (k - 1 >= 0) {
                        int idx_m_1 = j + i*n_cell_y + (k-1)*n_cell_x*n_cell_y;
                        gmhL = qi_array[idx_m_1][alpha];
                    } else {
                        int idx_m_1 = 4*j + 4*i*n_cell_y + 1;
                        gmhL = qi_nbr_eta[idx_m_1][alpha];
                    }
                    if (k + 2 < n_cell_eta) {
                        int idx_p_2 = j + i*n_cell_y + (k+2)*n_cell_x*n_cell_y;
                        gphR2 = qi_array[idx_p_2][alpha];
                    } else {
                        int idx_p_2 = 4*j + 4*i*n_cell_y + 4 + k - n_cell_eta;
                        gphR2 = qi_nbr_eta[idx_p_2][alpha];
                    }
                    if (k - 2 >= 0) {
                        int idx_m_2 = j + i*n_cell_y + (k-2)*n_cell_x*n_cell_y;
                        gmhL2 = qi_array[idx_m_2][alpha];
                    } else {
                        int idx_m_2 = 4*j + 4*i*n_cell_y + k;
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
                //cout << "eta-direction" << endl;

                // geometric terms
                rhs[0] -= (get_TJb_new(grid_array[idx], 3, 3)
                           *DATA_ptr->delta_tau);
                rhs[3] -= (get_TJb_new(grid_array[idx], 3, 0)
                           *DATA_ptr->delta_tau);
                
                for (int alpha = 0; alpha < 5; alpha++) {
                    qi_array_new[idx][alpha] = qi_array[idx][alpha] + rhs[alpha];
                }
            }
        }
    }

    // clean up
    //delete[] qiphL;
    //delete[] qiphR;
    //delete[] qimhL;
    //delete[] qimhR;

    //delete[] grid_array_hL;
    //delete[] grid_array_hR;
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

void Advance::update_grid_cell(double **grid_array, Field *hydro_fields, int rk_flag,
                               int ieta, int ix, int iy,
                               int n_cell_eta, int n_cell_x, int n_cell_y) {
    for (int k = 0; k < n_cell_eta; k++) {
        int idx_ieta = min(ieta + k, DATA_ptr->neta - 1);
        for (int i = 0; i < n_cell_x; i++) {
            int idx_ix = min(ix + i, DATA_ptr->nx);
            for (int j = 0; j < n_cell_y; j++) {
                int idx_iy = min(iy + j, DATA_ptr->ny);
                int field_idx = (idx_iy + idx_ix*(DATA_ptr->ny+1)
                                 + idx_ieta*(DATA_ptr->ny+1)*(DATA_ptr->nx+1));
                int idx = j + i*n_cell_y + k*n_cell_x*n_cell_y;
                update_grid_array_to_hydro_fields(
                        grid_array[idx], hydro_fields, field_idx, rk_flag);
            }
        }
    }
}           

void Advance::update_grid_cell_viscous(double **vis_array, Field *hydro_fields,
        int rk_flag, int ieta, int ix, int iy, int n_cell_eta, int n_cell_x,
        int n_cell_y) {

    int field_idx;
    int field_ny = DATA_ptr->ny + 1;
    int field_nperp = (DATA_ptr->ny + 1)*(DATA_ptr->nx + 1);

    for (int k = 0; k < n_cell_eta; k++) {
        int idx_ieta = min(ieta + k, DATA_ptr->neta - 1);
        for (int i = 0; i < n_cell_x; i++) {
            int idx_ix = min(ix + i, DATA_ptr->nx);
            for (int j = 0; j < n_cell_y; j++) {
                int idx_iy = min(iy + j, DATA_ptr->ny);
                int idx = j + i*n_cell_y + k*n_cell_x*n_cell_y;
                field_idx = (idx_iy + idx_ix*field_ny + idx_ieta*field_nperp);
                if (rk_flag == 0) {
                    for (int alpha = 0; alpha < 14; alpha++) {
                        hydro_fields->Wmunu_rk1[field_idx][alpha] = (
                                                        vis_array[idx][alpha]);
                    }
                    hydro_fields->pi_b_rk1[field_idx] = vis_array[idx][14];
                } else {
                    for (int alpha = 0; alpha < 14; alpha++) {
                        hydro_fields->Wmunu_rk0[field_idx][alpha] = (
                                                        vis_array[idx][alpha]);
                    }
                    hydro_fields->pi_b_rk0[field_idx] = vis_array[idx][14];
                }
            }
        }
    }
}           
