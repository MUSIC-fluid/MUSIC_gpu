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
        int sub_grid_neta, int sub_grid_x, int sub_grid_y,
        double qi_array[][5], double qi_nbr_x[][5],
        double qi_nbr_y[][5], double qi_nbr_eta[][5],
        double qi_rk0[][5], double grid_array[][5], double *grid_array_temp) {

    double tau_rk;
    if (rk_flag == 0) {
        tau_rk = tau;
    } else {
        tau_rk = tau + DELTA_TAU;
    }

    int field_idx;
    int field_ny = GRID_SIZE_Y + 1;
    int field_nperp = (GRID_SIZE_Y + 1)*(GRID_SIZE_X + 1);
    // first build qi cube sub_grid_x*sub_grid_x*sub_grid_neta
    for (int k = 0; k < sub_grid_neta; k++) {
        int idx_ieta = MIN(ieta + k, GRID_SIZE_ETA - 1);
        for (int i = 0; i < sub_grid_x; i++) {
            int idx_ix = MIN(ix + i, GRID_SIZE_X);
            for (int j = 0; j < sub_grid_y; j++) {
                int idx_iy = MIN(iy + j, GRID_SIZE_Y);
                int idx = j + sub_grid_y*i + sub_grid_x*sub_grid_y*k;
                field_idx = (idx_iy + idx_ix*field_ny + idx_ieta*field_nperp);
                update_grid_array_from_field(hydro_fields, field_idx,
                                             grid_array[idx], rk_flag);
                get_qmu_from_grid_array(tau_rk, qi_array[idx],
                                        grid_array[idx]);
            }
        }
    }

    if (rk_flag == 1) {
        for (int k = 0; k < sub_grid_neta; k++) {
            int idx_ieta = MIN(ieta + k, GRID_SIZE_ETA - 1);
            for (int i = 0; i < sub_grid_x; i++) {
                int idx_ix = MIN(ix + i, GRID_SIZE_X);
                for (int j = 0; j < sub_grid_y; j++) {
                    int idx_iy = MIN(iy + j, GRID_SIZE_Y);
                    int idx = j + sub_grid_y*i + sub_grid_x*sub_grid_y*k;
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
    for (int k = 0; k < sub_grid_neta; k++) {
        int idx_ieta = MIN(ieta + k, GRID_SIZE_ETA - 1);
        for (int i = 0; i < sub_grid_y; i++) {
            int idx_iy = MIN(iy + i, GRID_SIZE_Y);
            int idx = 4*i + 4*sub_grid_y*k;

            int idx_m_2 = MAX(0, ix - 2);
            int idx_m_1 = MAX(0, ix - 1);
            int idx_p_1 = MIN(ix + sub_grid_x, GRID_SIZE_X);
            int idx_p_2 = MIN(ix + sub_grid_x + 1, GRID_SIZE_X);

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
    for (int k = 0; k < sub_grid_neta; k++) {
        int idx_ieta = MIN(ieta + k, GRID_SIZE_ETA - 1);
        for (int i = 0; i < sub_grid_x; i++) {
            int idx_ix = MIN(ix + i, GRID_SIZE_X);
            int idx = 4*i + 4*sub_grid_x*k;

            int idx_m_2 = MAX(0, iy - 2);
            int idx_m_1 = MAX(0, iy - 1);
            int idx_p_1 = MIN(iy + sub_grid_y, GRID_SIZE_Y);
            int idx_p_2 = MIN(iy + sub_grid_y + 1, GRID_SIZE_Y);

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
    for (int i = 0; i < sub_grid_x; i++) {
        int idx_ix = MIN(ix + i, GRID_SIZE_X);
        for (int k = 0; k < sub_grid_y; k++) {
            int idx_iy = MIN(iy + k, GRID_SIZE_Y);
            int idx = 4*k + 4*sub_grid_y*i;

            int idx_m_2 = MAX(0, ieta - 2);
            int idx_m_1 = MAX(0, ieta - 1);
            int idx_p_1 = MIN(ieta + sub_grid_neta, GRID_SIZE_ETA-1);
            int idx_p_2 = MIN(ieta + sub_grid_neta + 1, GRID_SIZE_ETA-1);

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
        int sub_grid_neta, int sub_grid_x, int sub_grid_y,
        double vis_array[][19], double vis_nbr_tau[][19],
        double vis_nbr_x[][19], double vis_nbr_y[][19],
        double vis_nbr_eta[][19]) {

    int field_idx;
    int field_ny = GRID_SIZE_Y + 1;
    int field_nperp = (GRID_SIZE_Y + 1)*(GRID_SIZE_X + 1);

    // first build qi cube sub_grid_x*sub_grid_x*sub_grid_neta
    for (int k = 0; k < sub_grid_neta; k++) {
        int idx_ieta = MIN(ieta + k, GRID_SIZE_ETA - 1);
        for (int i = 0; i < sub_grid_x; i++) {
            int idx_ix = MIN(ix + i, GRID_SIZE_X);
            for (int j = 0; j < sub_grid_y; j++) {
                int idx_iy = MIN(iy + j, GRID_SIZE_Y);
                int idx = j + sub_grid_y*i + sub_grid_x*sub_grid_y*k;

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
    for (int k = 0; k < sub_grid_neta; k++) {
        int idx_ieta = MIN(ieta + k, GRID_SIZE_ETA - 1);
        for (int i = 0; i < sub_grid_y; i++) {
            int idx_iy = MIN(iy + i, GRID_SIZE_Y);
            int idx = 4*i + 4*sub_grid_y*k;

            int idx_m_2 = MAX(0, ix - 2);
            int idx_m_1 = MAX(0, ix - 1);
            int idx_p_1 = MIN(ix + sub_grid_x, GRID_SIZE_X);
            int idx_p_2 = MIN(ix + sub_grid_x + 1, GRID_SIZE_X);

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
    for (int k = 0; k < sub_grid_neta; k++) {
        int idx_ieta = MIN(ieta + k, GRID_SIZE_ETA - 1);
        for (int i = 0; i < sub_grid_x; i++) {
            int idx_ix = MIN(ix + i, GRID_SIZE_X);
            int idx = 4*i + 4*sub_grid_x*k;

            int idx_m_2 = MAX(0, iy - 2);
            int idx_m_1 = MAX(0, iy - 1);
            int idx_p_1 = MIN(iy + sub_grid_y, GRID_SIZE_Y);
            int idx_p_2 = MIN(iy + sub_grid_y + 1, GRID_SIZE_Y);

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
    for (int i = 0; i < sub_grid_x; i++) {
        int idx_ix = MIN(ix + i, GRID_SIZE_X);
        for (int k = 0; k < sub_grid_y; k++) {
            int idx_iy = MIN(iy + k, GRID_SIZE_Y);
            int idx = 4*k + 4*sub_grid_y*i;

            int idx_m_2 = MAX(0, ieta - 2);
            int idx_m_1 = MAX(0, ieta - 1);
            int idx_p_1 = MIN(ieta + sub_grid_neta, GRID_SIZE_ETA - 1);
            int idx_p_2 = MIN(ieta + sub_grid_neta + 1, GRID_SIZE_ETA - 1);

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
                                     int sub_grid_neta, int sub_grid_x,
                                     int sub_grid_y, double velocity_array[][20], 
                                     double grid_array[][5],
                                     double vis_array_new[][19],
                                     double *grid_array_temp) {
    int trk_flag = 1;
    if (rk_flag == 1) {
        trk_flag = 0;
    }

    int field_idx;
    int field_ny = GRID_SIZE_Y + 1;
    int field_nperp = (GRID_SIZE_Y + 1)*(GRID_SIZE_X + 1);

    for (int k = 0; k < sub_grid_neta; k++) {
        int idx_ieta = MIN(ieta + k, GRID_SIZE_ETA - 1);
        for (int i = 0; i < sub_grid_x; i++) {
            int idx_ix = MIN(ix + i, GRID_SIZE_X);
            for (int j = 0; j < sub_grid_y; j++) {
                int idx_iy = MIN(iy + j, GRID_SIZE_Y);
                int idx = j + sub_grid_y*i + sub_grid_x*sub_grid_y*k;

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
                        calculate_expansion_rate_1(
                                    tau_rk, hydro_fields, field_idx, rk_flag));
                calculate_Du_supmu_1(
                        tau_rk, hydro_fields, field_idx, rk_flag,
                        velocity_array[idx]);
                calculate_velocity_shear_tensor_2(
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
int Advance::AdvanceIt(double tau, Field *hydro_fields,
                       int rk_flag) {
    //const int cube_size=SUB_GRID_SIZE_X*SUB_GRID_SIZE_Y*SUB_GRID_SIZE_ETA;
    //const int neigh_sizex=4*SUB_GRID_SIZE_Y*SUB_GRID_SIZE_ETA;
    //const int neigh_sizey=4*SUB_GRID_SIZE_X*SUB_GRID_SIZE_ETA;
    //const int neigh_sizeeta=4*SUB_GRID_SIZE_X*SUB_GRID_SIZE_Y;
    double tmp[2]={-1.1, -2.2};
    double grid_array[1][5], qi_array[1][5], qi_array_new[1][5], qi_rk0[1][5];
    double qi_nbr_x[4][5], qi_nbr_y[4][5], qi_nbr_eta[4][5];
    double vis_array[1][19], vis_array_new[1][19], vis_nbr_tau[1][19];
    double velocity_array[1][20];
    double vis_nbr_x[4][19], vis_nbr_y[4][19], vis_nbr_eta[4][19];
    double grid_array_temp[5];
    double rhs[5];
    double qiphL[5];
    double qiphR[5];
    double qimhL[5];
    double qimhR[5];
    double grid_array_hL[5];
    double grid_array_hR[5];
    

    cout << "pre parallel" << endl;
    #pragma acc parallel loop gang worker vector collapse(3) copy(tmp[0:1]) present(hydro_fields[0:1],\
                         hydro_fields->e_rk0[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA],\
                         hydro_fields->e_prev[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA],\
                         hydro_fields->rhob_rk0[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA],\
                         hydro_fields->e_rk1[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA],\
                         hydro_fields->rhob_rk1[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA],\
                         hydro_fields->rhob_prev[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA], \
                         hydro_fields->u_rk0[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA][0:4], \
                         hydro_fields->u_rk1[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA][0:4], \
                         hydro_fields->u_prev[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA][0:4], \
                         hydro_fields->dUsup[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA][0:20], \
                         hydro_fields->Wmunu_rk0[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA][0:14], \
                         hydro_fields->Wmunu_rk1[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA][0:14], \
                         hydro_fields->Wmunu_prev[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA][0:14], \
                         hydro_fields->pi_b_rk0[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA], \
                         hydro_fields->pi_b_rk1[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA], \
                         hydro_fields->pi_b_prev[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA])\
                         private(this[0:1], grid_array[1][5], qi_array[1][5], qi_array_new[1][5], qi_rk0[1][5], \
                         qi_nbr_x[4][5], qi_nbr_y[4][5], qi_nbr_eta[4][5], vis_array[1][19], \
                         vis_array_new[1][19], vis_nbr_tau[1][19], velocity_array[1][20], \
                         vis_nbr_x[4][19], vis_nbr_y[4][19], vis_nbr_eta[4][19], grid_array_temp[5], \
                         grid_array_hL[0:5], qimhL[0:5], grid_array_hR[0:5], qiphL[0:5], qimhR[0:5], \
                         rhs[0:5], qiphR[0:5])
    for (int ieta = 0; ieta < GRID_SIZE_ETA; ieta += SUB_GRID_SIZE_ETA) {
//        #pragma omp parallel private(ix)
//        {
//            #pragma omp for
            for (int ix = 0; ix <= GRID_SIZE_X; ix += SUB_GRID_SIZE_X) {
                for (int iy = 0; iy <= GRID_SIZE_Y; iy += SUB_GRID_SIZE_Y) {

                        tmp[0]=tau; //hydro_fields->e_rk0[0];

                   prepare_qi_array(tau, hydro_fields, rk_flag, ieta, ix, iy,
                                    SUB_GRID_SIZE_ETA, SUB_GRID_SIZE_X, SUB_GRID_SIZE_Y, qi_array,
                                    qi_nbr_x, qi_nbr_y, qi_nbr_eta,
                                    qi_rk0, grid_array, grid_array_temp);
//                        tmp=grid_array[0][0]; //hydro_fields->e_rk0[10];
                    // viscous source terms
//                    prepare_vis_array(hydro_fields, rk_flag, ieta, ix, iy,
//                                      SUB_GRID_SIZE_ETA, SUB_GRID_SIZE_X, SUB_GRID_SIZE_Y,
//                                      vis_array, vis_nbr_tau, vis_nbr_x,
//                                      vis_nbr_y, vis_nbr_eta);

                   FirstRKStepT(tau, rk_flag,
                                qi_array, qi_nbr_x, qi_nbr_y, qi_nbr_eta,
                                SUB_GRID_SIZE_ETA, SUB_GRID_SIZE_X, SUB_GRID_SIZE_Y,
                                vis_array, vis_nbr_tau,
                                vis_nbr_x, vis_nbr_y, vis_nbr_eta,
                                qi_rk0, qi_array_new, grid_array,
                                rhs, qiphL, qiphR, qimhL, qimhR,
                                grid_array_hL, grid_array_hR);

                    update_grid_cell(grid_array, hydro_fields, rk_flag, ieta, ix, iy,
                                     SUB_GRID_SIZE_ETA, SUB_GRID_SIZE_X, SUB_GRID_SIZE_Y);

//                    if (VISCOUS_FLAG == 1) {
//                        double tau_rk = tau;
//                        if (rk_flag == 1) {
//                            tau_rk = tau + DELTA_TAU;
//                        }

      //                  prepare_velocity_array(tau_rk, hydro_fields,
      //                                         ieta, ix, iy,
      //                                         rk_flag, SUB_GRID_SIZE_ETA, SUB_GRID_SIZE_X,
      //                                         SUB_GRID_SIZE_Y, velocity_array,
      //                                         grid_array, vis_array_new,
      //                                         grid_array_temp);

      //                  FirstRKStepW(tau, rk_flag, SUB_GRID_SIZE_ETA, SUB_GRID_SIZE_X,
      //                               SUB_GRID_SIZE_Y, vis_array, vis_nbr_tau,
      //                               vis_nbr_x, vis_nbr_y, vis_nbr_eta,
      //                               velocity_array, grid_array,
      //                               vis_array_new);

      //                  update_grid_cell_viscous(vis_array_new, hydro_fields, rk_flag,
      //                                           ieta, ix, iy, SUB_GRID_SIZE_ETA,
      //                                           SUB_GRID_SIZE_X, SUB_GRID_SIZE_Y);
                   }
                }
            }
//        }
//        #pragma omp barrier
    //clean up
    std::cout << "tmp=" << tmp[0] << "\n";

    return(1);
}/* AdvanceIt */


/* %%%%%%%%%%%%%%%%%%%%%% First steps begins here %%%%%%%%%%%%%%%%%% */
int Advance::FirstRKStepT(double tau, int rk_flag,
                          double qi_array[][5], double qi_nbr_x[][5],
                          double qi_nbr_y[][5], double qi_nbr_eta[][5],
                          int sub_grid_neta, int sub_grid_x, int sub_grid_y,
                          double vis_array[][19], double vis_nbr_tau[][19],
                          double vis_nbr_x[][19], double vis_nbr_y[][19],
                          double vis_nbr_eta[][19], double qi_rk0[][5],
                          double qi_array_new[][5], double grid_array[][5],
                          double *rhs, double *qiphL, double *qiphR,
                          double *qimhL, double *qimhR,
                          double *grid_array_hL, double *grid_array_hR) {

    // this advances the ideal part
    double tau_next = tau + (DELTA_TAU);
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
                sub_grid_neta, sub_grid_x, sub_grid_y, qi_array_new, grid_array,
                rhs, qiphL, qiphR, qimhL, qimhR, grid_array_hL, grid_array_hR);

//    // now MakeWSource returns partial_a W^{a mu}
//    // (including geometric terms) 
//    MakeWSource(tau_rk, qi_array, sub_grid_neta, sub_grid_x, sub_grid_y,
//                vis_array, vis_nbr_tau, vis_nbr_x, vis_nbr_y,
//                vis_nbr_eta, qi_array_new, DATA);
    
    if (rk_flag == 1) {
        // if rk_flag == 1, we now have q0 + k1 + k2. 
        // So add q0 and multiply by 1/2
        for (int k = 0; k < sub_grid_neta; k++) {
            for (int i = 0; i < sub_grid_x; i++) {
                for (int j = 0; j < sub_grid_y; j++) {
                    int idx = j + i*sub_grid_y + k*sub_grid_y*sub_grid_x;
                    for (int alpha = 0; alpha < 5; alpha++) {
                        qi_array_new[idx][alpha] += qi_rk0[idx][alpha];
                        qi_array_new[idx][alpha] *= 0.5;
                    }
                }
            }
        }
    }

    for (int k = 0; k < sub_grid_neta; k++) {
        for (int i = 0; i < sub_grid_x; i++) {
            for (int j = 0; j < sub_grid_y; j++) {
                int idx = j + i*sub_grid_y + k*sub_grid_x*sub_grid_y;
                ReconstIt_velocity_Newton(grid_array[idx], tau_next,
                                          qi_array_new[idx],
                                          grid_array[idx]);
            }
        }
    }
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
    }/* if t00-k00/t00 < 0.0 */

    double u0, u1, u2, u3, epsilon, pressure, rhob;

    double u0_guess = 1./sqrt(1. - grid_array_p[1]*grid_array_p[1]
                              - grid_array_p[2]*grid_array_p[2]
                              - grid_array_p[3]*grid_array_p[3]);
    double v_guess = sqrt(1. - 1./(u0_guess*u0_guess + 1e-15));
    //if (isnan(v_guess)) {
    //    v_guess = 0.0;
    //}
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
        revert_grid(grid_array, grid_array_p);
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


/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */
/*
   Done with T 
   Start W
*/
/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */

int Advance::FirstRKStepW(double tau, int rk_flag, int sub_grid_neta,
                          int sub_grid_x, int sub_grid_y, double vis_array[][19],
                          double vis_nbr_tau[][19], double vis_nbr_x[][19],
                          double vis_nbr_y[][19], double vis_nbr_eta[][19],
                          double velocity_array[][20], double grid_array[][5],
                          double vis_array_new[][19]) {

    double tau_now = tau;
    double tau_next = tau + (DELTA_TAU);
    double tau_rk = tau_now;
    if (rk_flag == 1) {
        tau_rk = tau_next;
    }

    // Solve partial_a (u^a W^{mu nu}) = 0
    // Update W^{mu nu}
    // mu = 4 is the baryon current qmu

    // calculate delta uWmunu
    // need to use u[0][mu], remember rk_flag = 0 here
    // with the KT flux 
    // solve partial_tau (u^0 W^{kl}) = -partial_i (u^i W^{kl}
 
    /* Advance uWmunu */
    // add partial_\mu uW^\mu\nu terms using KT
    Make_uWRHS(tau_rk, sub_grid_neta, sub_grid_x, sub_grid_y,
               vis_array, vis_nbr_x, vis_nbr_y, vis_nbr_eta,
               velocity_array, vis_array_new);

    // add source terms
    Make_uWSource(tau_rk, sub_grid_neta, sub_grid_x, sub_grid_y, vis_array,
                  velocity_array, grid_array, vis_array_new);
    if (INCLUDE_BULK == 1) {
        Make_uPiSource(tau_rk, sub_grid_neta, sub_grid_x, sub_grid_y, vis_array,
                       velocity_array, grid_array, vis_array_new);
    }
    
    for (int k = 0; k < sub_grid_neta; k++) {
        for (int i = 0; i < sub_grid_x; i++) {
            for (int j = 0; j < sub_grid_y; j++) {
                int idx = j + i*sub_grid_y + k*sub_grid_x*sub_grid_y;
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
    for (int k = 0; k < sub_grid_neta; k++) {
        for (int i = 0; i < sub_grid_x; i++) {
            for (int j = 0; j < sub_grid_y; j++) {
                int idx = j + i*sub_grid_y + k*sub_grid_x*sub_grid_y;
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

                    vis_array_new[idx][10] = 0.0;

                // If the energy density of the fluid element is smaller
                // than 0.01GeV reduce Wmunu using the QuestRevert algorithm
                if (INITIAL_PROFILE >2) {
                    QuestRevert(tau, vis_array_new[idx], grid_array[idx]);
                }
            }
        }
    }

    return(1);
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
    double p_local = get_pressure(e_local, grid_array[4]);
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




//! This function computes the rhs array. It computes the spatial
//! derivatives of T^\mu\nu using the KT algorithm
void Advance::MakeDeltaQI(double tau, double qi_array[][5], double qi_nbr_x[][5],
                          double qi_nbr_y[][5], double qi_nbr_eta[][5],
                          int sub_grid_neta, int sub_grid_x, int sub_grid_y,
                          double qi_array_new[][5], double grid_array[][5],
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
    
    for (int k = 0; k < sub_grid_neta; k++) {
        for (int i = 0; i < sub_grid_x; i++) {
            for (int j = 0; j < sub_grid_y; j++) {
                int idx = j + i*sub_grid_y + k*sub_grid_x*sub_grid_y;

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
                    if (i + 1 < sub_grid_x) {
                        int idx_p_1 = j + (i+1)*sub_grid_y + k*sub_grid_x*sub_grid_y;
                        gphR = qi_array[idx_p_1][alpha];
                    } else {
                        int idx_p_1 = 4*j + k*4*sub_grid_y + 2;
                        gphR = qi_nbr_x[idx_p_1][alpha];
                    }
                    if (i - 1 >= 0) {
                        int idx_m_1 = j + (i-1)*sub_grid_y + k*sub_grid_x*sub_grid_y;
                        gmhL = qi_array[idx_m_1][alpha];
                    } else {
                        int idx_m_1 = 4*j + k*4*sub_grid_y + 1;
                        gmhL = qi_nbr_x[idx_m_1][alpha];
                    }
                    if (i + 2 < sub_grid_x) {
                        int idx_p_2 = j + (i+2)*sub_grid_y + k*sub_grid_x*sub_grid_y;
                        gphR2 = qi_array[idx_p_2][alpha];
                    } else {
                        int idx_p_2 = 4*j + k*4*sub_grid_y + 4 + i - sub_grid_x;
                        gphR2 = qi_nbr_x[idx_p_2][alpha];
                    }
                    if (i - 2 >= 0) {
                        int idx_m_2 = j + (i-2)*sub_grid_y + k*sub_grid_x*sub_grid_y;
                        gmhL2 = qi_array[idx_m_2][alpha];
                    } else {
                        int idx_m_2 = 4*j + k*4*sub_grid_y + i;
                        gmhL2 = qi_nbr_x[idx_m_2][alpha];
                    }

                    double fphL = 0.5*minmod_dx(gphR, gp, gmhL);
                    double fphR = -0.5*minmod_dx(gphR2, gphR, gp);
                    double fmhL = 0.5*minmod_dx(gp, gmhL, gmhL2);
                    double fmhR = -0.5*minmod_dx(gphR, gp, gmhL);
                    qiphL[alpha] = gphL + fphL;
                    qiphR[alpha] = gphR + fphR;
                    qimhL[alpha] = gmhL + fmhL;
                    qimhR[alpha] = gmhR + fmhR;
                }
                // for each direction, reconstruct half-way cells
                // reconstruct e, rhob, and u[4] for half way cells
                int flag = ReconstIt_velocity_Newton(
                                grid_array_hL, tau, qiphL, grid_array[idx]);
                double aiphL = MaxSpeed(tau, direc, grid_array_hL);

                flag *= ReconstIt_velocity_Newton(
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

                    rhs[alpha] = -Fiph/DELTA_X*DELTA_TAU;
                }

                flag *= ReconstIt_velocity_Newton(grid_array_hL, tau,
                                                     qimhL, grid_array[idx]);
                double aimhL = MaxSpeed(tau, direc, grid_array_hL);

                flag *= ReconstIt_velocity_Newton(grid_array_hR, tau,
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
                    rhs[alpha] += Fimh/DELTA_X*DELTA_TAU;
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
                    if (j + 1 < sub_grid_y) {
                        int idx_p_1 = j + 1 + i*sub_grid_y + k*sub_grid_x*sub_grid_y;
                        gphR = qi_array[idx_p_1][alpha];
                    } else {
                        int idx_p_1 = 4*i + 4*k*sub_grid_x + 2;
                        gphR = qi_nbr_y[idx_p_1][alpha];
                    }
                    if (j - 1 >= 0) {
                        int idx_m_1 = j - 1 + i*sub_grid_y + k*sub_grid_x*sub_grid_y;
                        gmhL = qi_array[idx_m_1][alpha];
                    } else {
                        int idx_m_1 = 4*i + 4*k*sub_grid_x + 1;
                        gmhL = qi_nbr_y[idx_m_1][alpha];
                    }
                    if (j + 2 < sub_grid_y) {
                        int idx_p_2 = j + 2 + i*sub_grid_y + k*sub_grid_x*sub_grid_y;
                        gphR2 = qi_array[idx_p_2][alpha];
                    } else {
                        int idx_p_2 = 4*i + 4*k*sub_grid_x + 4 + j - sub_grid_y;
                        gphR2 = qi_nbr_y[idx_p_2][alpha];
                    }
                    if (j - 2 >= 0) {
                        int idx_m_2 = j - 2 + i*sub_grid_y + k*sub_grid_x*sub_grid_y;
                        gmhL2 = qi_array[idx_m_2][alpha];
                    } else {
                        int idx_m_2 = 4*i + 4*k*sub_grid_x + j;
                        gmhL2 = qi_nbr_y[idx_m_2][alpha];
                    }

                    double fphL = 0.5*minmod_dx(gphR, gp, gmhL);
                    double fphR = -0.5*minmod_dx(gphR2, gphR, gp);
                    double fmhL = 0.5*minmod_dx(gp, gmhL, gmhL2);
                    double fmhR = -0.5*minmod_dx(gphR, gp, gmhL);
                    qiphL[alpha] = gphL + fphL;
                    qiphR[alpha] = gphR + fphR;
                    qimhL[alpha] = gmhL + fmhL;
                    qimhR[alpha] = gmhR + fmhR;
                }
                // for each direction, reconstruct half-way cells
                // reconstruct e, rhob, and u[4] for half way cells
                flag = ReconstIt_velocity_Newton(
                                grid_array_hL, tau, qiphL, grid_array[idx]);
                aiphL = MaxSpeed(tau, direc, grid_array_hL);

                flag *= ReconstIt_velocity_Newton(
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

                    rhs[alpha] -= Fiph/DELTA_Y*DELTA_TAU;
                }

                flag *= ReconstIt_velocity_Newton(grid_array_hL, tau,
                                                     qimhL, grid_array[idx]);
                aimhL = MaxSpeed(tau, direc, grid_array_hL);

                flag *= ReconstIt_velocity_Newton(grid_array_hR, tau,
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
                    rhs[alpha] += Fimh/DELTA_Y*DELTA_TAU;
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
                    if (k + 1 < sub_grid_neta) {
                        int idx_p_1 = j + i*sub_grid_y + (k+1)*sub_grid_x*sub_grid_y;
                        gphR = qi_array[idx_p_1][alpha];
                    } else {
                        int idx_p_1 = 4*j + 4*i*sub_grid_y + 2;
                        gphR = qi_nbr_eta[idx_p_1][alpha];
                    }
                    if (k - 1 >= 0) {
                        int idx_m_1 = j + i*sub_grid_y + (k-1)*sub_grid_x*sub_grid_y;
                        gmhL = qi_array[idx_m_1][alpha];
                    } else {
                        int idx_m_1 = 4*j + 4*i*sub_grid_y + 1;
                        gmhL = qi_nbr_eta[idx_m_1][alpha];
                    }
                    if (k + 2 < sub_grid_neta) {
                        int idx_p_2 = j + i*sub_grid_y + (k+2)*sub_grid_x*sub_grid_y;
                        gphR2 = qi_array[idx_p_2][alpha];
                    } else {
                        int idx_p_2 = 4*j + 4*i*sub_grid_y + 4 + k - sub_grid_neta;
                        gphR2 = qi_nbr_eta[idx_p_2][alpha];
                    }
                    if (k - 2 >= 0) {
                        int idx_m_2 = j + i*sub_grid_y + (k-2)*sub_grid_x*sub_grid_y;
                        gmhL2 = qi_array[idx_m_2][alpha];
                    } else {
                        int idx_m_2 = 4*j + 4*i*sub_grid_y + k;
                        gmhL2 = qi_nbr_eta[idx_m_2][alpha];
                    }

                    double fphL = 0.5*minmod_dx(gphR, gp, gmhL);
                    double fphR = -0.5*minmod_dx(gphR2, gphR, gp);
                    double fmhL = 0.5*minmod_dx(gp, gmhL, gmhL2);
                    double fmhR = -0.5*minmod_dx(gphR, gp, gmhL);
                    qiphL[alpha] = gphL + fphL;
                    qiphR[alpha] = gphR + fphR;
                    qimhL[alpha] = gmhL + fmhL;
                    qimhR[alpha] = gmhR + fmhR;
                }
                // for each direction, reconstruct half-way cells
                // reconstruct e, rhob, and u[4] for half way cells
                flag = ReconstIt_velocity_Newton(
                                grid_array_hL, tau, qiphL, grid_array[idx]);
                aiphL = MaxSpeed(tau, direc, grid_array_hL);

                flag *= ReconstIt_velocity_Newton(
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

                    rhs[alpha] -= Fiph/DELTA_ETA*DELTA_TAU;
                }

                flag *= ReconstIt_velocity_Newton(grid_array_hL, tau,
                                                     qimhL, grid_array[idx]);
                aimhL = MaxSpeed(tau, direc, grid_array_hL);

                flag *= ReconstIt_velocity_Newton(grid_array_hR, tau,
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
                    rhs[alpha] += Fimh/DELTA_ETA*DELTA_TAU;
                }
                //cout << "eta-direction" << endl;

                // geometric terms
                rhs[0] -= (get_TJb_new(grid_array[idx], 3, 3)
                           *DELTA_TAU);
                rhs[3] -= (get_TJb_new(grid_array[idx], 3, 0)
                           *DELTA_TAU);
                
                for (int alpha = 0; alpha < 5; alpha++) {
                    qi_array_new[idx][alpha] = qi_array[idx][alpha] + rhs[alpha];
                }
            }
        }
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

void Advance::update_grid_cell(double grid_array[][5], Field *hydro_fields, int rk_flag,
                               int ieta, int ix, int iy,
                               int sub_grid_neta, int sub_grid_x, int sub_grid_y) {
    for (int k = 0; k < sub_grid_neta; k++) {
        int idx_ieta = MIN(ieta + k, GRID_SIZE_ETA - 1);
        for (int i = 0; i < sub_grid_x; i++) {
            int idx_ix = MIN(ix + i, GRID_SIZE_X);
            for (int j = 0; j < sub_grid_y; j++) {
                int idx_iy = MIN(iy + j, GRID_SIZE_Y);
                int field_idx = (idx_iy + idx_ix*(GRID_SIZE_Y+1)
                                 + idx_ieta*(GRID_SIZE_Y+1)*(GRID_SIZE_X+1));
                int idx = j + i*sub_grid_y + k*sub_grid_x*sub_grid_y;
                update_grid_array_to_hydro_fields(
                        grid_array[idx], hydro_fields, field_idx, rk_flag);
            }
        }
    }
}           

void Advance::update_grid_cell_viscous(double vis_array[][19], Field *hydro_fields,
        int rk_flag, int ieta, int ix, int iy, int sub_grid_neta, int sub_grid_x,
        int sub_grid_y) {

    int field_idx;
    int field_ny = GRID_SIZE_Y + 1;
    int field_nperp = (GRID_SIZE_Y + 1)*(GRID_SIZE_X + 1);

    for (int k = 0; k < sub_grid_neta; k++) {
        int idx_ieta = MIN(ieta + k, GRID_SIZE_ETA - 1);
        for (int i = 0; i < sub_grid_x; i++) {
            int idx_ix = MIN(ix + i, GRID_SIZE_X);
            for (int j = 0; j < sub_grid_y; j++) {
                int idx_iy = MIN(iy + j, GRID_SIZE_Y);
                int idx = j + i*sub_grid_y + k*sub_grid_x*sub_grid_y;
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

//! This function reverts the grid information back its values
//! at the previous time step
void Advance::revert_grid(double *grid_array, double *grid_prev) {
    for (int i = 0; i < 5; i++) {
        grid_array[i] = grid_prev[i];
    }
}

double Advance::get_pressure(double e_local, double rhob) {
    double p = e_local/3.;
    return(p);
}

double Advance::get_cs2(double e_local, double rhob) {
    double cs2 = 1./3.;
    return(cs2);
}

double Advance::p_e_func(double e_local, double rhob) {
    double dPde = 1./3.;
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
    double Nc = 3;
    double Nf = 2.5;
    double res = 90.0/M_PI/M_PI*(e_local/3.0)/(2*(Nc*Nc-1)+7./2*Nc*Nf);
    return(pow(res, 0.25));
}

double Advance::minmod_dx(double up1, double u, double um1) {
    double theta_flux = 1.8;
    double diffup = (up1 - u)*theta_flux;
    double diffdown = (u - um1)*theta_flux;
    double diffmid = (up1 - um1)*0.5;

    double tempf;
    if ( (diffup > 0.0) && (diffdown > 0.0) && (diffmid > 0.0) ) {
        tempf = mini(diffdown, diffmid);
        return mini(diffup, tempf);
    } else if ( (diffup < 0.0) && (diffdown < 0.0) && (diffmid < 0.0) ) {
        tempf = maxi(diffdown, diffmid);
        return maxi(diffup, tempf);
    } else {
      return 0.0;
    }
}/* minmod_dx */


void Advance::MakeWSource(double tau, double qi_array[][5],
                          int sub_grid_neta, int sub_grid_x, int sub_grid_y,
                          double vis_array[][19],
                          double vis_nbr_tau[][19], double vis_nbr_x[][19],
                          double vis_nbr_y[][19], double vis_nbr_eta[][19],
                          double qi_array_new[][5], InitData *DATA) {
//! calculate d_m (tau W^{m,alpha}) + (geom source terms)
//! partial_tau W^tau alpha
//! this is partial_tau evaluated at tau
//! this is the first step. so rk_flag = 0
//! change: alpha first which is the case
//!         for everywhere else. also, this change is necessary
//!         to use Wmunu[rk_flag][4][mu] as the dissipative baryon current

    double shear_on, bulk_on;
    if (DATA->turn_on_shear)
        shear_on = 1.0;
    else
        shear_on = 0.0;

    if (DATA->turn_on_bulk)
        bulk_on = 1.0;
    else
        bulk_on = 0.0;

    int alpha_max = 5;
    if (DATA->turn_on_diff == 0) {
        alpha_max = 4;
    }
    for (int k = 0; k < sub_grid_neta; k++) {
        for (int i = 0; i < sub_grid_x; i++) {
            for (int j = 0; j < sub_grid_y; j++) {
                int idx = j + i*sub_grid_y + k*sub_grid_x*sub_grid_y;
                for (int alpha = 0; alpha < alpha_max; alpha++) {
                    // dW/dtau
                    // backward time derivative (first order is more stable)
                    int idx_1d_alpha0 = map_2d_idx_to_1d(alpha, 0);
                    double dWdtau;
                    dWdtau = ((vis_array[idx][idx_1d_alpha0]
                               - vis_nbr_tau[idx][idx_1d_alpha0])
                              /DATA->delta_tau);

                    // bulk pressure term
                    double dPidtau = 0.0;
                    double Pi_alpha0 = 0.0;
                    if (alpha < 4 && DATA->turn_on_bulk == 1) {
                        double gfac = (alpha == 0 ? -1.0 : 0.0);
                        Pi_alpha0 = (vis_array[idx][14]
                                     *(gfac + vis_array[idx][15+alpha]
                                              *vis_array[idx][15]));

                        dPidtau = (Pi_alpha0
                                   - vis_nbr_tau[idx][14]
                                     *(gfac + vis_nbr_tau[idx][alpha+15]
                                              *vis_nbr_tau[idx][15]));
                    }

                    // use central difference to preserve
                    // the conservation law exactly
                    int idx_1d;
                    int idx_p_1, idx_m_1;
                    double dWdx_perp = 0.0;
                    double dPidx_perp = 0.0;

                    double sgp1, sgm1, bgp1, bgm1;
                    // x-direction
                    idx_1d = map_2d_idx_to_1d(alpha, 1);
                    if (i + 1 < sub_grid_x) {
                        idx_p_1 = j + (i+1)*sub_grid_y + k*sub_grid_x*sub_grid_y;
                        sgp1 = vis_array[idx_p_1][idx_1d];
                    } else {
                        idx_p_1 = 4*j + k*4*sub_grid_y + 2;
                        sgp1 = vis_nbr_x[idx_p_1][idx_1d];
                    }
                    if (i - 1 >= 0) {
                        idx_m_1 = j + (i-1)*sub_grid_y + k*sub_grid_x*sub_grid_y;
                        sgm1 = vis_array[idx_m_1][idx_1d];
                    } else {
                        idx_m_1 = 4*j + k*4*sub_grid_y + 1;
                        sgm1 = vis_nbr_x[idx_m_1][idx_1d];
                    }
                    dWdx_perp += (sgp1 - sgm1)/(2.*DATA->delta_x);
                    if (alpha < 4 && DATA->turn_on_bulk == 1) {
                        double gfac1 = (alpha == 1 ? 1.0 : 0.0);
                        if (i + 1 < sub_grid_x) {
                            bgp1 = (vis_array[idx_p_1][14]
                                        *(gfac1 + vis_array[idx_p_1][15+alpha]
                                                  *vis_nbr_x[idx_p_1][16]));
                        } else {
                            bgp1 = (vis_nbr_x[idx_p_1][14]
                                        *(gfac1 + vis_nbr_x[idx_p_1][15+alpha]
                                                  *vis_nbr_x[idx_p_1][16]));
                        }
                        if (i - 1 >= 0) {
                            bgm1 = (vis_array[idx_m_1][14]
                                        *(gfac1 + vis_array[idx_m_1][15+alpha]
                                                  *vis_array[idx_m_1][16]));
                        } else {
                            bgm1 = (vis_nbr_x[idx_m_1][14]
                                        *(gfac1 + vis_nbr_x[idx_m_1][15+alpha]
                                                  *vis_nbr_x[idx_m_1][16]));
                        }
                        dPidx_perp += (bgp1 - bgm1)/(2.*DATA->delta_x);
                    }
                    // y-direction
                    idx_1d = map_2d_idx_to_1d(alpha, 2);
                    if (j + 1 < sub_grid_y) {
                        idx_p_1 = j + 1 + i*sub_grid_y + k*sub_grid_x*sub_grid_y;
                        sgp1 = vis_array[idx_p_1][idx_1d];
                    } else {
                        idx_p_1 = 4*i + 4*k*sub_grid_x + 2;
                        sgp1 = vis_nbr_y[idx_p_1][idx_1d];
                    }
                    if (j - 1 >= 0) {
                        idx_m_1 = j - 1 + i*sub_grid_y + k*sub_grid_x*sub_grid_y;
                        sgm1 = vis_array[idx_m_1][idx_1d];
                    } else {
                        idx_m_1 = 4*i + 4*k*sub_grid_x + 1;
                        sgm1 = vis_nbr_y[idx_m_1][idx_1d];
                    }
                    dWdx_perp += (sgp1 - sgm1)/(2.*DATA->delta_x);
                    if (alpha < 4 && DATA->turn_on_bulk == 1) {
                        double gfac1 = (alpha == 2 ? 1.0 : 0.0);
                        if (j + 1 < sub_grid_x) {
                            bgp1 = (vis_array[idx_p_1][14]
                                        *(gfac1 + vis_nbr_y[idx_p_1][15+alpha]
                                                  *vis_nbr_y[idx_p_1][17]));
                        } else {
                            bgp1 = (vis_nbr_y[idx_p_1][14]
                                        *(gfac1 + vis_nbr_y[idx_p_1][15+alpha]
                                                  *vis_nbr_y[idx_p_1][17]));
                        }
                        if (j - 1 >= 0) {
                            bgm1 = (vis_array[idx_m_1][14]
                                        *(gfac1 + vis_array[idx_m_1][15+alpha]
                                                  *vis_array[idx_m_1][17]));
                        } else {
                            bgm1 = (vis_nbr_y[idx_m_1][14]
                                        *(gfac1 + vis_nbr_y[idx_m_1][15+alpha]
                                                  *vis_nbr_y[idx_m_1][17]));
                        }
                        dPidx_perp += (bgp1 - bgm1)/(2.*DATA->delta_x);
                    }

                    // eta-direction
                    double taufactor = tau;
                    double dWdeta = 0.0;
                    double dPideta = 0.0;
                    idx_1d = map_2d_idx_to_1d(alpha, 3);
                    if (k + 1 < sub_grid_neta) {
                        idx_p_1 = j + i*sub_grid_y + (k+1)*sub_grid_x*sub_grid_y;
                        sgp1 = vis_array[idx_p_1][idx_1d];
                    } else {
                        idx_p_1 = 4*j + 4*i*sub_grid_y + 2;
                        sgp1 = vis_nbr_eta[idx_p_1][idx_1d];
                    }
                    if (k - 1 >= 0) {
                        idx_m_1 = j + i*sub_grid_y + (k-1)*sub_grid_x*sub_grid_y;
                        sgm1 = vis_array[idx_m_1][idx_1d];
                    } else {
                        idx_m_1 = 4*j + 4*i*sub_grid_y + 1;
                        sgm1 = vis_nbr_eta[idx_m_1][idx_1d];
                    }
                    dWdeta = (sgp1 - sgm1)/(2.*DATA->delta_eta*taufactor);
                    if (alpha < 4 && DATA->turn_on_bulk == 1) {
                        double gfac3 = (alpha == 3 ? 1.0 : 0.0);
                        if (k + 1 < sub_grid_neta) {
                            bgp1 = (vis_array[idx_p_1][14]
                                       *(gfac3 + vis_array[idx_p_1][15+alpha]
                                                 *vis_array[idx_p_1][18]));
                        } else {
                            bgp1 = (vis_nbr_eta[idx_p_1][14]
                                       *(gfac3 + vis_nbr_eta[idx_p_1][15+alpha]
                                                 *vis_nbr_eta[idx_p_1][18]));
                        }
                        if (k - 1 >= 0) {
                            bgm1 = (vis_array[idx_m_1][14]
                                       *(gfac3 + vis_array[idx_m_1][15+alpha]
                                                 *vis_array[idx_m_1][18]));
                        } else {
                            bgm1 = (vis_nbr_eta[idx_m_1][14]
                                       *(gfac3 + vis_nbr_eta[idx_m_1][15+alpha]
                                                 *vis_nbr_eta[idx_m_1][18]));
                        }
                        dPideta = ((bgp1 - bgm1)
                                   /(2.*DATA->delta_eta*taufactor));
                    }

                    // partial_m (tau W^mn) = W^0n + tau partial_m W^mn
                    double sf = (tau*(dWdtau + dWdx_perp + dWdeta)
                                 + vis_array[idx][idx_1d_alpha0]);
                    double bf = (tau*(dPidtau + dPidx_perp + dPideta)
                                 + Pi_alpha0);

                    // sources due to coordinate transform
                    // this is added to partial_m W^mn
                    if (alpha == 0) {
                        sf += vis_array[idx][9];
                        bf += vis_array[idx][14]*(1.0 + vis_array[idx][18]
                                                        *vis_array[idx][18]);
                    }
                    if (alpha == 3) {
                        sf += vis_array[idx][3];
                        bf += vis_array[idx][14]*(vis_array[idx][15]
                                                  *vis_array[idx][18]);
                    }

                    double result = 0.0;
                    if (alpha < 4) {
                        result = (sf*shear_on + bf*bulk_on);
                    } else if (alpha == 4) {
                        result = sf;
                    }
                    qi_array_new[idx][alpha] -= result*(DATA->delta_tau);
                }
            }
        }
    }
}

int Advance::Make_uWRHS(double tau, int sub_grid_neta, int sub_grid_x, int sub_grid_y,
                        double vis_array[][19], double vis_nbr_x[][19],
                        double vis_nbr_y[][19], double vis_nbr_eta[][19],
                        double velocity_array[][20],
                        double vis_array_new[][19]) {

    if (DATA_ptr->turn_on_shear == 0)
        return(1);

    for (int k = 0; k < sub_grid_neta; k++) {
        for (int i = 0; i < sub_grid_x; i++) {
            for (int j = 0; j < sub_grid_y; j++) {
                int idx = j + i*sub_grid_y + k*sub_grid_x*sub_grid_y;

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
    
                double u0 = vis_array[idx][15];
                double u1 = vis_array[idx][16];
                double u2 = vis_array[idx][17];
                double u3 = vis_array[idx][18];

                double taufactor;
                double g, gp1, gm1, gp2, gm2, a, am1, ap1, ax;
                double f, fp1, fm1, fp2, fm2;
                double uWphR, uWphL, uWmhR, uWmhL, WphR, WphL, WmhR, WmhL;
                double HWph, HWmh, HW;
                int idx_p_2, idx_p_1, idx_m_1, idx_m_2;
                double sum;
                for (unsigned int idx_1d = 4; idx_1d < 15; idx_1d++) {
                    // the derivative part is the same for all viscous
                    // components
                    vis_array_new[idx][idx_1d] = vis_array[idx][idx_1d]*u0;

                    sum = 0.0;
                    // x-direction
                    taufactor = 1.0;
                    /* Get_uWmns */
                    g = vis_array[idx][idx_1d]*u0;
                    f = vis_array[idx][idx_1d]*u1;
                    a = fabs(u1)/u0;

                    if (i + 2 < sub_grid_x) {
                        idx_p_2 = j + (i+2)*sub_grid_y + k*sub_grid_x*sub_grid_y;
                        gp2 = vis_array[idx_p_2][idx_1d];
                        fp2 = gp2*vis_array[idx_p_2][16];
                        gp2 *= vis_array[idx_p_2][15];
                    } else {
                        idx_p_2 = 4*j + k*4*sub_grid_y + 4 + i - sub_grid_x;
                        gp2 = vis_nbr_x[idx_p_2][idx_1d];
                        fp2 = gp2*vis_nbr_x[idx_p_2][16];
                        gp2 *= vis_nbr_x[idx_p_2][15];
                    }

                    if (i + 1 < sub_grid_x) {
                        idx_p_1 = j + (i+1)*sub_grid_y + k*sub_grid_x*sub_grid_y;
                        gp1 = vis_array[idx_p_1][idx_1d];
                        fp1 = gp1*vis_array[idx_p_1][16];
                        gp1 *= vis_array[idx_p_1][15];
                        ap1 = (fabs(vis_array[idx_p_1][16])
                               /vis_array[idx_p_1][15]);
                    } else {
                        idx_p_1 = 4*j + k*4*sub_grid_y + 2;
                        gp1 = vis_nbr_x[idx_p_1][idx_1d];
                        fp1 = gp1*vis_nbr_x[idx_p_1][16];
                        gp1 *= vis_nbr_x[idx_p_1][15];
                        ap1 = (fabs(vis_nbr_x[idx_p_1][16])
                               /vis_nbr_x[idx_p_1][15]);
                    }

                    if (i - 1 >= 0) {
                        idx_m_1 = j + (i-1)*sub_grid_y + k*sub_grid_x*sub_grid_y;
                        gm1 = vis_array[idx_m_1][idx_1d];
                        fm1 = gm1*vis_array[idx_m_1][16];
                        gm1 *= vis_array[idx_m_1][15];
                        am1 = (fabs(vis_array[idx_m_1][16])
                               /vis_array[idx_m_1][15]);
                    } else {
                        idx_m_1 = 4*j + k*4*sub_grid_y + 1;
                        gm1 = vis_nbr_x[idx_m_1][idx_1d];
                        fm1 = gm1*vis_nbr_x[idx_m_1][16];
                        gm1 *= vis_nbr_x[idx_m_1][15];
                        am1 = (fabs(vis_nbr_x[idx_m_1][16])
                               /vis_nbr_x[idx_m_1][15]);
                    }

                    if (i - 2 >= 0) {
                        idx_m_2 = j + (i-2)*sub_grid_y + k*sub_grid_x*sub_grid_y;
                        gm2 = vis_array[idx_m_2][idx_1d];
                        fm2 = gm2*vis_array[idx_m_2][16];
                        gm2 *= vis_array[idx_m_2][15];
                    } else {
                        idx_m_2 = 4*j + k*4*sub_grid_y + i;
                        gm2 = vis_nbr_x[idx_m_2][idx_1d];
                        fm2 = gm2*vis_nbr_x[idx_m_2][16];
                        gm2 *= vis_nbr_x[idx_m_2][15];
                    }
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
                    HW = (HWph - HWmh)/DATA_ptr->delta_x/taufactor;
                        
                    // make partial_i (u^i Wmn)
                    sum += -HW;
            
                    // y-direction
                    taufactor = 1.0;
                    /* Get_uWmns */
                    g = vis_array[idx][idx_1d]*u0;
                    f = vis_array[idx][idx_1d]*u2;
                    a = fabs(u2)/u0;

                    if (j + 2 < sub_grid_y) {
                        idx_p_2 = j + 2 + i*sub_grid_y + k*sub_grid_x*sub_grid_y;
                        gp2 = vis_array[idx_p_2][idx_1d];
                        fp2 = gp2*vis_array[idx_p_2][17];
                        gp2 *= vis_array[idx_p_2][15];
                    } else {
                        idx_p_2 = 4*i + 4*k*sub_grid_x + 4 + j - sub_grid_y;
                        gp2 = vis_nbr_y[idx_p_2][idx_1d];
                        fp2 = gp2*vis_nbr_y[idx_p_2][17];
                        gp2 *= vis_nbr_y[idx_p_2][15];
                    }

                    if (j + 1 < sub_grid_y) {
                        idx_p_1 = j + 1 + i*sub_grid_y + k*sub_grid_x*sub_grid_y;
                        gp1 = vis_array[idx_p_1][idx_1d];
                        fp1 = gp1*vis_array[idx_p_1][17];
                        gp1 *= vis_array[idx_p_1][15];
                        ap1 = (fabs(vis_array[idx_p_1][17])
                               /vis_array[idx_p_1][15]);
                    } else {
                        idx_p_1 = 4*i + 4*k*sub_grid_x + 2;
                        gp1 = vis_nbr_y[idx_p_1][idx_1d];
                        fp1 = gp1*vis_nbr_y[idx_p_1][17];
                        gp1 *= vis_nbr_y[idx_p_1][15];
                        ap1 = (fabs(vis_nbr_y[idx_p_1][17])
                               /vis_nbr_y[idx_p_1][15]);
                    }
                        
                    if (j - 1 >= 0) {
                        idx_m_1 = j - 1 + i*sub_grid_y + k*sub_grid_x*sub_grid_y;
                        gm1 = vis_array[idx_m_1][idx_1d];
                        fm1 = gm1*vis_array[idx_m_1][17];
                        gm1 *= vis_array[idx_m_1][15];
                        am1 = (fabs(vis_array[idx_m_1][17])
                               /vis_array[idx_m_1][15]);
                    } else {
                        idx_m_1 = 4*i + 4*k*sub_grid_x + 1;
                        gm1 = vis_nbr_y[idx_m_1][idx_1d];
                        fm1 = gm1*vis_nbr_y[idx_m_1][17];
                        gm1 *= vis_nbr_y[idx_m_1][15];
                        am1 = (fabs(vis_nbr_y[idx_m_1][17])
                               /vis_nbr_y[idx_m_1][15]);
                    }

                    if (j - 2 >= 0) {
                        idx_m_2 = j - 2 + i*sub_grid_y + k*sub_grid_x*sub_grid_y;
                        gm2 = vis_array[idx_m_2][idx_1d];
                        fm2 = gm2*vis_array[idx_m_2][17];
                        gm2 *= vis_array[idx_m_2][15];
                    } else {
                        idx_m_2 = 4*i + 4*k*sub_grid_x + j;
                        gm2 = vis_nbr_y[idx_m_2][idx_1d];
                        fm2 = gm2*vis_nbr_y[idx_m_2][17];
                        gm2 *= vis_nbr_y[idx_m_2][15];
                    }
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
                    HW = (HWph - HWmh)/DATA_ptr->delta_y/taufactor;
                    // make partial_i (u^i Wmn)
                    sum += -HW;
            
                    // eta-direction
                    taufactor = tau;
                    /* Get_uWmns */
                    g = vis_array[idx][idx_1d]*u0;
                    f = vis_array[idx][idx_1d]*u3;
                    a = fabs(u3)/u0;

                    if (k + 2 < sub_grid_neta) {
                        idx_p_2 = j + i*sub_grid_y + (k+2)*sub_grid_x*sub_grid_y;
                        gp2 = vis_array[idx_p_2][idx_1d];
                        fp2 = gp2*vis_array[idx_p_2][18];
                        gp2 *= vis_array[idx_p_2][15];
                    } else {
                        idx_p_2 = 4*j + 4*i*sub_grid_y + 4 + k - sub_grid_neta;
                        gp2 = vis_nbr_eta[idx_p_2][idx_1d];
                        fp2 = gp2*vis_nbr_eta[idx_p_2][18];
                        gp2 *= vis_nbr_eta[idx_p_2][15];
                    }

                    if (k + 1 < sub_grid_neta) {
                        idx_p_1 = j + i*sub_grid_y + (k+1)*sub_grid_x*sub_grid_y;
                        gp1 = vis_array[idx_p_1][idx_1d];
                        fp1 = gp1*vis_array[idx_p_1][18];
                        gp1 *= vis_array[idx_p_1][15];
                        ap1 = (fabs(vis_array[idx_p_1][18])
                                /vis_array[idx_p_1][15]);
                    } else {
                        idx_p_1 = 4*j + 4*i*sub_grid_y + 2;
                        gp1 = vis_nbr_eta[idx_p_1][idx_1d];
                        fp1 = gp1*vis_nbr_eta[idx_p_1][18];
                        gp1 *= vis_nbr_eta[idx_p_1][15];
                        ap1 = (fabs(vis_nbr_eta[idx_p_1][18])
                                /vis_nbr_eta[idx_p_1][15]);
                    }

                    if (k - 1 >= 0) {
                        idx_m_1 = j + i*sub_grid_y + (k-1)*sub_grid_x*sub_grid_y;
                        gm1 = vis_array[idx_m_1][idx_1d];
                        fm1 = gm1*vis_array[idx_m_1][18];
                        gm1 *= vis_array[idx_m_1][15];
                        am1 = (fabs(vis_array[idx_m_1][18])
                                /vis_array[idx_m_1][15]);
                    } else {
                        idx_m_1 = 4*j + 4*i*sub_grid_y + 1;
                        gm1 = vis_nbr_eta[idx_m_1][idx_1d];
                        fm1 = gm1*vis_nbr_eta[idx_m_1][18];
                        gm1 *= vis_nbr_eta[idx_m_1][15];
                        am1 = (fabs(vis_nbr_eta[idx_m_1][18])
                                /vis_nbr_eta[idx_m_1][15]);
                    }

                    if (k - 2 >= 0) {
                        idx_m_2 = j + i*sub_grid_y + (k-2)*sub_grid_x*sub_grid_y;
                        gm2 = vis_array[idx_m_2][idx_1d];
                        fm2 = gm2*vis_array[idx_m_2][18];
                        gm2 *= vis_array[idx_m_2][15];
                    } else {
                        idx_m_2 = 4*j + 4*i*sub_grid_y + k;
                        gm2 = vis_nbr_eta[idx_m_2][idx_1d];
                        fm2 = gm2*vis_nbr_eta[idx_m_2][18];
                        gm2 *= vis_nbr_eta[idx_m_2][15];
                    }
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
                    HW = (HWph - HWmh)/DATA_ptr->delta_eta/taufactor;
                    // make partial_i (u^i Wmn)
                    sum += -HW;
                    
                    //w_rhs[mu][nu] = sum*(DELTA_TAU);
                    vis_array_new[idx][idx_1d] += (
                                        sum*(DELTA_TAU));
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
                sum = (- (u0*vis_array[idx][4])/tau
                       + (velocity_array[idx][0]*vis_array[idx][4]));
                tempf = ((u3/tau)*2.*u1*(vis_array[idx][6]*u0
                                         - vis_array[idx][1]*u3));
                tempf += 2.*(
                    - velocity_array[idx][1]*(vis_array[idx][1]*u1)
                    + velocity_array[idx][2]*(vis_array[idx][4]*u1)
                    + velocity_array[idx][3]*(vis_array[idx][5]*u1)
                    + velocity_array[idx][4]*(vis_array[idx][6]*u1));
                vis_array_new[idx][4] += (sum + tempf)*DELTA_TAU;

                // W^12
                sum = (- (u0*vis_array[idx][5])/tau
                       + (velocity_array[idx][0]*vis_array[idx][5]));
                tempf = ((u3/tau)*((vis_array[idx][8]*u1
                                    + vis_array[idx][6]*u2)*u0
                                   - (vis_array[idx][1]*u2
                                      + vis_array[idx][2]*u1)*u3));
                tempf += (
                    - velocity_array[idx][1]*(vis_array[idx][2]*u1
                                              + vis_array[idx][1]*u2)
                    + velocity_array[idx][2]*(vis_array[idx][5]*u1
                                              + vis_array[idx][4]*u2)
                    + velocity_array[idx][3]*(vis_array[idx][7]*u1
                                              + vis_array[idx][5]*u2)
                    + velocity_array[idx][4]*(vis_array[idx][8]*u1
                                              + vis_array[idx][6]*u2));
                vis_array_new[idx][5] += (sum + tempf)*DELTA_TAU;

                
                // W^13
                sum = (- (u0*vis_array[idx][6])/tau
                       + (velocity_array[idx][0]*vis_array[idx][6]));
                tempf = ((u3/tau)*(- vis_array[idx][1]
                                   + (vis_array[idx][9]*u1
                                      + vis_array[idx][6]*u3)*u0
                                   - (vis_array[idx][1]*u3
                                      + vis_array[idx][3]*u1)*u3));
                tempf += (
                    - velocity_array[idx][1]*(vis_array[idx][3]*u1
                                              + vis_array[idx][1]*u3)
                    + velocity_array[idx][2]*(vis_array[idx][6]*u1
                                              + vis_array[idx][4]*u3)
                    + velocity_array[idx][3]*(vis_array[idx][8]*u1
                                              + vis_array[idx][5]*u3)
                    + velocity_array[idx][4]*(vis_array[idx][9]*u1
                                              + vis_array[idx][6]*u3));
                vis_array_new[idx][6] += (sum + tempf)*DELTA_TAU;
                
                // W^22
                sum = (- (u0*vis_array[idx][7])/tau
                       + (velocity_array[idx][0]*vis_array[idx][7]));
                tempf = ((u3/tau)*2.*u2*(vis_array[idx][8]*u0
                                         - vis_array[idx][2]*u3));
                tempf += 2.*(
                    - velocity_array[idx][1]*(vis_array[idx][2]*u2)
                    + velocity_array[idx][2]*(vis_array[idx][5]*u2)
                    + velocity_array[idx][3]*(vis_array[idx][7]*u2)
                    + velocity_array[idx][4]*(vis_array[idx][8]*u2));
                vis_array_new[idx][7] += (sum + tempf)*DELTA_TAU;
                
                // W^23
                sum = (- (u0*vis_array[idx][8])/tau
                       + (velocity_array[idx][0]*vis_array[idx][8]));
                tempf = ((u3/tau)*(- vis_array[idx][2]
                                   + (vis_array[idx][9]*u2
                                      + vis_array[idx][8]*u3)*u0
                                   - (vis_array[idx][2]*u3
                                      + vis_array[idx][3]*u2)*u3));
                tempf += (
                    - velocity_array[idx][1]*(vis_array[idx][2]*u3
                                              + vis_array[idx][3]*u2)
                    + velocity_array[idx][2]*(vis_array[idx][5]*u3
                                              + vis_array[idx][6]*u2)
                    + velocity_array[idx][3]*(vis_array[idx][7]*u3
                                              + vis_array[idx][8]*u2)
                    + velocity_array[idx][4]*(vis_array[idx][8]*u3
                                              + vis_array[idx][9]*u2));
                vis_array_new[idx][8] += (sum + tempf)*DELTA_TAU;

                // W^33
                sum = (- (u0*vis_array[idx][9])/tau
                       + (velocity_array[idx][0]*vis_array[idx][9]));
                tempf = ((u3/tau)*2.*(u3*(vis_array[idx][9]*u0
                                          - vis_array[idx][3]*u3)
                                      - vis_array[idx][3]));
                tempf += 2.*(
                    - velocity_array[idx][1]*(vis_array[idx][3]*u3)
                    + velocity_array[idx][2]*(vis_array[idx][6]*u3)
                    + velocity_array[idx][3]*(vis_array[idx][8]*u3)
                    + velocity_array[idx][4]*(vis_array[idx][9]*u3));
                vis_array_new[idx][9] += (sum + tempf)*DELTA_TAU;

                // bulk pressure (idx_1d == 14)
                // geometric terms for bulk Pi
                //sum -= (pi_b[rk_flag])*(u[rk_flag][0])/tau;
                //sum += (pi_b[rk_flag])*theta_local;
                vis_array_new[idx][14] += (
                    (- vis_array[idx][14]*vis_array[idx][15]/tau
                     + vis_array[idx][14]*velocity_array[idx][0])
                    *(DELTA_TAU));
            }
        }
    }
    return(1);
}

double Advance::Make_uWSource(double tau, int sub_grid_neta, int sub_grid_x,
                              int sub_grid_y, double vis_array[][19],
                              double velocity_array[][20],
                              double grid_array[][5],
                              double vis_array_new[][19]) {
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
    for (int k = 0; k < sub_grid_neta; k++) {
        for (int i = 0; i < sub_grid_x; i++) {
            for (int j = 0; j < sub_grid_y; j++) {
                int idx = j + i*sub_grid_y + k*sub_grid_x*sub_grid_y;

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
                double epsilon = grid_array[idx][0];
                double rhob = grid_array[idx][4];

                double T = get_temperature(epsilon, rhob);

                double shear_to_s = 0.0;
                    shear_to_s = DATA_ptr->shear_to_s;


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
                        - (1.0 + transport_coefficient2*velocity_array[idx][0])
                          *(vis_array[idx][idx_1d]));

                    // Navier-Stokes Term -- -2.*shear*sigma^munu
                    // full Navier-Stokes term is
                    // sign changes according to metric sign convention
                    double NS_term = - 2.*shear*velocity_array[idx][6+idx_1d];

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
                               vis_array[idx][0]*velocity_array[idx][6]
                             + vis_array[idx][4]*velocity_array[idx][10]
                             + vis_array[idx][7]*velocity_array[idx][13]
                             + vis_array[idx][9]*velocity_array[idx][15]
                             //- 2.*(  Wmunu[0][1]*sigma[0][1]
                             //      + Wmunu[0][2]*sigma[0][2]
                             //      + Wmunu[0][3]*sigma[0][3])
                             - 2.*(  vis_array[idx][1]*velocity_array[idx][7]
                                   + vis_array[idx][2]*velocity_array[idx][8]
                                   + vis_array[idx][3]*velocity_array[idx][9])
                             //+2.*(  Wmunu[1][2]*sigma[1][2]
                             //     + Wmunu[1][3]*sigma[1][3]
                             //     + Wmunu[2][3]*sigma[2][3]));
                             +2.*(  vis_array[idx][5]*velocity_array[idx][11]
                                  + vis_array[idx][6]*velocity_array[idx][12]
                                  + vis_array[idx][8]*velocity_array[idx][14]));

                        //term1_Wsigma = ( - Wmunu[mu][0]*sigma[nu][0]
                        //                 - Wmunu[nu][0]*sigma[mu][0]
                        //                 + Wmunu[mu][1]*sigma[nu][1]
                        //                 + Wmunu[nu][1]*sigma[mu][1]
                        //                 + Wmunu[mu][2]*sigma[nu][2]
                        //                 + Wmunu[nu][2]*sigma[mu][2]
                        //                 + Wmunu[mu][3]*sigma[nu][3]
                        //                 + Wmunu[nu][3]*sigma[mu][3])/2.;
                        //term2_Wsigma = (-(1./3.)*(DATA_ptr->gmunu[mu][nu]
                        //                          + vis_array[idx][15+mu]
                        //                            *vis_array[idx][15+nu])
                        //                         *Wsigma);
                        if (idx_1d == 4) {  // pi^xx
                            term1_Wsigma = (
                                - vis_array[idx][1]*velocity_array[idx][7]
                                + vis_array[idx][4]*velocity_array[idx][10]
                                + vis_array[idx][5]*velocity_array[idx][11]
                                + vis_array[idx][6]*velocity_array[idx][12]);
                            term2_Wsigma = (-(1./3.)*(1.+ vis_array[idx][16]
                                                          *vis_array[idx][16])
                                                     *Wsigma);
                        } else if (idx_1d == 5) {  // pi^xy
                            term1_Wsigma = 0.5*(
                                - (vis_array[idx][1]*velocity_array[idx][8]
                                    + vis_array[idx][2]*velocity_array[idx][7])
                                + (vis_array[idx][4]*velocity_array[idx][11]
                                    + vis_array[idx][5]*velocity_array[idx][10])
                                + (vis_array[idx][5]*velocity_array[idx][13]
                                    + vis_array[idx][7]*velocity_array[idx][11])
                                + (vis_array[idx][6]*velocity_array[idx][14]
                                    + vis_array[idx][8]*velocity_array[idx][12])
                            );
                            term2_Wsigma = (-(1./3.)*(vis_array[idx][16]
                                                      *vis_array[idx][17])
                                                     *Wsigma);
                        } else if (idx_1d == 6) {  // pi^xeta
                            term1_Wsigma = 0.5*(
                                - (vis_array[idx][1]*velocity_array[idx][9]
                                    + vis_array[idx][3]*velocity_array[idx][7])
                                + (vis_array[idx][4]*velocity_array[idx][12]
                                    + vis_array[idx][6]*velocity_array[idx][10])
                                + (vis_array[idx][5]*velocity_array[idx][14]
                                    + vis_array[idx][8]*velocity_array[idx][11])
                                + (vis_array[idx][6]*velocity_array[idx][15]
                                    + vis_array[idx][9]*velocity_array[idx][12])
                            );
                            term2_Wsigma = (-(1./3.)*(vis_array[idx][16]
                                                          *vis_array[idx][18])
                                                     *Wsigma);
                        } else if (idx_1d == 7) {  // pi^yy
                            term1_Wsigma = (
                                - vis_array[idx][2]*velocity_array[idx][8]
                                + vis_array[idx][5]*velocity_array[idx][11]
                                + vis_array[idx][7]*velocity_array[idx][13]
                                + vis_array[idx][8]*velocity_array[idx][14]);
                            term2_Wsigma = (-(1./3.)*(1.+ vis_array[idx][17]
                                                          *vis_array[idx][17])
                                                     *Wsigma);
                        } else if (idx_1d == 8) {  // pi^yeta
                            term1_Wsigma = 0.5*(
                                - (vis_array[idx][2]*velocity_array[idx][9]
                                    + vis_array[idx][3]*velocity_array[idx][8])
                                + (vis_array[idx][5]*velocity_array[idx][12]
                                    + vis_array[idx][6]*velocity_array[idx][11])
                                + (vis_array[idx][7]*velocity_array[idx][14]
                                    + vis_array[idx][8]*velocity_array[idx][13])
                                + (vis_array[idx][8]*velocity_array[idx][15]
                                    + vis_array[idx][9]*velocity_array[idx][14])
                            );
                            term2_Wsigma = (-(1./3.)*(vis_array[idx][17]
                                                          *vis_array[idx][18])
                                                     *Wsigma);
                        } else if (idx_1d == 9) {  // pi^etaeta
                            term1_Wsigma = (
                                - vis_array[idx][3]*velocity_array[idx][9]
                                + vis_array[idx][6]*velocity_array[idx][12]
                                + vis_array[idx][8]*velocity_array[idx][14]
                                + vis_array[idx][9]*velocity_array[idx][15]);
                            term2_Wsigma = (-(1./3.)*(1.+ vis_array[idx][18]
                                                          *vis_array[idx][18])
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
                        Wsquare = (  vis_array[idx][0]*vis_array[idx][0]
                                   + vis_array[idx][4]*vis_array[idx][4]
                                   + vis_array[idx][7]*vis_array[idx][7]
                                   + vis_array[idx][9]*vis_array[idx][9]
                            - 2.*(  vis_array[idx][1]*vis_array[idx][1]
                                  + vis_array[idx][2]*vis_array[idx][2]
                                  + vis_array[idx][3]*vis_array[idx][3])
                            + 2.*(  vis_array[idx][5]*vis_array[idx][5]
                                  + vis_array[idx][6]*vis_array[idx][6]
                                  + vis_array[idx][8]*vis_array[idx][8]));

                        //term1_WW = ( - Wmunu[mu][0]*Wmunu[nu][0]
                        //             + Wmunu[mu][1]*Wmunu[nu][1]
                        //             + Wmunu[mu][2]*Wmunu[nu][2]
                        //             + Wmunu[mu][3]*Wmunu[nu][3]);
                        //term2_WW = (
                        //    -(1./3.)*(DATA_ptr->gmunu[mu][nu]
                        //              + vis_array[idx][15+mu]
                        //                *vis_array[idx][15+nu])
                        //    *Wsquare);
                        if (idx_1d == 4) {  // pi^xx
                            term1_WW = (
                                - vis_array[idx][1]*vis_array[idx][1]
                                + vis_array[idx][4]*vis_array[idx][4]
                                + vis_array[idx][5]*vis_array[idx][5]
                                + vis_array[idx][6]*vis_array[idx][6]);
                            term2_WW = (- (1./3.)*(1.+ vis_array[idx][16]
                                                       *vis_array[idx][16])
                                                  *Wsquare);
                        } else if (idx_1d == 5) {  // pi^xy
                            term1_WW = (
                                - vis_array[idx][1]*vis_array[idx][2]
                                + vis_array[idx][4]*vis_array[idx][5]
                                + vis_array[idx][5]*vis_array[idx][7]
                                + vis_array[idx][6]*vis_array[idx][8]);
                            term2_WW = (- (1./3.)*(vis_array[idx][16]
                                                       *vis_array[idx][17])
                                                  *Wsquare);
                        } else if (idx_1d == 6) {  // pi^xeta
                            term1_WW = (
                                - vis_array[idx][1]*vis_array[idx][3]
                                + vis_array[idx][4]*vis_array[idx][6]
                                + vis_array[idx][5]*vis_array[idx][8]
                                + vis_array[idx][6]*vis_array[idx][9]);
                            term2_WW = (- (1./3.)*(vis_array[idx][16]
                                                       *vis_array[idx][18])
                                                  *Wsquare);
                        } else if (idx_1d == 7) {  // pi^yy
                            term1_WW = (
                                - vis_array[idx][2]*vis_array[idx][2]
                                + vis_array[idx][5]*vis_array[idx][5]
                                + vis_array[idx][7]*vis_array[idx][7]
                                + vis_array[idx][8]*vis_array[idx][8]);
                            term2_WW = (- (1./3.)*(1.+ vis_array[idx][17]
                                                       *vis_array[idx][17])
                                                  *Wsquare);
                        } else if (idx_1d == 8) {  // pi^yeta
                            term1_WW = (
                                - vis_array[idx][2]*vis_array[idx][3]
                                + vis_array[idx][5]*vis_array[idx][6]
                                + vis_array[idx][7]*vis_array[idx][8]
                                + vis_array[idx][8]*vis_array[idx][9]);
                            term2_WW = (- (1./3.)*(vis_array[idx][17]
                                                       *vis_array[idx][18])
                                                  *Wsquare);
                        } else if (idx_1d == 9) {  // pi^etaeta
                            term1_WW = (
                                - vis_array[idx][3]*vis_array[idx][3]
                                + vis_array[idx][6]*vis_array[idx][6]
                                + vis_array[idx][8]*vis_array[idx][8]
                                + vis_array[idx][9]*vis_array[idx][9]);
                            term2_WW = (- (1./3.)*(1.+ vis_array[idx][18]
                                                       *vis_array[idx][18])
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
                    Bulk_Sigma = (vis_array[idx][14]
                                    *velocity_array[idx][6+idx_1d]);
                    Bulk_W = vis_array[idx][14]*vis_array[idx][idx_1d];

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
                    vis_array_new[idx][idx_1d] += SW*(DELTA_TAU);
                }
            }
        }
    }
    return(0);
}


double Advance::Make_uPiSource(double tau, int sub_grid_neta, int sub_grid_x,
                               int sub_grid_y, double vis_array[][19],
                               double velocity_array[][20],
                               double grid_array[][5],
                               double vis_array_new[][19]) {
    // switch to include non-linear coupling terms in the bulk pi evolution
    int include_BBterm = 1;
    int include_coupling_to_shear = 1;
 
    for (int k = 0; k < sub_grid_neta; k++) {
        for (int i = 0; i < sub_grid_x; i++) {
            for (int j = 0; j < sub_grid_y; j++) {
                int idx = j + i*sub_grid_y + k*sub_grid_x*sub_grid_y;
                // defining bulk viscosity coefficient
                double epsilon = grid_array[idx][0];
                double rhob = grid_array[idx][4];
                double temperature = get_temperature(epsilon, rhob);

                // cs2 is the velocity of sound squared
                double cs2 = get_cs2(epsilon, rhob);  
                double pressure = get_pressure(epsilon, rhob);

                // T dependent bulk viscosity from Gabriel
                double bulk =
                            get_temperature_dependent_zeta_s(temperature);
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
                double NS_term = -bulk*velocity_array[idx][0];

                // Computing relaxation term and nonlinear term:
                // - Bulk - transport_coeff1*Bulk*theta
                //double tempf = (-(grid_pt->pi_b[rk_flag])
                //         - transport_coeff1*theta_local
                //           *(grid_pt->pi_b[rk_flag]));
                double tempf = (- vis_array[idx][14]
                                - transport_coeff1*velocity_array[idx][0]
                                  *vis_array[idx][14]);

                // Computing nonlinear term: + transport_coeff2*Bulk*Bulk
                double BB_term = 0.0;
                if (include_BBterm == 1) {
                    //BB_term = (transport_coeff2*(grid_pt->pi_b[rk_flag])
                    //           *(grid_pt->pi_b[rk_flag]));
                    BB_term = (transport_coeff2*vis_array[idx][14]
                               *vis_array[idx][14]);
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
                    //        sigma[a][b] = velocity_array[idx][6+idx_1d];
                    //        Wmunu[a][b] = vis_array[idx][idx_1d];
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
                    Wsigma = (  vis_array[idx][0]*velocity_array[idx][6]
                              + vis_array[idx][4]*velocity_array[idx][10]
                              + vis_array[idx][7]*velocity_array[idx][13]
                              + vis_array[idx][9]*velocity_array[idx][15]
                              - 2.*(  vis_array[idx][1]*velocity_array[idx][7]
                                    + vis_array[idx][2]*velocity_array[idx][8]
                                    + vis_array[idx][3]*velocity_array[idx][9])
                              + 2.*(  vis_array[idx][5]*velocity_array[idx][11]
                                    + vis_array[idx][6]*velocity_array[idx][12]
                                    + vis_array[idx][8]*velocity_array[idx][14])
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
                    WW = (  vis_array[idx][0]*vis_array[idx][0]
                          + vis_array[idx][4]*vis_array[idx][4]
                          + vis_array[idx][8]*vis_array[idx][8]
                          + vis_array[idx][9]*vis_array[idx][9]
                          - 2.*(  vis_array[idx][1]*vis_array[idx][1]
                                + vis_array[idx][2]*vis_array[idx][2]
                                + vis_array[idx][3]*vis_array[idx][3])
                          + 2.*(  vis_array[idx][5]*vis_array[idx][5]
                                + vis_array[idx][6]*vis_array[idx][6]
                                + vis_array[idx][8]*vis_array[idx][8]));
                    // multiply term by its transport coefficient
                    Shear_Sigma_term = Wsigma*transport_coeff1_s;
                    Shear_Shear_term = WW*transport_coeff2_s;

                    // full term that couples to shear is
                    Coupling_to_Shear = (- Shear_Sigma_term
                                         + Shear_Shear_term);
                } else {
                    Coupling_to_Shear = 0.0;
                }
                
                // Final Answer
                double Final_Answer = (NS_term + tempf + BB_term
                                        + Coupling_to_Shear)/Bulk_Relax_time;
                vis_array_new[idx][14] += Final_Answer*(DELTA_TAU);
            }
        }
    }
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

//! this function returns the expansion rate on the grid
double Advance::calculate_expansion_rate_1(
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

void Advance::calculate_Du_supmu_1(double tau, Field *hydro_fields,
                                        int idx, int rk_flag, double *a) {
    // the array idx is corresponds to the velocity array in advanced
    if (rk_flag == 0) {
        for (int mu = 0; mu < 5; mu++) {
            a[1+mu] = (
                - hydro_fields->u_rk0[idx][0]*hydro_fields->dUsup[idx][4*mu]
                + hydro_fields->u_rk0[idx][1]*hydro_fields->dUsup[idx][4*mu+1]
                + hydro_fields->u_rk0[idx][2]*hydro_fields->dUsup[idx][4*mu+2]
                + hydro_fields->u_rk0[idx][3]*hydro_fields->dUsup[idx][4*mu+3]
            );
        }
    } else {
        for (int mu = 0; mu < 5; mu++) {
            a[1+mu] = (
                - hydro_fields->u_rk1[idx][0]*hydro_fields->dUsup[idx][4*mu]
                + hydro_fields->u_rk1[idx][1]*hydro_fields->dUsup[idx][4*mu+1]
                + hydro_fields->u_rk1[idx][2]*hydro_fields->dUsup[idx][4*mu+2]
                + hydro_fields->u_rk1[idx][3]*hydro_fields->dUsup[idx][4*mu+3]
            );
        }
    }
}

void Advance::calculate_velocity_shear_tensor_2(
                    double tau, Field *hydro_fields, int idx, int rk_flag,
                    double *velocity_array) {
    double theta_u_local = velocity_array[0];
    double u0, u1, u2, u3;
    if (rk_flag == 0) {
        u0 = hydro_fields->u_rk0[idx][0];
        u1 = hydro_fields->u_rk0[idx][1];
        u2 = hydro_fields->u_rk0[idx][2];
        u3 = hydro_fields->u_rk0[idx][3];
    } else {
        u0 = hydro_fields->u_rk1[idx][0];
        u1 = hydro_fields->u_rk1[idx][1];
        u2 = hydro_fields->u_rk1[idx][2];
        u3 = hydro_fields->u_rk1[idx][3];
    }
    // sigma^11
    velocity_array[10] = (
        hydro_fields->dUsup[idx][5] - (1. + u1*u1)*theta_u_local/3.
        + (u1*velocity_array[2]));
    // sigma^12
    velocity_array[11] = (
        (hydro_fields->dUsup[idx][6] + hydro_fields->dUsup[idx][9])/2.
        - (0. + u1*u2)*theta_u_local/3.
        + (u1*velocity_array[3] + u2*velocity_array[2])/2.);
    // sigma^13
    velocity_array[12] = (
        (hydro_fields->dUsup[idx][7] + hydro_fields->dUsup[idx][13])/2.
        - (0. + u1*u3)*theta_u_local/3.
        + (u1*velocity_array[4] + u3*velocity_array[2])/2.
        + u3*u0/(2.*tau)*u1);
    // sigma^22
    velocity_array[13] = (
        hydro_fields->dUsup[idx][10] - (1. + u2*u2)*theta_u_local/3.
        + u2*velocity_array[3]);
    // sigma^23
    velocity_array[14] = (
        (hydro_fields->dUsup[idx][11] + hydro_fields->dUsup[idx][14])/2.
        - (0. + u2*u3)*theta_u_local/3.
        + (u2*velocity_array[4] + u3*velocity_array[3])/2.
        + u3*u0/(2.*tau)*u2);

    // make sigma^33 using traceless condition
    velocity_array[15] = (
        (2.*(u1*u2*velocity_array[11]
             + u1*u3*velocity_array[12]
             + u2*u3*velocity_array[14])
         - (u0*u0 - u1*u1)*velocity_array[10]
         - (u0*u0 - u2*u2)*velocity_array[13])
        /(u0*u0 - u3*u3));
    // make sigma^01 using transversality
    velocity_array[7] = ((velocity_array[10]*u1 + velocity_array[11]*u2
                          + velocity_array[12]*u3)/u0);
    velocity_array[8] = ((velocity_array[11]*u1 + velocity_array[13]*u2
                          + velocity_array[14]*u3)/u0);
    velocity_array[9] = ((velocity_array[12]*u1 + velocity_array[14]*u2
                          + velocity_array[15]*u3)/u0);
    velocity_array[6] = ((velocity_array[7]*u1 + velocity_array[8]*u2
                          + velocity_array[9]*u3)/u0);
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
