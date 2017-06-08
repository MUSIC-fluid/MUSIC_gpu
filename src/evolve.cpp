// Copyright 2012 Bjoern Schenke, Sangyong Jeon, and Charles Gale
#include <omp.h>
#include "./evolve.h"
#include "./util.h"
#include "./data.h"
#include "./grid.h"
#include "./eos.h"
#include "./advance.h"
#include "./cornelius.h"
#include "./field.h"

using namespace std;

Evolve::Evolve(EOS *eosIn, InitData *DATA_in) {
    eos = eosIn;
    grid = new Grid;
    grid_info = new Grid_info(DATA_in, eosIn);
    util = new Util;
    u_derivative = new U_derivative(eosIn, DATA_in);
    advance = new Advance(eosIn, DATA_in);
   
    DATA_ptr = DATA_in;
    rk_order = DATA_in->rk_order;
    grid_nx = DATA_in->nx;
    grid_ny = DATA_in->ny;
    grid_neta = DATA_in->neta;
  
    //if (DATA_ptr->freezeOutMethod == 4) {
    //    initialize_freezeout_surface_info();
    //}
}

// destructor
Evolve::~Evolve() {
    delete grid;
    delete grid_info;
    delete util;
    delete advance;
    delete u_derivative;
}

void Evolve::clean_up_hydro_fields(Field *hydro_fields) {
    int n_cell = DATA_ptr->neta*(DATA_ptr->nx + 1)*(DATA_ptr->ny + 1);
    delete[] hydro_fields->e_rk0;
    delete[] hydro_fields->e_rk1;
    delete[] hydro_fields->e_prev;
    delete[] hydro_fields->rhob_rk0;
    delete[] hydro_fields->rhob_rk1;
    delete[] hydro_fields->rhob_prev;
    for (int i = 0; i < n_cell; i++) {
        delete[] hydro_fields->u_rk0[i];
        delete[] hydro_fields->u_rk1[i];
        delete[] hydro_fields->u_prev[i];
        delete[] hydro_fields->dUsup[i];
        delete[] hydro_fields->Wmunu_rk0[i];
        delete[] hydro_fields->Wmunu_rk1[i];
        delete[] hydro_fields->Wmunu_prev[i];
    }
    delete[] hydro_fields->u_rk0;
    delete[] hydro_fields->u_rk1;
    delete[] hydro_fields->u_prev;
    delete[] hydro_fields->dUsup;
    delete[] hydro_fields->Wmunu_rk0;
    delete[] hydro_fields->Wmunu_rk1;
    delete[] hydro_fields->Wmunu_prev;
    delete[] hydro_fields->pi_b_rk0;
    delete[] hydro_fields->pi_b_rk1;
    delete[] hydro_fields->pi_b_prev;
}

void Evolve::initial_field_with_ideal_Gubser(double tau, Field *hydro_fields) {
    double x_min = -GRID_SIZE_X/2.*DELTA_X;
    double y_min = -GRID_SIZE_Y/2.*DELTA_Y;
    double e_local, utau_local, ux_local, uy_local;
    for (int ieta = 0; ieta < GRID_SIZE_ETA; ieta++) {
        for (int ix = 0; ix <= GRID_SIZE_X; ix++) {
            double x_local = x_min + ix*DELTA_X;
            for (int iy = 0; iy <= GRID_SIZE_Y; iy++) {
                double y_local = y_min + iy*DELTA_Y;
                int idx = (iy + ix*(GRID_SIZE_Y+1)
                            + ieta*(GRID_SIZE_Y+1)*(GRID_SIZE_X+1));
                e_local = energy_gubser(tau, x_local, y_local);
                flow_gubser(tau, x_local, y_local, &utau_local, &ux_local,
                            &uy_local);
                hydro_fields->e_rk0[idx] = e_local;
                hydro_fields->e_rk1[idx] = e_local;
                hydro_fields->e_prev[idx] = e_local;
                hydro_fields->rhob_rk0[idx] = 0.0;
                hydro_fields->rhob_rk1[idx] = 0.0;
                hydro_fields->rhob_prev[idx] = 0.0;
                hydro_fields->u_rk0[idx][0] = utau_local;
                hydro_fields->u_rk0[idx][1] = ux_local;
                hydro_fields->u_rk0[idx][2] = uy_local;
                hydro_fields->u_rk0[idx][3] = 0.0;
                hydro_fields->u_rk1[idx][0] = utau_local;
                hydro_fields->u_rk1[idx][1] = ux_local;
                hydro_fields->u_rk1[idx][2] = uy_local;
                hydro_fields->u_rk1[idx][3] = 0.0;
                hydro_fields->u_prev[idx][0] = utau_local;
                hydro_fields->u_prev[idx][1] = ux_local;
                hydro_fields->u_prev[idx][2] = uy_local;
                hydro_fields->u_prev[idx][3] = 0.0;
                for (int ii = 0; ii < 20; ii++) {
                    hydro_fields->dUsup[idx][ii] = 0.0;
                }
                for (int ii = 0; ii < 14; ii++) {
                    hydro_fields->Wmunu_rk0[idx][ii] = 0.0;
                    hydro_fields->Wmunu_rk1[idx][ii] = 0.0;
                    hydro_fields->Wmunu_prev[idx][ii] = 0.0;
                }
                hydro_fields->pi_b_rk0[idx] = 0.0;
                hydro_fields->pi_b_rk1[idx] = 0.0;
                hydro_fields->pi_b_prev[idx] = 0.0;
            }
        }
    }
}

void Evolve::check_field_with_ideal_Gubser(double tau, Field *hydro_fields) {
    double x_min = -GRID_SIZE_X/2.*DELTA_X;
    double y_min = -GRID_SIZE_Y/2.*DELTA_Y;
    double e_local, utau_local, ux_local, uy_local;
    double e_diff = 0.0;
    double e_total = 0.0;
    double ux_diff = 0.0;
    double ux_total = 0.0;
    double uy_diff = 0.0;
    double uy_total = 0.0;
    for (int ieta = 0; ieta < GRID_SIZE_ETA; ieta++) {
        for (int ix = 0; ix <= GRID_SIZE_X; ix++) {
            double x_local = x_min + ix*DELTA_X;
            for (int iy = 0; iy <= GRID_SIZE_Y; iy++) {
                double y_local = y_min + iy*DELTA_Y;
                int idx = (iy + ix*(GRID_SIZE_Y+1)
                            + ieta*(GRID_SIZE_Y+1)*(GRID_SIZE_X+1));
                e_local = energy_gubser(tau, x_local, y_local);
                flow_gubser(tau, x_local, y_local, &utau_local, &ux_local,
                            &uy_local);
                e_diff += fabs(hydro_fields->e_rk0[idx] - e_local);
                e_total += fabs(e_local);
                ux_diff += fabs(hydro_fields->u_rk0[idx][1] - ux_local);
                ux_total += fabs(ux_local);
                uy_diff += fabs(hydro_fields->u_rk0[idx][2] - uy_local);
                uy_total += fabs(uy_local);
            }
        }
    }
    cout << "e_diff: " << e_diff/e_total << ", ux_diff: " << ux_diff/ux_total
         << ", uy_diff: " << uy_diff/uy_total << endl;
}

// master control function for hydrodynamic evolution
int Evolve::EvolveIt(InitData *DATA, Field *hydro_fields) {
    initial_field_with_ideal_Gubser(DATA->tau0, hydro_fields);
    check_field_with_ideal_Gubser(DATA->tau0, hydro_fields);
    // first pass some control parameters
    //facTau = DATA->facTau;
    //int Nskip_timestep = DATA->output_evolution_every_N_timesteps;
    //int outputEvo_flag = DATA->outputEvolutionData;
    //int output_movie_flag = DATA->output_movie_flag;
    //int freezeout_flag = DATA->doFreezeOut;
    //int freezeout_lowtemp_flag = DATA->doFreezeOut_lowtemp;
    //int freezeout_method = DATA->freezeOutMethod;
    //int boost_invariant_flag = DATA->boost_invariant;

    // Output information about the hydro parameters 
    // in the format of a C header file
    //if (DATA->output_hydro_params_header || outputEvo_flag == 1)
    //    grid_info->Output_hydro_information_header(DATA);
{

    // main loop starts ...
    DATA->delta_tau= DELTA_TAU;
    DATA->delta_x = DELTA_X;
    DATA->delta_y= DELTA_Y;
    DATA->delta_eta= DELTA_ETA;

    int itmax = static_cast<int>(DATA->tau_size/DATA->delta_tau);
    double tau0 = DATA->tau0;
    double dt = DATA->delta_tau;
    DATA->delta_tau = DELTA_TAU;
    DATA->delta_x = DELTA_X;
    DATA->delta_y = DELTA_Y;
    DATA->delta_eta = DELTA_ETA;
    double tau;
    //int it_start = 0;
    cout << "Pre data copy" << endl;
    #pragma acc data copyin (hydro_fields[0:1],\
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
                         hydro_fields->pi_b_prev[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA])
    {
    cout << "Post data copy" << endl;
    
    for (int it = 0; it <= itmax; it++) {
        tau = tau0 + dt*it;
        // store initial conditions
        //if (it == it_start) {
        //    store_previous_step_for_freezeout(arena);
        //}

        //convert_grid_to_field(arena, hydro_fields);
        
        if (DATA->Initial_profile == 0) {
            if (fabs(tau - 1.0) < 1e-8) {
                grid_info->Gubser_flow_check_file(hydro_fields, tau);
            }
            if (fabs(tau - 1.2) < 1e-8) {
                grid_info->Gubser_flow_check_file(hydro_fields, tau);
            }
            if (fabs(tau - 1.5) < 1e-8) {
                grid_info->Gubser_flow_check_file(hydro_fields, tau);
            }
            if (fabs(tau - 2.0) < 1e-8) {
                grid_info->Gubser_flow_check_file(hydro_fields, tau);
            }
            if (fabs(tau - 3.0) < 1e-8) {
                grid_info->Gubser_flow_check_file(hydro_fields, tau);
            }
        }

        //if (it % Nskip_timestep == 0) {
        //    if (outputEvo_flag == 1) {
        //        grid_info->OutputEvolutionDataXYEta(arena, DATA, tau);
        //    } else if (outputEvo_flag == 2) {
        //        grid_info->OutputEvolutionDataXYEta_chun(arena, DATA, tau);
        //    }
        //    if (output_movie_flag == 1) {
        //        grid_info->output_evolution_for_movie(arena, tau);
        //    }
        //}
        // grid_info->output_average_phase_diagram_trajectory(tau, -0.5, 0.5,
        //                                                    arena);

        // check energy conservation
        //if (boost_invariant_flag == 0)
        //    grid_info->check_conservation_law(hydro_fields, DATA, tau);
        //grid_info->get_maximum_energy_density(hydro_fields);

        /* execute rk steps */
        // all the evolution are at here !!!
        AdvanceRK(tau, DATA, hydro_fields);
        #pragma acc update host(hydro_fields->e_rk0[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA])
        #pragma acc update host(hydro_fields->u_rk0[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA][0:4])
        //for (int x = 0; x < (GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA; x += 100){
        //    cout << hydro_fields->e_rk0[x] << endl;
        //}
        check_field_with_ideal_Gubser(tau, hydro_fields);
        //copy_fields_to_grid(hydro_fields, arena);
        
        //determine freeze-out surface
        //int frozen = 0;
        //if (freezeout_flag == 1) {
        //    if (freezeout_lowtemp_flag == 1) {
        //        if (it == it_start) {
        //            frozen = FreezeOut_equal_tau_Surface(tau, DATA, arena);
        //        }
        //    }
        //    // avoid freeze-out at the first time step
        //    if ((it - it_start)%facTau == 0 && it > it_start) {
        //       if (freezeout_method == 4) {
        //           if (boost_invariant_flag == 0) {
        //               frozen = FindFreezeOutSurface_Cornelius(tau, DATA,
        //                                                       arena);
        //           } else {
        //               frozen = FindFreezeOutSurface_boostinvariant_Cornelius(
        //                                                    tau, DATA, arena);
        //           }
        //       }
        //       store_previous_step_for_freezeout(arena);
        //    }
        //}/* do freeze-out determination */
    
        fprintf(stdout, "Done time step %d/%d. tau = %6.3f fm/c \n", 
                it, itmax, tau);
        //if (frozen) break;
    }/* it */ 
    }
}

    // clean up
    clean_up_hydro_fields(hydro_fields);

    return(1); /* successful */
}/* Evolve */

void Evolve::store_previous_step_for_freezeout(Grid ***arena) {
    int nx = grid_nx;
    int ny = grid_ny;
    int neta = grid_neta;
    for (int ieta=0; ieta<neta; ieta++) {
        for (int ix=0; ix<=nx; ix++) {
            for (int iy=0; iy<=ny; iy++) {
                arena[ieta][ix][iy].epsilon_prev = arena[ieta][ix][iy].epsilon;
                arena[ieta][ix][iy].u_prev[0] = arena[ieta][ix][iy].u[0][0];
                arena[ieta][ix][iy].u_prev[1] = arena[ieta][ix][iy].u[0][1];
                arena[ieta][ix][iy].u_prev[2] = arena[ieta][ix][iy].u[0][2];
                arena[ieta][ix][iy].u_prev[3] = arena[ieta][ix][iy].u[0][3];
                arena[ieta][ix][iy].rhob_prev = arena[ieta][ix][iy].rhob;

                arena[ieta][ix][iy].pi_b_prev = arena[ieta][ix][iy].pi_b[0];
                
                for (int ii = 0; ii < 10; ii++) {
                    arena[ieta][ix][iy].W_prev[ii] =
                                            arena[ieta][ix][iy].Wmunu[0][ii];
                }
                if (DATA_ptr->turn_on_diff == 1) {
                    for (int ii = 10; ii < 14; ii++) {
                        arena[ieta][ix][iy].W_prev[ii] =
                                            arena[ieta][ix][iy].Wmunu[0][ii];
                    }
                } else {
                    for (int ii = 10; ii < 14; ii++) {
                        arena[ieta][ix][iy].W_prev[ii] = 0.0;
                    }
                }
            }
        }
    }
}

// update grid information after the tau RK evolution 
int Evolve::Update_prev_Arena(Grid ***arena) {
    int neta = grid_neta;
    int ieta;
    #pragma omp parallel private(ieta)
    {
        #pragma omp for
        for (ieta = 0; ieta < neta; ieta++) {
            Update_prev_Arena_XY(ieta, arena);
        } /* ieta */
    }
    return 1;
}

void Evolve::Update_prev_Arena_XY(int ieta, Grid ***arena) {
    int nx = grid_nx;
    int ny = grid_ny;
    for (int ix = 0; ix <= nx; ix++) {
        for (int iy = 0; iy <= ny; iy++) {
            arena[ieta][ix][iy].prev_epsilon = arena[ieta][ix][iy].epsilon;
            arena[ieta][ix][iy].prev_rhob = arena[ieta][ix][iy].rhob;
     
            // previous pi_b is stored in prevPimunu
            arena[ieta][ix][iy].prev_pi_b[0] = arena[ieta][ix][iy].pi_b[0];
            for (int ii = 0; ii < 14; ii++) {
                /* this was the previous value */
                arena[ieta][ix][iy].prevWmunu[0][ii] = (
                                arena[ieta][ix][iy].Wmunu[0][ii]); 
            }
            for (int mu = 0; mu < 4; mu++) {
                /* this was the previous value */
                arena[ieta][ix][iy].prev_u[0][mu] = (
                                        arena[ieta][ix][iy].u[0][mu]); 
            }
        }
    }
}

void Evolve::update_prev_field(Field *hydro_fields) {
    int n_cell = DATA_ptr->neta*(DATA_ptr->nx + 1)*(DATA_ptr->ny + 1);
    for (int i = 0; i < n_cell; i++) {
        hydro_fields->e_prev[i] = hydro_fields->e_rk0[i];
        hydro_fields->rhob_prev[i] = hydro_fields->rhob_rk0[i];
        for (int ii = 0; ii < 4; ii++) {
            hydro_fields->u_prev[i][ii] = hydro_fields->u_rk0[i][ii];
        }
        for (int ii = 0; ii < 14; ii++) {
            hydro_fields->Wmunu_prev[i][ii] = hydro_fields->Wmunu_rk0[i][ii];
        }
        hydro_fields->pi_b_prev[i] = hydro_fields->pi_b_rk0[i];
    }

}

void Evolve::copy_dUsup_from_grid_to_field(Grid ***arena, Field *hydro_fields) {
    int nx = grid_nx + 1;
    int ny = grid_ny + 1;
    int neta = grid_neta;
    for (int ieta = 0; ieta < neta; ieta++) {
        for (int ix = 0; ix < nx; ix++) {
            for (int iy = 0; iy < ny; iy++) {
                int idx = iy + ix*ny + ieta*ny*nx;
                for (int ii = 0; ii < 5; ii++) {
                    for (int jj = 0; jj < 4; jj++) {
                        int iidx = jj + 4*ii;
                        hydro_fields->dUsup[idx][iidx] =
                            arena[ieta][ix][iy].dUsup[0][ii][jj];
                    }
                }
            }
        }
    }
}

void Evolve::convert_grid_to_field(Grid ***arena, Field *hydro_fields) {
    int nx = grid_nx + 1;
    int ny = grid_ny + 1;
    int neta = grid_neta;
    for (int ieta = 0; ieta < neta; ieta++) {
        for (int ix = 0; ix < nx; ix++) {
            for (int iy = 0; iy < ny; iy++) {
                int idx = iy + ix*ny + ieta*ny*nx;
                hydro_fields->e_rk0[idx] = arena[ieta][ix][iy].epsilon;
                hydro_fields->e_rk1[idx] = arena[ieta][ix][iy].epsilon_t;
                hydro_fields->e_prev[idx] = arena[ieta][ix][iy].prev_epsilon;
                hydro_fields->rhob_rk0[idx] = arena[ieta][ix][iy].rhob;
                hydro_fields->rhob_rk1[idx] = arena[ieta][ix][iy].rhob_t;
                hydro_fields->rhob_prev[idx] = arena[ieta][ix][iy].prev_rhob;
                for (int ii = 0; ii < 4; ii++) {
                    hydro_fields->u_rk0[idx][ii] =
                                            arena[ieta][ix][iy].u[0][ii];
                    hydro_fields->u_rk1[idx][ii] =
                                            arena[ieta][ix][iy].u[1][ii];
                    hydro_fields->u_prev[idx][ii] =
                                            arena[ieta][ix][iy].prev_u[0][ii];
                }
                for (int ii = 0; ii < 5; ii++) {
                    for (int jj = 0; jj < 4; jj++) {
                        int iidx = jj + 4*ii;
                        hydro_fields->dUsup[idx][iidx] =
                            arena[ieta][ix][iy].dUsup[0][ii][jj];
                    }
                }
                for (int ii = 0; ii < 14; ii++) {
                    hydro_fields->Wmunu_rk0[idx][ii] = 
                                            arena[ieta][ix][iy].Wmunu[0][ii];
                    hydro_fields->Wmunu_rk1[idx][ii] = 
                                            arena[ieta][ix][iy].Wmunu[1][ii];
                    hydro_fields->Wmunu_prev[idx][ii] = 
                                        arena[ieta][ix][iy].prevWmunu[0][ii];
                }
                hydro_fields->pi_b_rk0[idx] = arena[ieta][ix][iy].pi_b[0];
                hydro_fields->pi_b_rk1[idx] = arena[ieta][ix][iy].pi_b[1];
                hydro_fields->pi_b_prev[idx] =
                                            arena[ieta][ix][iy].prev_pi_b[0];
            }
        }
    }
}


//! This is a control function for Runge-Kutta evolution in tau
int Evolve::AdvanceRK(double tau, InitData *DATA, Field *hydro_fields) {
    int flag = 0;
    // loop over Runge-Kutta steps
    for (int rk_flag = 0; rk_flag < rk_order; rk_flag++) {
        //flag = u_derivative->MakedU(tau, hydro_fields, rk_flag);
        flag = advance->AdvanceIt(tau, hydro_fields, rk_flag);
        if (rk_flag == 0) {
            update_prev_field(hydro_fields);
        }
    }  /* loop over rk_flag */
    return(flag);
}

void Evolve::copy_fields_to_grid(Field *hydro_fields, Grid ***arena) {
    int nx = grid_nx + 1;
    int ny = grid_ny + 1;
    int neta = grid_neta;
    for (int ieta = 0; ieta < neta; ieta++) {
        for (int ix = 0; ix < nx; ix++) {
            for (int iy = 0; iy < ny; iy++) {
                int idx = iy + ix*ny + ieta*ny*nx;
                arena[ieta][ix][iy].epsilon = hydro_fields->e_rk0[idx];
                arena[ieta][ix][iy].epsilon_t = hydro_fields->e_rk1[idx];
                arena[ieta][ix][iy].prev_epsilon = hydro_fields->e_prev[idx];
                arena[ieta][ix][iy].rhob = hydro_fields->rhob_rk0[idx];
                arena[ieta][ix][iy].rhob_t = hydro_fields->rhob_rk1[idx];
                arena[ieta][ix][iy].prev_rhob = hydro_fields->rhob_prev[idx];
                for (int ii = 0; ii < 4; ii++) {
                    arena[ieta][ix][iy].u[0][ii] = hydro_fields->u_rk0[idx][ii];
                    arena[ieta][ix][iy].u[1][ii] = hydro_fields->u_rk1[idx][ii];
                    arena[ieta][ix][iy].prev_u[0][ii] = hydro_fields->u_prev[idx][ii];
                }
                for (int ii = 0; ii < 14; ii++) {
                    arena[ieta][ix][iy].Wmunu[0][ii] = hydro_fields->Wmunu_rk0[idx][ii];
                    arena[ieta][ix][iy].Wmunu[1][ii] = hydro_fields->Wmunu_rk1[idx][ii];
                    arena[ieta][ix][iy].prevWmunu[0][ii] = hydro_fields->Wmunu_prev[idx][ii];
                }
                arena[ieta][ix][iy].pi_b[0] = hydro_fields->pi_b_rk0[idx];
                arena[ieta][ix][iy].pi_b[1] = hydro_fields->pi_b_rk1[idx];
                arena[ieta][ix][iy].prev_pi_b[0] = hydro_fields->pi_b_prev[idx];
            }
        }
    }
}
      
// Cornelius freeze out  (C. Shen, 11/2014)
int Evolve::FindFreezeOutSurface_Cornelius(double tau, InitData *DATA,
                                           Grid ***arena) {
    // output hyper-surfaces for Cooper-Frye
    int *all_frozen = new int [n_freeze_surf];
    for(int i = 0; i < n_freeze_surf; i++)
        all_frozen[i] = 0;
      
    int neta = grid_neta;
    int fac_eta = 1;
    for (int i_freezesurf = 0; i_freezesurf < n_freeze_surf; i_freezesurf++) {
        double epsFO = epsFO_list[i_freezesurf]/hbarc;
        
        int intersections = 0;
        int ieta;
        #pragma omp parallel private(ieta)
        {
            #pragma omp for
            for (ieta = 0; ieta < (neta-fac_eta); ieta += fac_eta) {
                int thread_id = omp_get_thread_num();
                intersections += FindFreezeOutSurface_Cornelius_XY(
                                    tau, DATA, ieta, arena, thread_id, epsFO);
            }
            #pragma omp barrier
        }

        if (intersections == 0)
            all_frozen[i_freezesurf] = 1;
    }

    int all_frozen_flag = 1;
    for(int ii = 0; ii < n_freeze_surf; ii++)
        all_frozen_flag *= all_frozen[ii];

    if (all_frozen_flag == 1) {
        cout << "All cells frozen out. Exiting." << endl;
    }
    delete[] all_frozen;
    return(all_frozen_flag);
}

int Evolve::FindFreezeOutSurface_Cornelius_XY(double tau, InitData *DATA,
                                              int ieta, Grid ***arena,
                                              int thread_id, double epsFO) {
    int nx = grid_nx;
    int ny = grid_ny;

    stringstream strs_name;
    strs_name << "surface_eps_" << setprecision(4) << epsFO*hbarc
              << "_" << thread_id << ".dat";
    ofstream s_file;
    s_file.open(strs_name.str().c_str(), ios::out | ios::app);
   
    double FULLSU[4];  // d^3 \sigma_\mu

    double tau_center, x_center, y_center, eta_center;
    double Wtautau_center, Wtaux_center, Wtauy_center, Wtaueta_center;
    double Wxx_center, Wxy_center, Wxeta_center;
    double Wyy_center, Wyeta_center, Wetaeta_center;
    double rhob_center;
    double qtau_center, qx_center, qy_center, qeta_center;
    double utau_center, ux_center, uy_center, ueta_center;
    double TFO, muB, pressure, eps_plus_p_over_T_FO;
    double pi_b_center; // bulk viscous pressure

    int dim = 4;
    double lattice_spacing[4];

    double *lattice_spacing_ptr = new double [dim];
    // for 4-d linear interpolation
    double** x_fraction = new double* [2];
    for(int i = 0; i < 2; i++)
        x_fraction[i] = new double [dim];

    int intersect;
    int intersections = 0;

    facTau = DATA->facTau;   // step to skip in tau direction
    int fac_x = DATA->fac_x;
    int fac_y = DATA->fac_y;
    int fac_eta = 1;
    
    double DTAU=facTau*DATA->delta_tau;
    double DX=fac_x*DATA->delta_x;
    double DY=fac_y*DATA->delta_y;
    double DETA=fac_eta*DATA->delta_eta;

    lattice_spacing[0] = DTAU;
    lattice_spacing[1] = DX;
    lattice_spacing[2] = DY;
    lattice_spacing[3] = DETA;
    
    lattice_spacing_ptr[0] = DTAU;
    lattice_spacing_ptr[1] = DX;
    lattice_spacing_ptr[2] = DY;
    lattice_spacing_ptr[3] = DETA;

    // initialize Cornelius
    Cornelius* cornelius_ptr = new Cornelius();
    cornelius_ptr->init(dim, epsFO, lattice_spacing);

    // initialize the hyper-cube for Cornelius
    double ****cube = new double*** [2];
    for (int i = 0; i < 2; i++) {
        cube[i] = new double** [2];
        for (int j = 0; j < 2; j++) {
            cube[i][j] = new double* [2];
            for (int k = 0; k < 2; k++) {
                cube[i][j][k] = new double[2];
                for (int l = 0; l < 2; l++)
                    cube[i][j][k][l] =0.0;
            }
        }
    }

    double eta = (DATA->delta_eta)*ieta - (DATA->eta_size)/2.0;
    for (int ix = 0; ix <= nx - fac_x; ix += fac_x) {
        double x = ix*(DATA->delta_x) - (DATA->x_size/2.0); 
        for (int iy = 0; iy <= ny - fac_y; iy += fac_y) {
            double y = iy*(DATA->delta_y) - (DATA->y_size/2.0);

            // make sure the epsilon value is never exactly 
            // the same as epsFO...
            if (arena[ieta+fac_eta][ix+fac_x][iy+fac_y].epsilon
                == epsFO)
                arena[ieta+fac_eta][ix+fac_x][iy+fac_y].epsilon += 0.000001;
            if (arena[ieta][ix][iy].epsilon_prev == epsFO)
                arena[ieta][ix][iy].epsilon_prev += 0.000001;
            if (arena[ieta][ix+fac_x][iy].epsilon == epsFO)
                arena[ieta][ix+fac_x][iy].epsilon += 0.000001;
            if (arena[ieta+fac_eta][ix][iy+fac_y].epsilon_prev
                == epsFO)
                arena[ieta+fac_eta][ix][iy+fac_y].epsilon_prev += 0.000001;
            if (arena[ieta][ix][iy+fac_y].epsilon == epsFO)
                arena[ieta][ix][iy+fac_y].epsilon += 0.000001;
            if (arena[ieta+fac_eta][ix+fac_x][iy].epsilon_prev
                == epsFO)
                arena[ieta+fac_eta][ix+fac_x][iy].epsilon_prev += 0.000001;
            if (arena[ieta+fac_eta][ix][iy].epsilon == epsFO)
                arena[ieta+fac_eta][ix][iy].epsilon += 0.000001;
            if (arena[ieta][ix+fac_x][iy+fac_y].epsilon_prev == epsFO)
                arena[ieta][ix+fac_x][iy+fac_y].epsilon_prev += 0.000001;
            if (arena[ieta][ix+fac_x][iy+fac_y].epsilon == epsFO)
                arena[ieta][ix+fac_x][iy+fac_y].epsilon += 0.000001;
            if (arena[ieta+fac_eta][ix][iy].epsilon_prev == epsFO)
                arena[ieta+fac_eta][ix][iy].epsilon_prev += 0.000001;
            if (arena[ieta+fac_eta][ix+fac_x][iy].epsilon == epsFO)
                arena[ieta+fac_eta][ix+fac_x][iy].epsilon += 0.000001;
            if (arena[ieta][ix][iy+fac_y].epsilon_prev == epsFO)
                arena[ieta][ix][iy+fac_y].epsilon_prev += 0.000001;
            if (arena[ieta+fac_eta][ix][iy+fac_y].epsilon == epsFO)
                arena[ieta+fac_eta][ix][iy+fac_y].epsilon += 0.000001;
            if (arena[ieta][ix+fac_x][iy].epsilon_prev == epsFO)
                arena[ieta][ix+fac_x][iy].epsilon_prev += 0.000001;
            if (arena[ieta][ix][iy].epsilon == epsFO)
                arena[ieta][ix][iy].epsilon += 0.000001;
            if (arena[ieta+fac_eta][ix+fac_x][iy+fac_y].epsilon_prev == epsFO)
                arena[ieta+fac_eta][ix+fac_x][iy+fac_y].epsilon_prev += 0.000001;
    
            // judge intersection (from Bjoern)
            intersect = 1;
            if ((arena[ieta+fac_eta][ix+fac_x][iy+fac_y].epsilon-epsFO)
                *(arena[ieta][ix][iy].epsilon_prev-epsFO)>0.)
                if((arena[ieta][ix+fac_x][iy].epsilon-epsFO)
                    *(arena[ieta+fac_eta][ix][iy+fac_y].epsilon_prev-epsFO)>0.)
                    if((arena[ieta][ix][iy+fac_y].epsilon-epsFO)
                        *(arena[ieta+fac_eta][ix+fac_x][iy].epsilon_prev-epsFO)>0.)
                        if((arena[ieta+fac_eta][ix][iy].epsilon-epsFO)
                            *(arena[ieta][ix+fac_x][iy+fac_y].epsilon_prev-epsFO)>0.)
                            if((arena[ieta][ix+fac_x][iy+fac_y].epsilon-epsFO)
                                *(arena[ieta+fac_eta][ix][iy].epsilon_prev-epsFO)>0.)
                                if((arena[ieta+fac_eta][ix+fac_x][iy].epsilon-epsFO)
                                    *(arena[ieta][ix][iy+fac_y].epsilon_prev-epsFO)>0.)
                                    if((arena[ieta+fac_eta][ix][iy+fac_y].epsilon-epsFO)
                                        *(arena[ieta][ix+fac_x][iy].epsilon_prev-epsFO)>0.)
                                        if((arena[ieta][ix][iy].epsilon-epsFO)
                                            *(arena[ieta+fac_eta][ix+fac_x][iy+fac_y].epsilon_prev-epsFO)>0.)
                                                intersect=0;

            if (intersect==0) {
                continue;
            }

            // if intersect, prepare for the hyper-cube
            intersections++;
            cube[0][0][0][0] = arena[ieta][ix][iy].epsilon_prev;
            cube[0][0][1][0] = arena[ieta][ix][iy+fac_y].epsilon_prev;
            cube[0][1][0][0] = arena[ieta][ix+fac_x][iy].epsilon_prev;
            cube[0][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].epsilon_prev;
            cube[1][0][0][0] = arena[ieta][ix][iy].epsilon;
            cube[1][0][1][0] = arena[ieta][ix][iy+fac_y].epsilon;
            cube[1][1][0][0] = arena[ieta][ix+fac_x][iy].epsilon;
            cube[1][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].epsilon;
            cube[0][0][0][1] = arena[ieta+fac_eta][ix][iy].epsilon_prev;
            cube[0][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].epsilon_prev;
            cube[0][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].epsilon_prev;
            cube[0][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].epsilon_prev;
            cube[1][0][0][1] = arena[ieta+fac_eta][ix][iy].epsilon;
            cube[1][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].epsilon;
            cube[1][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].epsilon;
            cube[1][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].epsilon;
    
            // Now, the magic will happen in the Cornelius ...
            cornelius_ptr->find_surface_4d(cube);

            // get positions of the freeze-out surface
            // and interpolating results
            for (int isurf = 0; isurf < cornelius_ptr->get_Nelements();
                 isurf++) {
                // surface normal vector d^3 \sigma_\mu
                for (int ii = 0; ii < 4; ii++)
                    FULLSU[ii] = 
                        cornelius_ptr->get_normal_elem(isurf, ii);

                // check the size of the surface normal vector
                if (fabs(FULLSU[0]) > (DX*DY*DETA+0.01)) {
                    cerr << "problem: volume in tau direction " 
                         << fabs(FULLSU[0]) << "  > DX*DY*DETA = "  
                         << DX*DY*DETA << endl;
                }
                if (fabs(FULLSU[1]) > (DTAU*DY*DETA+0.01)) {
                   cerr << "problem: volume in x direction " 
                        << fabs(FULLSU[1]) << "  > DTAU*DY*DETA = "  
                        << DTAU*DY*DETA << endl;
                }
                if (fabs(FULLSU[2]) > (DX*DTAU*DETA+0.01)) {
                   cerr << "problem: volume in y direction " 
                        << fabs(FULLSU[2]) << "  > DX*DTAU*DETA = "  
                        << DX*DTAU*DETA << endl;
                }
                if (fabs(FULLSU[3]) > (DX*DY*DTAU+0.01)) {
                   cerr << "problem: volume in eta direction " 
                        << fabs(FULLSU[3]) << "  > DX*DY*DTAU = "  
                        << DX*DY*DTAU << endl;
                }

                // position of the freeze-out fluid cell
                for (int ii = 0; ii < 4; ii++) {
                    x_fraction[1][ii] = 
                        cornelius_ptr->get_centroid_elem(isurf, ii);
                    x_fraction[0][ii] = 
                        lattice_spacing_ptr[ii] - x_fraction[1][ii];
                }
                tau_center = tau - DTAU + x_fraction[1][0];
                x_center = x + x_fraction[1][1];
                y_center = y + x_fraction[1][2];
                eta_center = eta + x_fraction[1][3];

                // perform 4-d linear interpolation for all fluid
                // quantities

                // flow velocity u^x
                cube[0][0][0][0] = arena[ieta][ix][iy].u_prev[1];
                cube[0][0][1][0] = arena[ieta][ix][iy+fac_y].u_prev[1];
                cube[0][1][0][0] = arena[ieta][ix+fac_x][iy].u_prev[1];
                cube[0][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].u_prev[1];
                cube[1][0][0][0] = arena[ieta][ix][iy].u[0][1];
                cube[1][0][1][0] = arena[ieta][ix][iy+fac_y].u[0][1];
                cube[1][1][0][0] = arena[ieta][ix+fac_x][iy].u[0][1];
                cube[1][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].u[0][1];
                cube[0][0][0][1] = arena[ieta+fac_eta][ix][iy].u_prev[1];
                cube[0][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].u_prev[1];
                cube[0][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].u_prev[1];
                cube[0][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].u_prev[1];
                cube[1][0][0][1] = arena[ieta+fac_eta][ix][iy].u[0][1];
                cube[1][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].u[0][1];
                cube[1][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].u[0][1];
                cube[1][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].u[0][1];
                ux_center = 
                    util->four_dimension_linear_interpolation(
                                lattice_spacing_ptr, x_fraction, cube);

                // flow velocity u^y
                cube[0][0][0][0] = arena[ieta][ix][iy].u_prev[2];
                cube[0][0][1][0] = arena[ieta][ix][iy+fac_y].u_prev[2];
                cube[0][1][0][0] = arena[ieta][ix+fac_x][iy].u_prev[2];
                cube[0][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].u_prev[2];
                cube[1][0][0][0] = arena[ieta][ix][iy].u[0][2];
                cube[1][0][1][0] = arena[ieta][ix][iy+fac_y].u[0][2];
                cube[1][1][0][0] = arena[ieta][ix+fac_x][iy].u[0][2];
                cube[1][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].u[0][2];
                cube[0][0][0][1] = arena[ieta+fac_eta][ix][iy].u_prev[2];
                cube[0][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].u_prev[2];
                cube[0][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].u_prev[2];
                cube[0][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].u_prev[2];
                cube[1][0][0][1] = arena[ieta+fac_eta][ix][iy].u[0][2];
                cube[1][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].u[0][2];
                cube[1][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].u[0][2];
                cube[1][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].u[0][2];
                uy_center = 
                    util->four_dimension_linear_interpolation(
                                lattice_spacing_ptr, x_fraction, cube);

                // flow velocity u^eta
                cube[0][0][0][0] = arena[ieta][ix][iy].u_prev[3];
                cube[0][0][1][0] = arena[ieta][ix][iy+fac_y].u_prev[3];
                cube[0][1][0][0] = arena[ieta][ix+fac_x][iy].u_prev[3];
                cube[0][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].u_prev[3];
                cube[1][0][0][0] = arena[ieta][ix][iy].u[0][3];
                cube[1][0][1][0] = arena[ieta][ix][iy+fac_y].u[0][3];
                cube[1][1][0][0] = arena[ieta][ix+fac_x][iy].u[0][3];
                cube[1][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].u[0][3];
                cube[0][0][0][1] = arena[ieta+fac_eta][ix][iy].u_prev[3];
                cube[0][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].u_prev[3];
                cube[0][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].u_prev[3];
                cube[0][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].u_prev[3];
                cube[1][0][0][1] = arena[ieta+fac_eta][ix][iy].u[0][3];
                cube[1][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].u[0][3];
                cube[1][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].u[0][3];
                cube[1][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].u[0][3];
                ueta_center = 
                    util->four_dimension_linear_interpolation(
                                lattice_spacing_ptr, x_fraction, cube);

                // reconstruct u^tau from u^i
                utau_center = sqrt(1. + ux_center*ux_center 
                                   + uy_center*uy_center 
                                   + ueta_center*ueta_center);

                // baryon density rho_b
                cube[0][0][0][0] = arena[ieta][ix][iy].rhob_prev;
                cube[0][0][1][0] = arena[ieta][ix][iy+fac_y].rhob_prev;
                cube[0][1][0][0] = arena[ieta][ix+fac_x][iy].rhob_prev;
                cube[0][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].rhob_prev;
                cube[1][0][0][0] = arena[ieta][ix][iy].rhob;
                cube[1][0][1][0] = arena[ieta][ix][iy+fac_y].rhob;
                cube[1][1][0][0] = arena[ieta][ix+fac_x][iy].rhob;
                cube[1][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].rhob;
                cube[0][0][0][1] = arena[ieta+fac_eta][ix][iy].rhob_prev;
                cube[0][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].rhob_prev;
                cube[0][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].rhob_prev;
                cube[0][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].rhob_prev;
                cube[1][0][0][1] = arena[ieta+fac_eta][ix][iy].rhob;
                cube[1][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].rhob;
                cube[1][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].rhob;
                cube[1][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].rhob;
                rhob_center = 
                    util->four_dimension_linear_interpolation(
                                lattice_spacing_ptr, x_fraction, cube);
          
                // baryon diffusion current q^tau
                cube[0][0][0][0] = arena[ieta][ix][iy].W_prev[10];
                cube[0][0][1][0] = arena[ieta][ix][iy+fac_y].W_prev[10];
                cube[0][1][0][0] = arena[ieta][ix+fac_x][iy].W_prev[10];
                cube[0][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].W_prev[10];
                cube[1][0][0][0] = arena[ieta][ix][iy].Wmunu[0][10];
                cube[1][0][1][0] = arena[ieta][ix][iy+fac_y].Wmunu[0][10];
                cube[1][1][0][0] = arena[ieta][ix+fac_x][iy].Wmunu[0][10];
                cube[1][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].Wmunu[0][10];
                cube[0][0][0][1] = arena[ieta+fac_eta][ix][iy].W_prev[10];
                cube[0][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].W_prev[10];
                cube[0][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].W_prev[10];
                cube[0][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].W_prev[10];
                cube[1][0][0][1] = arena[ieta+fac_eta][ix][iy].Wmunu[0][10];
                cube[1][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].Wmunu[0][10];
                cube[1][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].Wmunu[0][10];
                cube[1][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].Wmunu[0][10];
                qtau_center = 
                    util->four_dimension_linear_interpolation(
                                lattice_spacing_ptr, x_fraction, cube);
          
                // baryon diffusion current q^x
                cube[0][0][0][0] = arena[ieta][ix][iy].W_prev[11];
                cube[0][0][1][0] = arena[ieta][ix][iy+fac_y].W_prev[11];
                cube[0][1][0][0] = arena[ieta][ix+fac_x][iy].W_prev[11];
                cube[0][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].W_prev[11];
                cube[1][0][0][0] = arena[ieta][ix][iy].Wmunu[0][11];
                cube[1][0][1][0] = arena[ieta][ix][iy+fac_y].Wmunu[0][11];
                cube[1][1][0][0] = arena[ieta][ix+fac_x][iy].Wmunu[0][11];
                cube[1][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].Wmunu[0][11];
                cube[0][0][0][1] = arena[ieta+fac_eta][ix][iy].W_prev[11];
                cube[0][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].W_prev[11];
                cube[0][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].W_prev[11];
                cube[0][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].W_prev[11];
                cube[1][0][0][1] = arena[ieta+fac_eta][ix][iy].Wmunu[0][11];
                cube[1][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].Wmunu[0][11];
                cube[1][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].Wmunu[0][11];
                cube[1][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].Wmunu[0][11];
                qx_center = 
                    util->four_dimension_linear_interpolation(
                                lattice_spacing_ptr, x_fraction, cube);

                // baryon diffusion current q^y
                cube[0][0][0][0] = arena[ieta][ix][iy].W_prev[12];
                cube[0][0][1][0] = arena[ieta][ix][iy+fac_y].W_prev[12];
                cube[0][1][0][0] = arena[ieta][ix+fac_x][iy].W_prev[12];
                cube[0][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].W_prev[12];
                cube[1][0][0][0] = arena[ieta][ix][iy].Wmunu[0][12];
                cube[1][0][1][0] = arena[ieta][ix][iy+fac_y].Wmunu[0][12];
                cube[1][1][0][0] = arena[ieta][ix+fac_x][iy].Wmunu[0][12];
                cube[1][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].Wmunu[0][12];
                cube[0][0][0][1] = arena[ieta+fac_eta][ix][iy].W_prev[12];
                cube[0][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].W_prev[12];
                cube[0][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].W_prev[12];
                cube[0][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].W_prev[12];
                cube[1][0][0][1] = arena[ieta+fac_eta][ix][iy].Wmunu[0][12];
                cube[1][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].Wmunu[0][12];
                cube[1][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].Wmunu[0][12];
                cube[1][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].Wmunu[0][12];
                qy_center = 
                    util->four_dimension_linear_interpolation(
                            lattice_spacing_ptr, x_fraction, cube);
          
                // baryon diffusion current q^eta
                cube[0][0][0][0] = arena[ieta][ix][iy].W_prev[13];
                cube[0][0][1][0] = arena[ieta][ix][iy+fac_y].W_prev[13];
                cube[0][1][0][0] = arena[ieta][ix+fac_x][iy].W_prev[13];
                cube[0][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].W_prev[13];
                cube[1][0][0][0] = arena[ieta][ix][iy].Wmunu[0][13];
                cube[1][0][1][0] = arena[ieta][ix][iy+fac_y].Wmunu[0][13];
                cube[1][1][0][0] = arena[ieta][ix+fac_x][iy].Wmunu[0][13];
                cube[1][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].Wmunu[0][13];
                cube[0][0][0][1] = arena[ieta+fac_eta][ix][iy].W_prev[13];
                cube[0][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].W_prev[13];
                cube[0][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].W_prev[13];
                cube[0][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].W_prev[13];
                cube[1][0][0][1] = arena[ieta+fac_eta][ix][iy].Wmunu[0][13];
                cube[1][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].Wmunu[0][13];
                cube[1][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].Wmunu[0][13];
                cube[1][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].Wmunu[0][13];
                qeta_center = 
                    util->four_dimension_linear_interpolation(
                                lattice_spacing_ptr, x_fraction, cube);

                // reconstruct q^\tau from the transverality criteria
                double *u_flow = new double [4];
                u_flow[0] = utau_center;
                u_flow[1] = ux_center;
                u_flow[2] = uy_center;
                u_flow[3] = ueta_center;

                double *q_mu = new double [4];
                q_mu[0] = qtau_center;
                q_mu[1] = qx_center;
                q_mu[2] = qy_center;
                q_mu[3] = qeta_center;
                
                double *q_regulated = new double [4];
                for (int i = 0; i < 4; i++)
                   q_regulated[i] = 0.0;
                
                regulate_qmu(u_flow, q_mu, q_regulated);
                
                qtau_center = q_regulated[0];
                qx_center = q_regulated[1];
                qy_center = q_regulated[2];
                qeta_center = q_regulated[3];
    
                // clean up
                delete [] q_mu;
                delete [] q_regulated;

                // bulk viscous pressure pi_b
                cube[0][0][0][0] = arena[ieta][ix][iy].pi_b_prev;
                cube[0][0][1][0] = arena[ieta][ix][iy+fac_y].pi_b_prev;
                cube[0][1][0][0] = arena[ieta][ix+fac_x][iy].pi_b_prev;
                cube[0][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].pi_b_prev;
                cube[1][0][0][0] = arena[ieta][ix][iy].pi_b[0];
                cube[1][0][1][0] = arena[ieta][ix][iy+fac_y].pi_b[0];
                cube[1][1][0][0] = arena[ieta][ix+fac_x][iy].pi_b[0];
                cube[1][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].pi_b[0];
                cube[0][0][0][1] = arena[ieta+fac_eta][ix][iy].pi_b_prev;
                cube[0][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].pi_b_prev;
                cube[0][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].pi_b_prev;
                cube[0][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].pi_b_prev;
                cube[1][0][0][1] = arena[ieta+fac_eta][ix][iy].pi_b[0];
                cube[1][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].pi_b[0];
                cube[1][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].pi_b[0];
                cube[1][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].pi_b[0];
                pi_b_center = 
                    util->four_dimension_linear_interpolation(
                                lattice_spacing_ptr, x_fraction, cube);

                // shear viscous tensor W^\tau\tau
                cube[0][0][0][0] = arena[ieta][ix][iy].W_prev[0];
                cube[0][0][1][0] = arena[ieta][ix][iy+fac_y].W_prev[0];
                cube[0][1][0][0] = arena[ieta][ix+fac_x][iy].W_prev[0];
                cube[0][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].W_prev[0];
                cube[1][0][0][0] = arena[ieta][ix][iy].Wmunu[0][0];
                cube[1][0][1][0] = arena[ieta][ix][iy+fac_y].Wmunu[0][0];
                cube[1][1][0][0] = arena[ieta][ix+fac_x][iy].Wmunu[0][0];
                cube[1][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].Wmunu[0][0];
                cube[0][0][0][1] = arena[ieta+fac_eta][ix][iy].W_prev[0];
                cube[0][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].W_prev[0];
                cube[0][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].W_prev[0];
                cube[0][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].W_prev[0];
                cube[1][0][0][1] = arena[ieta+fac_eta][ix][iy].Wmunu[0][0];
                cube[1][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].Wmunu[0][0];
                cube[1][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].Wmunu[0][0];
                cube[1][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].Wmunu[0][0];
                Wtautau_center = 
                    util->four_dimension_linear_interpolation(
                                lattice_spacing_ptr, x_fraction, cube);
      
                // shear viscous tensor W^{\tau x}
                cube[0][0][0][0] = arena[ieta][ix][iy].W_prev[1];
                cube[0][0][1][0] = arena[ieta][ix][iy+fac_y].W_prev[1];
                cube[0][1][0][0] = arena[ieta][ix+fac_x][iy].W_prev[1];
                cube[0][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].W_prev[1];
                cube[1][0][0][0] = arena[ieta][ix][iy].Wmunu[0][1];
                cube[1][0][1][0] = arena[ieta][ix][iy+fac_y].Wmunu[0][1];
                cube[1][1][0][0] = arena[ieta][ix+fac_x][iy].Wmunu[0][1];
                cube[1][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].Wmunu[0][1];
                cube[0][0][0][1] = arena[ieta+fac_eta][ix][iy].W_prev[1];
                cube[0][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].W_prev[1];
                cube[0][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].W_prev[1];
                cube[0][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].W_prev[1];
                cube[1][0][0][1] = arena[ieta+fac_eta][ix][iy].Wmunu[0][1];
                cube[1][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].Wmunu[0][1];
                cube[1][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].Wmunu[0][1];
                cube[1][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].Wmunu[0][1];
                Wtaux_center = 
                    util->four_dimension_linear_interpolation(
                                lattice_spacing_ptr, x_fraction, cube);

                // shear viscous tensor W^{\tau y}
                cube[0][0][0][0] = arena[ieta][ix][iy].W_prev[2];
                cube[0][0][1][0] = arena[ieta][ix][iy+fac_y].W_prev[2];
                cube[0][1][0][0] = arena[ieta][ix+fac_x][iy].W_prev[2];
                cube[0][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].W_prev[2];
                cube[1][0][0][0] = arena[ieta][ix][iy].Wmunu[0][2];
                cube[1][0][1][0] = arena[ieta][ix][iy+fac_y].Wmunu[0][2];
                cube[1][1][0][0] = arena[ieta][ix+fac_x][iy].Wmunu[0][2];
                cube[1][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].Wmunu[0][2];
                cube[0][0][0][1] = arena[ieta+fac_eta][ix][iy].W_prev[2];
                cube[0][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].W_prev[2];
                cube[0][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].W_prev[2];
                cube[0][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].W_prev[2];
                cube[1][0][0][1] = arena[ieta+fac_eta][ix][iy].Wmunu[0][2];
                cube[1][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].Wmunu[0][2];
                cube[1][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].Wmunu[0][2];
                cube[1][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].Wmunu[0][2];
                Wtauy_center = 
                    util->four_dimension_linear_interpolation(
                                lattice_spacing_ptr, x_fraction, cube);
      
                // shear viscous tensor W^{\tau \eta}
                cube[0][0][0][0] = arena[ieta][ix][iy].W_prev[3];
                cube[0][0][1][0] = arena[ieta][ix][iy+fac_y].W_prev[3];
                cube[0][1][0][0] = arena[ieta][ix+fac_x][iy].W_prev[3];
                cube[0][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].W_prev[3];
                cube[1][0][0][0] = arena[ieta][ix][iy].Wmunu[0][3];
                cube[1][0][1][0] = arena[ieta][ix][iy+fac_y].Wmunu[0][3];
                cube[1][1][0][0] = arena[ieta][ix+fac_x][iy].Wmunu[0][3];
                cube[1][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].Wmunu[0][3];
                cube[0][0][0][1] = arena[ieta+fac_eta][ix][iy].W_prev[3];
                cube[0][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].W_prev[3];
                cube[0][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].W_prev[3];
                cube[0][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].W_prev[3];
                cube[1][0][0][1] = arena[ieta+fac_eta][ix][iy].Wmunu[0][3];
                cube[1][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].Wmunu[0][3];
                cube[1][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].Wmunu[0][3];
                cube[1][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].Wmunu[0][3];
                Wtaueta_center = 
                    util->four_dimension_linear_interpolation(
                                lattice_spacing_ptr, x_fraction, cube);
      
                // shear viscous tensor W^{xx}
                cube[0][0][0][0] = arena[ieta][ix][iy].W_prev[4];
                cube[0][0][1][0] = arena[ieta][ix][iy+fac_y].W_prev[4];
                cube[0][1][0][0] = arena[ieta][ix+fac_x][iy].W_prev[4];
                cube[0][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].W_prev[4];
                cube[1][0][0][0] = arena[ieta][ix][iy].Wmunu[0][4];
                cube[1][0][1][0] = arena[ieta][ix][iy+fac_y].Wmunu[0][4];
                cube[1][1][0][0] = arena[ieta][ix+fac_x][iy].Wmunu[0][4];
                cube[1][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].Wmunu[0][4];
                cube[0][0][0][1] = arena[ieta+fac_eta][ix][iy].W_prev[4];
                cube[0][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].W_prev[4];
                cube[0][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].W_prev[4];
                cube[0][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].W_prev[4];
                cube[1][0][0][1] = arena[ieta+fac_eta][ix][iy].Wmunu[0][4];
                cube[1][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].Wmunu[0][4];
                cube[1][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].Wmunu[0][4];
                cube[1][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].Wmunu[0][4];
                Wxx_center = 
                    util->four_dimension_linear_interpolation(
                                lattice_spacing_ptr, x_fraction, cube);

                // shear viscous tensor W^{xy}
                cube[0][0][0][0] = arena[ieta][ix][iy].W_prev[5];
                cube[0][0][1][0] = arena[ieta][ix][iy+fac_y].W_prev[5];
                cube[0][1][0][0] = arena[ieta][ix+fac_x][iy].W_prev[5];
                cube[0][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].W_prev[5];
                cube[1][0][0][0] = arena[ieta][ix][iy].Wmunu[0][5];
                cube[1][0][1][0] = arena[ieta][ix][iy+fac_y].Wmunu[0][5];
                cube[1][1][0][0] = arena[ieta][ix+fac_x][iy].Wmunu[0][5];
                cube[1][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].Wmunu[0][5];
                cube[0][0][0][1] = arena[ieta+fac_eta][ix][iy].W_prev[5];
                cube[0][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].W_prev[5];
                cube[0][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].W_prev[5];
                cube[0][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].W_prev[5];
                cube[1][0][0][1] = arena[ieta+fac_eta][ix][iy].Wmunu[0][5];
                cube[1][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].Wmunu[0][5];
                cube[1][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].Wmunu[0][5];
                cube[1][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].Wmunu[0][5];
                Wxy_center = 
                    util->four_dimension_linear_interpolation(
                                lattice_spacing_ptr, x_fraction, cube);

                // shear viscous tensor W^{x\eta}
                cube[0][0][0][0] = arena[ieta][ix][iy].W_prev[6];
                cube[0][0][1][0] = arena[ieta][ix][iy+fac_y].W_prev[6];
                cube[0][1][0][0] = arena[ieta][ix+fac_x][iy].W_prev[6];
                cube[0][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].W_prev[6];
                cube[1][0][0][0] = arena[ieta][ix][iy].Wmunu[0][6];
                cube[1][0][1][0] = arena[ieta][ix][iy+fac_y].Wmunu[0][6];
                cube[1][1][0][0] = arena[ieta][ix+fac_x][iy].Wmunu[0][6];
                cube[1][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].Wmunu[0][6];
                cube[0][0][0][1] = arena[ieta+fac_eta][ix][iy].W_prev[6];
                cube[0][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].W_prev[6];
                cube[0][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].W_prev[6];
                cube[0][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].W_prev[6];
                cube[1][0][0][1] = arena[ieta+fac_eta][ix][iy].Wmunu[0][6];
                cube[1][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].Wmunu[0][6];
                cube[1][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].Wmunu[0][6];
                cube[1][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].Wmunu[0][6];
                Wxeta_center = 
                    util->four_dimension_linear_interpolation(
                                lattice_spacing_ptr, x_fraction, cube);
      
                // shear viscous tensor W^{yy}
                cube[0][0][0][0] = arena[ieta][ix][iy].W_prev[7];
                cube[0][0][1][0] = arena[ieta][ix][iy+fac_y].W_prev[7];
                cube[0][1][0][0] = arena[ieta][ix+fac_x][iy].W_prev[7];
                cube[0][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].W_prev[7];
                cube[1][0][0][0] = arena[ieta][ix][iy].Wmunu[0][7];
                cube[1][0][1][0] = arena[ieta][ix][iy+fac_y].Wmunu[0][7];
                cube[1][1][0][0] = arena[ieta][ix+fac_x][iy].Wmunu[0][7];
                cube[1][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].Wmunu[0][7];
                cube[0][0][0][1] = arena[ieta+fac_eta][ix][iy].W_prev[7];
                cube[0][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].W_prev[7];
                cube[0][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].W_prev[7];
                cube[0][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].W_prev[7];
                cube[1][0][0][1] = arena[ieta+fac_eta][ix][iy].Wmunu[0][7];
                cube[1][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].Wmunu[0][7];
                cube[1][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].Wmunu[0][7];
                cube[1][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].Wmunu[0][7];
                Wyy_center = 
                    util->four_dimension_linear_interpolation(
                                lattice_spacing_ptr, x_fraction, cube);
      
                // shear viscous tensor W^{y\eta}
                cube[0][0][0][0] = arena[ieta][ix][iy].W_prev[8];
                cube[0][0][1][0] = arena[ieta][ix][iy+fac_y].W_prev[8];
                cube[0][1][0][0] = arena[ieta][ix+fac_x][iy].W_prev[8];
                cube[0][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].W_prev[8];
                cube[1][0][0][0] = arena[ieta][ix][iy].Wmunu[0][8];
                cube[1][0][1][0] = arena[ieta][ix][iy+fac_y].Wmunu[0][8];
                cube[1][1][0][0] = arena[ieta][ix+fac_x][iy].Wmunu[0][8];
                cube[1][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].Wmunu[0][8];
                cube[0][0][0][1] = arena[ieta+fac_eta][ix][iy].W_prev[8];
                cube[0][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].W_prev[8];
                cube[0][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].W_prev[8];
                cube[0][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].W_prev[8];
                cube[1][0][0][1] = arena[ieta+fac_eta][ix][iy].Wmunu[0][8];
                cube[1][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].Wmunu[0][8];
                cube[1][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].Wmunu[0][8];
                cube[1][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].Wmunu[0][8];
                Wyeta_center = 
                    util->four_dimension_linear_interpolation(
                                lattice_spacing_ptr, x_fraction, cube);
      
                // shear viscous tensor W^{\eta\eta}
                cube[0][0][0][0] = arena[ieta][ix][iy].W_prev[9];
                cube[0][0][1][0] = arena[ieta][ix][iy+fac_y].W_prev[9];
                cube[0][1][0][0] = arena[ieta][ix+fac_x][iy].W_prev[9];
                cube[0][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].W_prev[9];
                cube[1][0][0][0] = arena[ieta][ix][iy].Wmunu[0][9];
                cube[1][0][1][0] = arena[ieta][ix][iy+fac_y].Wmunu[0][9];
                cube[1][1][0][0] = arena[ieta][ix+fac_x][iy].Wmunu[0][9];
                cube[1][1][1][0] = arena[ieta][ix+fac_x][iy+fac_y].Wmunu[0][9];
                cube[0][0][0][1] = arena[ieta+fac_eta][ix][iy].W_prev[9];
                cube[0][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].W_prev[9];
                cube[0][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].W_prev[9];
                cube[0][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].W_prev[9];
                cube[1][0][0][1] = arena[ieta+fac_eta][ix][iy].Wmunu[0][9];
                cube[1][0][1][1] = arena[ieta+fac_eta][ix][iy+fac_y].Wmunu[0][9];
                cube[1][1][0][1] = arena[ieta+fac_eta][ix+fac_x][iy].Wmunu[0][9];
                cube[1][1][1][1] = arena[ieta+fac_eta][ix+fac_x][iy+fac_y].Wmunu[0][9];
                Wetaeta_center = 
                    util->four_dimension_linear_interpolation(
                                lattice_spacing_ptr, x_fraction, cube);

                // regulate Wmunu according to transversality and traceless
                double **Wmunu_input = new double* [4];
                double **Wmunu_regulated = new double* [4];
                for (int i = 0; i < 4; i++) {
                    Wmunu_input[i] = new double [4];
                    Wmunu_regulated[i] = new double [4];
                    for (int j = 0; j < 4; j++)
                        Wmunu_regulated[i][j] = 0.0;
                }
                Wmunu_input[0][0] = Wtautau_center;
                Wmunu_input[0][1] = Wmunu_input[1][0] = Wtaux_center;
                Wmunu_input[0][2] = Wmunu_input[2][0] = Wtauy_center;
                Wmunu_input[0][3] = Wmunu_input[3][0] = Wtaueta_center;
                Wmunu_input[1][1] = Wxx_center;
                Wmunu_input[1][2] = Wmunu_input[2][1] = Wxy_center;
                Wmunu_input[1][3] = Wmunu_input[3][1] = Wxeta_center;
                Wmunu_input[2][2] = Wyy_center;
                Wmunu_input[2][3] = Wmunu_input[3][2] = Wyeta_center;
                Wmunu_input[3][3] = Wetaeta_center;
                regulate_Wmunu(u_flow, Wmunu_input, Wmunu_regulated);
                Wtautau_center = Wmunu_regulated[0][0];
                Wtaux_center = Wmunu_regulated[0][1];
                Wtauy_center = Wmunu_regulated[0][2];
                Wtaueta_center = Wmunu_regulated[0][3];
                Wxx_center = Wmunu_regulated[1][1];
                Wxy_center = Wmunu_regulated[1][2];
                Wxeta_center = Wmunu_regulated[1][3];
                Wyy_center = Wmunu_regulated[2][2];
                Wyeta_center = Wmunu_regulated[2][3];
                Wetaeta_center = Wmunu_regulated[3][3];

                // clean up
                delete [] u_flow;
                for (int i = 0; i < 4; i++) {
                    delete [] Wmunu_input[i];
                    delete [] Wmunu_regulated[i];
                }
                delete [] Wmunu_input;
                delete [] Wmunu_regulated;

                // 4-dimension interpolation done
                TFO = eos->get_temperature(epsFO, rhob_center);
                muB = eos->get_mu(epsFO, rhob_center);
                if (TFO < 0) {
                    cout << "TFO=" << TFO 
                         << "<0. ERROR. exiting." << endl;
                    exit(1);
                }

                pressure = eos->get_pressure(epsFO, rhob_center);
                eps_plus_p_over_T_FO = (epsFO + pressure)/TFO;

                // finally output results !!!!
                s_file << scientific << setprecision(10) 
                       << tau_center << " " << x_center << " " 
                       << y_center << " " << eta_center << " " 
                       << FULLSU[0] << " " << FULLSU[1] << " " 
                       << FULLSU[2] << " " << FULLSU[3] << " " 
                       << utau_center << " " << ux_center << " " 
                       << uy_center << " " << ueta_center << " " 
                       << epsFO << " " << TFO << " " << muB << " " 
                       << eps_plus_p_over_T_FO << " " 
                       << Wtautau_center << " " << Wtaux_center << " " 
                       << Wtauy_center << " " << Wtaueta_center << " " 
                       << Wxx_center << " " << Wxy_center << " " 
                       << Wxeta_center << " " 
                       << Wyy_center << " " << Wyeta_center << " " 
                       << Wetaeta_center << " " ;
                if(DATA->turn_on_bulk)
                    s_file << pi_b_center << " " ;
                if(DATA->turn_on_rhob)
                    s_file << rhob_center << " " ;
                if(DATA->turn_on_diff)
                    s_file << qtau_center << " " << qx_center << " " 
                           << qy_center << " " << qeta_center << " " ;
                s_file << endl;
            }
        }
    }
    s_file.close();

    // clean up
    for(int i = 0; i < 2; i++)
    {
        for(int j = 0; j < 2; j++)
        {
            for(int k = 0; k < 2; k++)
                delete [] cube[i][j][k];
            delete [] cube[i][j];
        }
        delete [] cube[i];
        delete [] x_fraction[i];
    }
    delete [] cube;
    delete [] x_fraction;
    delete [] lattice_spacing_ptr;
    delete cornelius_ptr;
    return(intersections);
}

// Cornelius freeze out (C. Shen, 11/2014)
int Evolve::FreezeOut_equal_tau_Surface(double tau, InitData *DATA,
                                        Grid ***arena) {
    // this function freeze-out fluid cells between epsFO and epsFO_low
    // on an equal time hyper-surface at the first time step
    // this function will be trigged if freezeout_lowtemp_flag == 1
    int neta = grid_neta;
    int fac_eta = 1;
   
    for (int i_freezesurf = 0; i_freezesurf < n_freeze_surf; i_freezesurf++) {
        double epsFO = epsFO_list[i_freezesurf]/hbarc;
        int ieta;
        #pragma omp parallel private(ieta)
        {
            #pragma omp for
            for (ieta = 0; ieta < neta - fac_eta; ieta += fac_eta) {
                int thread_id = omp_get_thread_num();
                FreezeOut_equal_tau_Surface_XY(tau, DATA, ieta, arena,
                                               thread_id, epsFO);
            }
            #pragma omp barrier
        }
    }
    return(0);
}

void Evolve::FreezeOut_equal_tau_Surface_XY(double tau, InitData *DATA,
                                            int ieta, Grid ***arena,
                                            int thread_id, double epsFO) {

    double epsFO_low = 0.05/hbarc;        // 1/fm^4

    int nx = grid_nx;
    int ny = grid_ny;

    stringstream strs_name;
    strs_name << "surface_eps_" << setprecision(4) << epsFO*hbarc
              << "_" << thread_id << ".dat";
    ofstream s_file;
    s_file.open(strs_name.str().c_str(), ios::out | ios::app);

    double FULLSU[4];  // d^3 \sigma_\mu

    double tau_center, x_center, y_center, eta_center;
    double Wtautau_center, Wtaux_center, Wtauy_center, Wtaueta_center;
    double Wxx_center, Wxy_center, Wxeta_center;
    double Wyy_center, Wyeta_center, Wetaeta_center;
    double rhob_center;
    double qtau_center, qx_center, qy_center, qeta_center;
    double utau_center, ux_center, uy_center, ueta_center;
    double pi_b_center; // bulk viscous pressure

    int intersect;
    int intersections = 0;

    int fac_x = DATA->fac_x;
    int fac_y = DATA->fac_y;
    int fac_eta = 1;
    
    double DX = fac_x*DATA->delta_x;
    double DY = fac_y*DATA->delta_y;
    double DETA = fac_eta*DATA->delta_eta;

    double eta = (DATA->delta_eta)*ieta - (DATA->eta_size)/2.0;
    for (int ix = 0; ix <= nx - fac_x; ix += fac_x) {
        double x = ix*(DATA->delta_x) - (DATA->x_size/2.0); 
        for (int iy = 0; iy <= ny - fac_y; iy += fac_y) {
            double y = iy*(DATA->delta_y) - (DATA->y_size/2.0);

            // judge intersection
            intersect = 0;
            if (arena[ieta][ix][iy].epsilon < epsFO
                && arena[ieta][ix][iy].epsilon > epsFO_low)
                intersect = 1;

            if (intersect == 0) {
                continue;
            }
            
            // if intersect output the freeze-out cell
            intersections++;

            // surface normal vector d^3 \sigma_\mu
            FULLSU[0] = DX*DY*DETA;
            FULLSU[1] = 0.0;
            FULLSU[2] = 0.0;
            FULLSU[3] = 0.0;

            // get positions of the freeze-out surface
            tau_center = tau;
            x_center = x;
            y_center = y;
            eta_center = eta;

            // flow velocity
            ux_center = arena[ieta][ix][iy].u[0][1];
            uy_center = arena[ieta][ix][iy].u[0][2];
            ueta_center = arena[ieta][ix][iy].u[0][3];  // u^eta/tau
            // reconstruct u^tau from u^i
            utau_center = sqrt(1. + ux_center*ux_center 
                               + uy_center*uy_center 
                               + ueta_center*ueta_center);

            // baryon density rho_b
            rhob_center = arena[ieta][ix][iy].rhob;

            // baryon diffusion current
            qtau_center = arena[ieta][ix][iy].Wmunu[0][10];
            qx_center = arena[ieta][ix][iy].Wmunu[0][11];
            qy_center = arena[ieta][ix][iy].Wmunu[0][12];
            qeta_center = arena[ieta][ix][iy].Wmunu[0][13];
            // reconstruct q^\tau from the transverality criteria
            double *u_flow = new double [4];
            u_flow[0] = utau_center;
            u_flow[1] = ux_center;
            u_flow[2] = uy_center;
            u_flow[3] = ueta_center;
            double *q_mu = new double [4];
            q_mu[0] = qtau_center;
            q_mu[1] = qx_center;
            q_mu[2] = qy_center;
            q_mu[3] = qeta_center;
            double *q_regulated = new double [4];
            for(int i = 0; i < 4; i++)
               q_regulated[i] = 0.0;
            regulate_qmu(u_flow, q_mu, q_regulated);
            qtau_center = q_regulated[0];
            qx_center = q_regulated[1];
            qy_center = q_regulated[2];
            qeta_center = q_regulated[3];
            // clean up
            delete [] q_mu;
            delete [] q_regulated;

            // bulk viscous pressure pi_b
            pi_b_center = arena[ieta][ix][iy].pi_b[0];

            // shear viscous tensor
            Wtautau_center = arena[ieta][ix][iy].Wmunu[0][0];
            Wtaux_center = arena[ieta][ix][iy].Wmunu[0][1];
            Wtauy_center = arena[ieta][ix][iy].Wmunu[0][2];
            Wtaueta_center = arena[ieta][ix][iy].Wmunu[0][3];
            Wxx_center = arena[ieta][ix][iy].Wmunu[0][4];
            Wxy_center = arena[ieta][ix][iy].Wmunu[0][5];
            Wxeta_center = arena[ieta][ix][iy].Wmunu[0][6];
            Wyy_center = arena[ieta][ix][iy].Wmunu[0][7];
            Wyeta_center = arena[ieta][ix][iy].Wmunu[0][8];
            Wetaeta_center = arena[ieta][ix][iy].Wmunu[0][9];
            // regulate Wmunu according to transversality and traceless
            double **Wmunu_input = new double* [4];
            double **Wmunu_regulated = new double* [4];
            for (int i = 0; i < 4; i++) {
                Wmunu_input[i] = new double[4];
                Wmunu_regulated[i] = new double[4];
                for (int j = 0; j < 4; j++)
                    Wmunu_regulated[i][j] = 0.0;
            }
            Wmunu_input[0][0] = Wtautau_center;
            Wmunu_input[0][1] = Wmunu_input[1][0] = Wtaux_center;
            Wmunu_input[0][2] = Wmunu_input[2][0] = Wtauy_center;
            Wmunu_input[0][3] = Wmunu_input[3][0] = Wtaueta_center;
            Wmunu_input[1][1] = Wxx_center;
            Wmunu_input[1][2] = Wmunu_input[2][1] = Wxy_center;
            Wmunu_input[1][3] = Wmunu_input[3][1] = Wxeta_center;
            Wmunu_input[2][2] = Wyy_center;
            Wmunu_input[2][3] = Wmunu_input[3][2] = Wyeta_center;
            Wmunu_input[3][3] = Wetaeta_center;
            regulate_Wmunu(u_flow, Wmunu_input, Wmunu_regulated);
            Wtautau_center = Wmunu_regulated[0][0];
            Wtaux_center = Wmunu_regulated[0][1];
            Wtauy_center = Wmunu_regulated[0][2];
            Wtaueta_center = Wmunu_regulated[0][3];
            Wxx_center = Wmunu_regulated[1][1];
            Wxy_center = Wmunu_regulated[1][2];
            Wxeta_center = Wmunu_regulated[1][3];
            Wyy_center = Wmunu_regulated[2][2];
            Wyeta_center = Wmunu_regulated[2][3];
            Wetaeta_center = Wmunu_regulated[3][3];
            // clean up
            delete [] u_flow;
            for (int i = 0; i < 4; i++) {
                delete [] Wmunu_input[i];
                delete [] Wmunu_regulated[i];
            }
            delete [] Wmunu_input;
            delete [] Wmunu_regulated;

            // get other thermodynamical quantities
            double e_local = arena[ieta][ix][iy].epsilon;
            double T_local = eos->get_temperature(e_local, rhob_center);
            double muB_local = eos->get_mu(e_local, rhob_center);
            if (T_local < 0) {
                cout << "Error:Evolve::FreezeOut_equal_tau_Surface: "
                     << "T_local = " << T_local
                     << " <0. ERROR. exiting." << endl;
                exit(1);
            }

            double pressure = eos->get_pressure(e_local, rhob_center);
            double eps_plus_p_over_T = (e_local + pressure)/T_local;

            // finally output results !!!!
            s_file << scientific << setprecision(10) 
                   << tau_center << " " << x_center << " " 
                   << y_center << " " << eta_center << " " 
                   << FULLSU[0] << " " << FULLSU[1] << " " 
                   << FULLSU[2] << " " << FULLSU[3] << " " 
                   << utau_center << " " << ux_center << " " 
                   << uy_center << " " << ueta_center << " " 
                   << e_local << " " << T_local << " "
                   << muB_local << " " 
                   << eps_plus_p_over_T << " " 
                   << Wtautau_center << " " << Wtaux_center << " " 
                   << Wtauy_center << " " << Wtaueta_center << " " 
                   << Wxx_center << " " << Wxy_center << " " 
                   << Wxeta_center << " " 
                   << Wyy_center << " " << Wyeta_center << " " 
                   << Wetaeta_center << " " ;
            if (DATA->turn_on_bulk)
                s_file << pi_b_center << " " ;
            if (DATA->turn_on_rhob)
                s_file << rhob_center << " " ;
            if (DATA->turn_on_diff)
                s_file << qtau_center << " " << qx_center << " " 
                       << qy_center << " " << qeta_center << " " ;
            s_file << endl;
        }
    }
    s_file.close();
}

int Evolve::FindFreezeOutSurface_boostinvariant_Cornelius(
                                double tau, InitData *DATA, Grid ***arena) {
    // find boost-invariant hyper-surfaces
    int *all_frozen = new int [n_freeze_surf];
    for (int i_freezesurf = 0; i_freezesurf < n_freeze_surf; i_freezesurf++) {
        double epsFO;
        epsFO = epsFO_list[i_freezesurf]/hbarc;

        stringstream strs_name;
        strs_name << "surface_eps_" << setprecision(4) << epsFO*hbarc
                  << ".dat";

        ofstream s_file;
        s_file.open(strs_name.str().c_str() , ios::out | ios::app );
    
        int nx = grid_nx;
        int ny = grid_ny;
        double FULLSU[4];  // d^3 \sigma_\mu
    
        double tau_center, x_center, y_center, eta_center;
        double Wtautau_center, Wtaux_center, Wtauy_center, Wtaueta_center;
        double Wxx_center, Wxy_center, Wxeta_center;
        double Wyy_center, Wyeta_center, Wetaeta_center;
        double rhob_center;
        double qtau_center, qx_center, qy_center, qeta_center;
        double utau_center, ux_center, uy_center, ueta_center;
        double TFO, muB, pressure, eps_plus_p_over_T_FO;
        double pi_b_center;  // bulk viscous pressure
    
        int dim = 3;
        double lattice_spacing[dim];
    
        double *lattice_spacing_ptr = new double [dim];
        // for 4-d linear interpolation
        double** x_fraction = new double* [2];
        for (int i = 0; i < 2; i++)
            x_fraction[i] = new double[dim];

        int intersect;
        int intersections = 0;

        facTau = DATA->facTau;   // step to skip in tau direction
        int fac_x = DATA->fac_x;
        int fac_y = DATA->fac_y;

        double DX=fac_x*DATA->delta_x;
        double DY=fac_y*DATA->delta_y;
        double DETA=1.0;
        double DTAU=facTau*DATA->delta_tau;

        lattice_spacing[0] = DTAU;
        lattice_spacing[1] = DX;
        lattice_spacing[2] = DY;
    
        lattice_spacing_ptr[0] = DTAU;
        lattice_spacing_ptr[1] = DX;
        lattice_spacing_ptr[2] = DY;

        // initialize Cornelius
        Cornelius* cornelius_ptr = new Cornelius();
        cornelius_ptr->init(dim, epsFO, lattice_spacing);

        // initialize the hyper-cube for Cornelius
        double ***cube = new double ** [2];
        for (int i = 0; i < 2; i++) {
            cube[i] = new double * [2];
            for (int j = 0; j < 2; j++) {
                cube[i][j] = new double[2];
                for (int k = 0; k < 2; k++)
                    cube[i][j][k] = 0.0;
            }
        }
  
        for (int ix=0; ix <= nx - fac_x; ix += fac_x) {
            double x = ix*(DATA->delta_x) - (DATA->x_size/2.0); 
            for (int iy=0; iy <= ny - fac_y; iy += fac_y) {
                double y = iy*(DATA->delta_y) - (DATA->y_size/2.0);
    
                // make sure the epsilon value is never 
                // exactly the same as epsFO...
                if(arena[0][ix+fac_x][iy+fac_y].epsilon==epsFO)
                    arena[0][ix+fac_x][iy+fac_y].epsilon+=0.000001;
                if(arena[0][ix+fac_x][iy+fac_y].epsilon_prev==epsFO)
                    arena[0][ix+fac_x][iy+fac_y].epsilon_prev+=0.000001;
                if(arena[0][ix][iy].epsilon==epsFO)
                    arena[0][ix][iy].epsilon+=0.000001;
                if(arena[0][ix][iy].epsilon_prev==epsFO)
                    arena[0][ix][iy].epsilon_prev+=0.000001;
                if(arena[0][ix+fac_x][iy].epsilon==epsFO)
                    arena[0][ix+fac_x][iy].epsilon+=0.000001;
                if(arena[0][ix+fac_x][iy].epsilon_prev==epsFO)
                    arena[0][ix+fac_x][iy].epsilon_prev+=0.000001;
                if(arena[0][ix][iy+fac_y].epsilon==epsFO)
                    arena[0][ix][iy+fac_y].epsilon+=0.000001;
                if(arena[0][ix][iy+fac_y].epsilon_prev==epsFO)
                    arena[0][ix][iy+fac_y].epsilon_prev+=0.000001;
               
                // judge intersection (from Bjoern)
                intersect=1;
                if ((arena[0][ix+fac_x][iy+fac_y].epsilon-epsFO)
                    *(arena[0][ix][iy].epsilon_prev-epsFO) > 0.)
                    if ((arena[0][ix+fac_x][iy].epsilon-epsFO)
                        *(arena[0][ix][iy+fac_y].epsilon_prev-epsFO) > 0.)
                        if ((arena[0][ix][iy+fac_y].epsilon-epsFO)
                            *(arena[0][ix+fac_x][iy].epsilon_prev-epsFO) > 0.)
                            if ((arena[0][ix][iy].epsilon-epsFO)
                                *(arena[0][ix+fac_x][iy+fac_y].epsilon_prev-epsFO) > 0.)
                                    intersect = 0;
                if (intersect == 0) {
                    continue;
                } else {
                    // if intersect, prepare for the hyper-cube
                    intersections++;
                    cube[0][0][0] = arena[0][ix][iy].epsilon_prev;
                    cube[0][0][1] = arena[0][ix][iy+fac_y].epsilon_prev;
                    cube[0][1][0] = arena[0][ix+fac_x][iy].epsilon_prev;
                    cube[0][1][1] = arena[0][ix+fac_x][iy+fac_y].epsilon_prev;
                    cube[1][0][0] = arena[0][ix][iy].epsilon;
                    cube[1][0][1] = arena[0][ix][iy+fac_y].epsilon;
                    cube[1][1][0] = arena[0][ix+fac_x][iy].epsilon;
                    cube[1][1][1] = arena[0][ix+fac_x][iy+fac_y].epsilon;
                }
           
                // Now, the magic will happen in the Cornelius ...
                cornelius_ptr->find_surface_3d(cube);

                // get positions of the freeze-out surface 
                // and interpolating results
                for (int isurf = 0; isurf < cornelius_ptr->get_Nelements(); 
                     isurf++) {
                    // surface normal vector d^3 \sigma_\mu
                    for (int ii = 0; ii < dim; ii++)
                        FULLSU[ii] = cornelius_ptr->get_normal_elem(isurf, ii);

                    FULLSU[3] = 0.0; // rapidity direction is set to 0

                    // check the size of the surface normal vector
                    if (fabs(FULLSU[0]) > (DX*DY*DETA + 0.01)) {
                       cerr << "problem: volume in tau direction " 
                            << fabs(FULLSU[0]) << "  > DX*DY*DETA = "  
                            << DX*DY*DETA << endl;      
                       //FULLSU[0] = DX*DY*DETA*(FULLSU[0])/fabs(FULLSU[0]);
                    }
                    if (fabs(FULLSU[1]) > (DTAU*DY*DETA + 0.01)) {
                       cerr << "problem: volume in x direction " 
                            << fabs(FULLSU[1]) << "  > DTAU*DY*DETA = "  
                            << DTAU*DY*DETA << endl;
                        //FULLSU[1] = DTAU*DY*DETA*(FULLSU[1])/fabs(FULLSU[1]);
                    }
                    if (fabs(FULLSU[2]) > (DX*DTAU*DETA+0.01)) {
                        cerr << "problem: volume in y direction " 
                             << fabs(FULLSU[2]) << "  > DX*DTAU*DETA = "  
                             << DX*DTAU*DETA << endl;
                        //FULLSU[2] = DX*DTAU*DETA*(FULLSU[2])/fabs(FULLSU[2]);
                    }

                    // position of the freeze-out fluid cell
                    for (int ii = 0; ii < dim; ii++) {
                        x_fraction[1][ii] = 
                            cornelius_ptr->get_centroid_elem(isurf, ii);
                        x_fraction[0][ii] = (
                            lattice_spacing_ptr[ii] - x_fraction[1][ii]);
                    }
                    tau_center = tau - DTAU + x_fraction[1][0];
                    x_center = x + x_fraction[1][1];
                    y_center = y + x_fraction[1][2];
                    eta_center = 0.0;

                    // perform 3-d linear interpolation for all fluid quantities

                    // flow velocity u^\tau
                    int ieta = 0;
                    cube[0][0][0] = arena[ieta][ix][iy].u_prev[0];
                    cube[0][0][1] = arena[ieta][ix][iy+fac_y].u_prev[0];
                    cube[0][1][0] = arena[ieta][ix+fac_x][iy].u_prev[0];
                    cube[0][1][1] = arena[ieta][ix+fac_x][iy+fac_y].u_prev[0];
                    cube[1][0][0] = arena[ieta][ix][iy].u[0][0];
                    cube[1][0][1] = arena[ieta][ix][iy+fac_y].u[0][0];
                    cube[1][1][0] = arena[ieta][ix+fac_x][iy].u[0][0];
                    cube[1][1][1] = arena[ieta][ix+fac_x][iy+fac_y].u[0][0];
                    utau_center = util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);

                    // flow velocity u^x
                    cube[0][0][0] = arena[ieta][ix][iy].u_prev[1];
                    cube[0][0][1] = arena[ieta][ix][iy+fac_y].u_prev[1];
                    cube[0][1][0] = arena[ieta][ix+fac_x][iy].u_prev[1];
                    cube[0][1][1] = arena[ieta][ix+fac_x][iy+fac_y].u_prev[1];
                    cube[1][0][0] = arena[ieta][ix][iy].u[0][1];
                    cube[1][0][1] = arena[ieta][ix][iy+fac_y].u[0][1];
                    cube[1][1][0] = arena[ieta][ix+fac_x][iy].u[0][1];
                    cube[1][1][1] = arena[ieta][ix+fac_x][iy+fac_y].u[0][1];
                    ux_center = util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);

                    // flow velocity u^y
                    cube[0][0][0] = arena[ieta][ix][iy].u_prev[2];
                    cube[0][0][1] = arena[ieta][ix][iy+fac_y].u_prev[2];
                    cube[0][1][0] = arena[ieta][ix+fac_x][iy].u_prev[2];
                    cube[0][1][1] = arena[ieta][ix+fac_x][iy+fac_y].u_prev[2];
                    cube[1][0][0] = arena[ieta][ix][iy].u[0][2];
                    cube[1][0][1] = arena[ieta][ix][iy+fac_y].u[0][2];
                    cube[1][1][0] = arena[ieta][ix+fac_x][iy].u[0][2];
                    cube[1][1][1] = arena[ieta][ix+fac_x][iy+fac_y].u[0][2];
                    uy_center = util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);

                    // flow velocity u^eta
                    cube[0][0][0] = arena[ieta][ix][iy].u_prev[3];
                    cube[0][0][1] = arena[ieta][ix][iy+fac_y].u_prev[3];
                    cube[0][1][0] = arena[ieta][ix+fac_x][iy].u_prev[3];
                    cube[0][1][1] = arena[ieta][ix+fac_x][iy+fac_y].u_prev[3];
                    cube[1][0][0] = arena[ieta][ix][iy].u[0][3];
                    cube[1][0][1] = arena[ieta][ix][iy+fac_y].u[0][3];
                    cube[1][1][0] = arena[ieta][ix+fac_x][iy].u[0][3];
                    cube[1][1][1] = arena[ieta][ix+fac_x][iy+fac_y].u[0][3];
                    ueta_center = util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);

                    // baryon density rho_b
                    cube[0][0][0] = arena[ieta][ix][iy].rhob_prev;
                    cube[0][0][1] = arena[ieta][ix][iy+fac_y].rhob_prev;
                    cube[0][1][0] = arena[ieta][ix+fac_x][iy].rhob_prev;
                    cube[0][1][1] = arena[ieta][ix+fac_x][iy+fac_y].rhob_prev;
                    cube[1][0][0] = arena[ieta][ix][iy].rhob;
                    cube[1][0][1] = arena[ieta][ix][iy+fac_y].rhob;
                    cube[1][1][0] = arena[ieta][ix+fac_x][iy].rhob;
                    cube[1][1][1] = arena[ieta][ix+fac_x][iy+fac_y].rhob;
                    rhob_center = util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);

                    // bulk viscous pressure pi_b
                    cube[0][0][0] = arena[ieta][ix][iy].pi_b_prev;
                    cube[0][0][1] = arena[ieta][ix][iy+fac_y].pi_b_prev;
                    cube[0][1][0] = arena[ieta][ix+fac_x][iy].pi_b_prev;
                    cube[0][1][1] = arena[ieta][ix+fac_x][iy+fac_y].pi_b_prev;
                    cube[1][0][0] = arena[ieta][ix][iy].pi_b[0];
                    cube[1][0][1] = arena[ieta][ix][iy+fac_y].pi_b[0];
                    cube[1][1][0] = arena[ieta][ix+fac_x][iy].pi_b[0];
                    cube[1][1][1] = arena[ieta][ix+fac_x][iy+fac_y].pi_b[0];
                    pi_b_center = util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);
               
                    // baryon diffusion current q^\tau
                    cube[0][0][0] = arena[ieta][ix][iy].W_prev[10];
                    cube[0][0][1] = arena[ieta][ix][iy+fac_y].W_prev[10];
                    cube[0][1][0] = arena[ieta][ix+fac_x][iy].W_prev[10];
                    cube[0][1][1] = arena[ieta][ix+fac_x][iy+fac_y].W_prev[10];
                    cube[1][0][0] = arena[ieta][ix][iy].Wmunu[0][10];
                    cube[1][0][1] = arena[ieta][ix][iy+fac_y].Wmunu[0][10];
                    cube[1][1][0] = arena[ieta][ix+fac_x][iy].Wmunu[0][10];
                    cube[1][1][1] = arena[ieta][ix+fac_x][iy+fac_y].Wmunu[0][10];
                    qtau_center = util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);
               
                    // baryon diffusion current q^x
                    cube[0][0][0] = arena[ieta][ix][iy].W_prev[11];
                    cube[0][0][1] = arena[ieta][ix][iy+fac_y].W_prev[11];
                    cube[0][1][0] = arena[ieta][ix+fac_x][iy].W_prev[11];
                    cube[0][1][1] = arena[ieta][ix+fac_x][iy+fac_y].W_prev[11];
                    cube[1][0][0] = arena[ieta][ix][iy].Wmunu[0][11];
                    cube[1][0][1] = arena[ieta][ix][iy+fac_y].Wmunu[0][11];
                    cube[1][1][0] = arena[ieta][ix+fac_x][iy].Wmunu[0][11];
                    cube[1][1][1] = arena[ieta][ix+fac_x][iy+fac_y].Wmunu[0][11];
                    qx_center = util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);
               
                    // baryon diffusion current q^y
                    cube[0][0][0] = arena[ieta][ix][iy].W_prev[12];
                    cube[0][0][1] = arena[ieta][ix][iy+fac_y].W_prev[12];
                    cube[0][1][0] = arena[ieta][ix+fac_x][iy].W_prev[12];
                    cube[0][1][1] = arena[ieta][ix+fac_x][iy+fac_y].W_prev[12];
                    cube[1][0][0] = arena[ieta][ix][iy].Wmunu[0][12];
                    cube[1][0][1] = arena[ieta][ix][iy+fac_y].Wmunu[0][12];
                    cube[1][1][0] = arena[ieta][ix+fac_x][iy].Wmunu[0][12];
                    cube[1][1][1] = arena[ieta][ix+fac_x][iy+fac_y].Wmunu[0][12];
                    qy_center = util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);
               
                    // baryon diffusion current q^eta
                    cube[0][0][0] = arena[ieta][ix][iy].W_prev[13];
                    cube[0][0][1] = arena[ieta][ix][iy+fac_y].W_prev[13];
                    cube[0][1][0] = arena[ieta][ix+fac_x][iy].W_prev[13];
                    cube[0][1][1] = arena[ieta][ix+fac_x][iy+fac_y].W_prev[13];
                    cube[1][0][0] = arena[ieta][ix][iy].Wmunu[0][13];
                    cube[1][0][1] = arena[ieta][ix][iy+fac_y].Wmunu[0][13];
                    cube[1][1][0] = arena[ieta][ix+fac_x][iy].Wmunu[0][13];
                    cube[1][1][1] = arena[ieta][ix+fac_x][iy+fac_y].Wmunu[0][13];
                    qeta_center = util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);

                    // reconstruct q^\tau from the transverality criteria
                    double *u_flow = new double [4];
                    u_flow[0] = utau_center;
                    u_flow[1] = ux_center;
                    u_flow[2] = uy_center;
                    u_flow[3] = ueta_center;

                    double *q_mu = new double [4];
                    q_mu[0] = qtau_center;
                    q_mu[1] = qx_center;
                    q_mu[2] = qy_center;
                    q_mu[3] = qeta_center;

                    double *q_regulated = new double [4];
                    for(int i = 0; i < 4; i++)
                        q_regulated[i] = 0.0;

                    regulate_qmu(u_flow, q_mu, q_regulated);

                    qtau_center = q_regulated[0];
                    qx_center = q_regulated[1];
                    qy_center = q_regulated[2];
                    qeta_center = q_regulated[3];

                    // clean up
                    delete [] q_mu;
                    delete [] q_regulated;

                    // shear viscous tensor W^\tau\tau
                    cube[0][0][0] = arena[ieta][ix][iy].W_prev[0];
                    cube[0][0][1] = arena[ieta][ix][iy+fac_y].W_prev[0];
                    cube[0][1][0] = arena[ieta][ix+fac_x][iy].W_prev[0];
                    cube[0][1][1] = arena[ieta][ix+fac_x][iy+fac_y].W_prev[0];
                    cube[1][0][0] = arena[ieta][ix][iy].Wmunu[0][0];
                    cube[1][0][1] = arena[ieta][ix][iy+fac_y].Wmunu[0][0];
                    cube[1][1][0] = arena[ieta][ix+fac_x][iy].Wmunu[0][0];
                    cube[1][1][1] = arena[ieta][ix+fac_x][iy+fac_y].Wmunu[0][0];
                    Wtautau_center = 
                        util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);
                  
                    // shear viscous tensor W^{\tau x}
                    cube[0][0][0] = arena[ieta][ix][iy].W_prev[1];
                    cube[0][0][1] = arena[ieta][ix][iy+fac_y].W_prev[1];
                    cube[0][1][0] = arena[ieta][ix+fac_x][iy].W_prev[1];
                    cube[0][1][1] = arena[ieta][ix+fac_x][iy+fac_y].W_prev[1];
                    cube[1][0][0] = arena[ieta][ix][iy].Wmunu[0][1];
                    cube[1][0][1] = arena[ieta][ix][iy+fac_y].Wmunu[0][1];
                    cube[1][1][0] = arena[ieta][ix+fac_x][iy].Wmunu[0][1];
                    cube[1][1][1] = arena[ieta][ix+fac_x][iy+fac_y].Wmunu[0][1];
                    Wtaux_center = 
                        util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);

                    // shear viscous tensor W^{\tau y}
                    cube[0][0][0] = arena[ieta][ix][iy].W_prev[2];
                    cube[0][0][1] = arena[ieta][ix][iy+fac_y].W_prev[2];
                    cube[0][1][0] = arena[ieta][ix+fac_x][iy].W_prev[2];
                    cube[0][1][1] = arena[ieta][ix+fac_x][iy+fac_y].W_prev[2];
                    cube[1][0][0] = arena[ieta][ix][iy].Wmunu[0][2];
                    cube[1][0][1] = arena[ieta][ix][iy+fac_y].Wmunu[0][2];
                    cube[1][1][0] = arena[ieta][ix+fac_x][iy].Wmunu[0][2];
                    cube[1][1][1] = arena[ieta][ix+fac_x][iy+fac_y].Wmunu[0][2];
                    Wtauy_center = 
                        util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);
                  
                    // shear viscous tensor W^{\tau \eta}
                    cube[0][0][0] = arena[ieta][ix][iy].W_prev[3];
                    cube[0][0][1] = arena[ieta][ix][iy+fac_y].W_prev[3];
                    cube[0][1][0] = arena[ieta][ix+fac_x][iy].W_prev[3];
                    cube[0][1][1] = arena[ieta][ix+fac_x][iy+fac_y].W_prev[3];
                    cube[1][0][0] = arena[ieta][ix][iy].Wmunu[0][3];
                    cube[1][0][1] = arena[ieta][ix][iy+fac_y].Wmunu[0][3];
                    cube[1][1][0] = arena[ieta][ix+fac_x][iy].Wmunu[0][3];
                    cube[1][1][1] = arena[ieta][ix+fac_x][iy+fac_y].Wmunu[0][3];
                    Wtaueta_center = 
                        util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);
                  
                    // shear viscous tensor W^{xx}
                    cube[0][0][0] = arena[ieta][ix][iy].W_prev[4];
                    cube[0][0][1] = arena[ieta][ix][iy+fac_y].W_prev[4];
                    cube[0][1][0] = arena[ieta][ix+fac_x][iy].W_prev[4];
                    cube[0][1][1] = arena[ieta][ix+fac_x][iy+fac_y].W_prev[4];
                    cube[1][0][0] = arena[ieta][ix][iy].Wmunu[0][4];
                    cube[1][0][1] = arena[ieta][ix][iy+fac_y].Wmunu[0][4];
                    cube[1][1][0] = arena[ieta][ix+fac_x][iy].Wmunu[0][4];
                    cube[1][1][1] = arena[ieta][ix+fac_x][iy+fac_y].Wmunu[0][4];
                    Wxx_center = 
                        util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);

                    // shear viscous tensor W^{xy}
                    cube[0][0][0] = arena[ieta][ix][iy].W_prev[5];
                    cube[0][0][1] = arena[ieta][ix][iy+fac_y].W_prev[5];
                    cube[0][1][0] = arena[ieta][ix+fac_x][iy].W_prev[5];
                    cube[0][1][1] = arena[ieta][ix+fac_x][iy+fac_y].W_prev[5];
                    cube[1][0][0] = arena[ieta][ix][iy].Wmunu[0][5];
                    cube[1][0][1] = arena[ieta][ix][iy+fac_y].Wmunu[0][5];
                    cube[1][1][0] = arena[ieta][ix+fac_x][iy].Wmunu[0][5];
                    cube[1][1][1] = arena[ieta][ix+fac_x][iy+fac_y].Wmunu[0][5];
                    Wxy_center = 
                        util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);

                    // shear viscous tensor W^{x \eta}
                    cube[0][0][0] = arena[ieta][ix][iy].W_prev[6];
                    cube[0][0][1] = arena[ieta][ix][iy+fac_y].W_prev[6];
                    cube[0][1][0] = arena[ieta][ix+fac_x][iy].W_prev[6];
                    cube[0][1][1] = arena[ieta][ix+fac_x][iy+fac_y].W_prev[6];
                    cube[1][0][0] = arena[ieta][ix][iy].Wmunu[0][6];
                    cube[1][0][1] = arena[ieta][ix][iy+fac_y].Wmunu[0][6];
                    cube[1][1][0] = arena[ieta][ix+fac_x][iy].Wmunu[0][6];
                    cube[1][1][1] = arena[ieta][ix+fac_x][iy+fac_y].Wmunu[0][6];
                    Wxeta_center = 
                        util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);
                  
                    // shear viscous tensor W^{yy}
                    cube[0][0][0] = arena[ieta][ix][iy].W_prev[7];
                    cube[0][0][1] = arena[ieta][ix][iy+fac_y].W_prev[7];
                    cube[0][1][0] = arena[ieta][ix+fac_x][iy].W_prev[7];
                    cube[0][1][1] = arena[ieta][ix+fac_x][iy+fac_y].W_prev[7];
                    cube[1][0][0] = arena[ieta][ix][iy].Wmunu[0][7];
                    cube[1][0][1] = arena[ieta][ix][iy+fac_y].Wmunu[0][7];
                    cube[1][1][0] = arena[ieta][ix+fac_x][iy].Wmunu[0][7];
                    cube[1][1][1] = arena[ieta][ix+fac_x][iy+fac_y].Wmunu[0][7];
                    Wyy_center = 
                        util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);
                  
                    // shear viscous tensor W^{yeta}
                    cube[0][0][0] = arena[ieta][ix][iy].W_prev[8];
                    cube[0][0][1] = arena[ieta][ix][iy+fac_y].W_prev[8];
                    cube[0][1][0] = arena[ieta][ix+fac_x][iy].W_prev[8];
                    cube[0][1][1] = arena[ieta][ix+fac_x][iy+fac_y].W_prev[8];
                    cube[1][0][0] = arena[ieta][ix][iy].Wmunu[0][8];
                    cube[1][0][1] = arena[ieta][ix][iy+fac_y].Wmunu[0][8];
                    cube[1][1][0] = arena[ieta][ix+fac_x][iy].Wmunu[0][8];
                    cube[1][1][1] = arena[ieta][ix+fac_x][iy+fac_y].Wmunu[0][8];
                    Wyeta_center = 
                        util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);
                  
                    // shear viscous tensor W^{\eta\eta}
                    cube[0][0][0] = arena[ieta][ix][iy].W_prev[9];
                    cube[0][0][1] = arena[ieta][ix][iy+fac_y].W_prev[9];
                    cube[0][1][0] = arena[ieta][ix+fac_x][iy].W_prev[9];
                    cube[0][1][1] = arena[ieta][ix+fac_x][iy+fac_y].W_prev[9];
                    cube[1][0][0] = arena[ieta][ix][iy].Wmunu[0][9];
                    cube[1][0][1] = arena[ieta][ix][iy+fac_y].Wmunu[0][9];
                    cube[1][1][0] = arena[ieta][ix+fac_x][iy].Wmunu[0][9];
                    cube[1][1][1] = arena[ieta][ix+fac_x][iy+fac_y].Wmunu[0][9];
                    Wetaeta_center = 
                        util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);

                    // regulate Wmunu according to transversality and traceless
                    double **Wmunu_input = new double* [4];
                    double **Wmunu_regulated = new double* [4];
                    for(int i = 0; i < 4; i++)
                    {
                        Wmunu_input[i] = new double [4];
                        Wmunu_regulated[i] = new double [4];
                        for(int j = 0; j < 4; j++)
                            Wmunu_regulated[i][j] = 0.0;
                    }
                    Wmunu_input[0][0] = Wtautau_center;
                    Wmunu_input[0][1] = Wmunu_input[1][0] = Wtaux_center;
                    Wmunu_input[0][2] = Wmunu_input[2][0] = Wtauy_center;
                    Wmunu_input[0][3] = Wmunu_input[3][0] = Wtaueta_center;
                    Wmunu_input[1][1] = Wxx_center;
                    Wmunu_input[1][2] = Wmunu_input[2][1] = Wxy_center;
                    Wmunu_input[1][3] = Wmunu_input[3][1] = Wxeta_center;
                    Wmunu_input[2][2] = Wyy_center;
                    Wmunu_input[2][3] = Wmunu_input[3][2] = Wyeta_center;
                    Wmunu_input[3][3] = Wetaeta_center;

                    regulate_Wmunu(u_flow, Wmunu_input, Wmunu_regulated);

                    Wtautau_center = Wmunu_regulated[0][0];
                    Wtaux_center = Wmunu_regulated[0][1];
                    Wtauy_center = Wmunu_regulated[0][2];
                    Wtaueta_center = Wmunu_regulated[0][3];
                    Wxx_center = Wmunu_regulated[1][1];
                    Wxy_center = Wmunu_regulated[1][2];
                    Wxeta_center = Wmunu_regulated[1][3];
                    Wyy_center = Wmunu_regulated[2][2];
                    Wyeta_center = Wmunu_regulated[2][3];
                    Wetaeta_center = Wmunu_regulated[3][3];
        
                    // clean up
                    delete [] u_flow;
                    for(int i = 0; i < 4; i++)
                    {
                       delete [] Wmunu_input[i];
                       delete [] Wmunu_regulated[i];
                    }
                    delete [] Wmunu_input;
                    delete [] Wmunu_regulated;

                    // 3-dimension interpolation done
                
                    TFO = eos->get_temperature(epsFO, rhob_center);
                    muB = eos->get_mu(epsFO, rhob_center);
                    if (TFO < 0)
                    {
                        cout << "TFO=" << TFO << "<0. ERROR. exiting." << endl;
                        exit(1);
                    }

                    pressure = eos->get_pressure(epsFO, rhob_center);
                    eps_plus_p_over_T_FO = (epsFO + pressure)/TFO;

                    // finally output results !!!!
                    s_file << scientific << setprecision(10) 
                           << tau_center << " " << x_center << " " 
                           << y_center << " " << eta_center << " " 
                           << FULLSU[0] << " " << FULLSU[1] << " " 
                           << FULLSU[2] << " " << FULLSU[3] << " " 
                           << utau_center << " " << ux_center << " " 
                           << uy_center << " " << ueta_center << " " 
                           << epsFO << " " << TFO << " " << muB << " " 
                           << eps_plus_p_over_T_FO << " " 
                           << Wtautau_center << " " << Wtaux_center << " " 
                           << Wtauy_center << " " << Wtaueta_center << " " 
                           << Wxx_center << " " << Wxy_center << " " 
                           << Wxeta_center << " " 
                           << Wyy_center << " " << Wyeta_center << " " 
                           << Wetaeta_center << " " ;
                    if(DATA->turn_on_bulk)   // 27th column
                        s_file << pi_b_center << " " ;
                    if(DATA->turn_on_rhob)   // 28th column
                        s_file << rhob_center << " " ;
                    if(DATA->turn_on_diff)   // 29-32th column
                        s_file << qtau_center << " " << qx_center << " " 
                               << qy_center << " " << qeta_center << " " ;
                    s_file << endl;
                }
            }
        }

        s_file.close();
        // clean up
        for(int i = 0; i < 2; i++)
        {
            for(int j = 0; j < 2; j++)
                delete [] cube[i][j];
            delete [] cube[i];
            delete [] x_fraction[i];
        }
        delete [] cube;
        delete [] x_fraction;
        delete [] lattice_spacing_ptr;
        delete cornelius_ptr;
      
        // judge whether the entire fireball is freeze-out
        all_frozen[i_freezesurf] = 0;
        if (intersections == 0)
            all_frozen[i_freezesurf] = 1;
    }
   
    int all_frozen_flag = 1;
    for (int ii = 0; ii < n_freeze_surf; ii++)
        all_frozen_flag *= all_frozen[ii];
    if (all_frozen_flag == 1)
        cout << "All cells frozen out. Exiting." << endl;

    delete [] all_frozen;
    return(all_frozen_flag);
}

void Evolve::regulate_qmu(double *u, double *q, double *q_regulated) {
    double u_dot_q = - u[0]*q[0] + u[1]*q[1] + u[2]*q[2] + u[3]*q[3];
    for (int i = 0; i < 4; i++) {
          q_regulated[i] = q[i] + u[i]*u_dot_q;
    }
    return;
}
    
void Evolve::regulate_Wmunu(double* u, double** Wmunu,
                            double** Wmunu_regulated) {
    double gmunu[4][4] = {
        {-1, 0, 0, 0},
        { 0, 1, 0, 0},
        { 0, 0, 1, 0},
        { 0, 0, 0, 1}
    };
    double u_dot_pi[4];
    double u_mu[4];
    for (int i = 0; i < 4; i++) {
        u_dot_pi[i] = (- u[0]*Wmunu[0][i] + u[1]*Wmunu[1][i] 
                       + u[2]*Wmunu[2][i] + u[3]*Wmunu[3][i]);
        u_mu[i] = gmunu[i][i]*u[i];
    }
    double tr_pi = - Wmunu[0][0] + Wmunu[1][1] + Wmunu[2][2] + Wmunu[3][3];
    double u_dot_pi_dot_u = 0.0;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            u_dot_pi_dot_u += u_mu[i]*Wmunu[i][j]*u_mu[j];
        }
    }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            Wmunu_regulated[i][j] = (
                Wmunu[i][j] + u[i]*u_dot_pi[j] + u[j]*u_dot_pi[i] 
                + u[i]*u[j]*u_dot_pi_dot_u 
                - 1./3.*(gmunu[i][j] + u[i]*u[j])*(tr_pi + u_dot_pi_dot_u));
        }
    }
}

void Evolve::initialize_freezeout_surface_info() {
    int freeze_eps_flag = DATA_ptr->freeze_eps_flag;
    if (freeze_eps_flag == 0) {
        // constant spacing the energy density
        n_freeze_surf = DATA_ptr->N_freeze_out;
        double freeze_max_ed = DATA_ptr->eps_freeze_max;
        double freeze_min_ed = DATA_ptr->eps_freeze_min;
        double d_epsFO = ((freeze_max_ed - freeze_min_ed)
                          /(n_freeze_surf - 1 + 1e-15));
        for(int isurf = 0; isurf < n_freeze_surf; isurf++)
        {
            double temp_epsFO = freeze_min_ed + isurf*d_epsFO;
            epsFO_list.push_back(temp_epsFO);
        }
    } else if(freeze_eps_flag == 1) {
        // read in from a file
        //string eps_freeze_list_filename = DATA_ptr->freeze_list_filename;
        string eps_freeze_list_filename = "";
        cout << "read in freeze out surface information from " 
             << eps_freeze_list_filename << endl;
        ifstream freeze_list_file(eps_freeze_list_filename.c_str());
        if (!freeze_list_file) {
            cout << "Evolve::initialize_freezeout_surface_info: "
                 << "can not open freeze-out list file: " 
                 << eps_freeze_list_filename << endl;
            exit(1);
        }
        int temp_n_surf = 0;
        string dummy;
        double temp_epsFO, dummyd;
        getline(freeze_list_file, dummy);  // get rid of the comment
        while(1) {
            freeze_list_file >> temp_epsFO >> dummyd >> dummyd 
                             >> dummyd >> dummyd >> dummyd >> dummyd;  
            if (!freeze_list_file.eof()) {    
                epsFO_list.push_back(temp_epsFO);    
                temp_n_surf++;   
            } else {
                break;
            }
        }
        freeze_list_file.close();
        n_freeze_surf = temp_n_surf;
        cout << "totally " << n_freeze_surf 
             << " freeze-out surface will be generated ..." << endl;
    } else {
        cout << "Evolve::initialize_freezeout_surface_info: "
             << "unrecoginze freeze_eps_flag = " << freeze_eps_flag << endl;
        exit(1);
    }
}

double Evolve::energy_gubser(double tau, double x, double y) {

    const double qparam = GUBSER_Q;
    const double xperp = sqrt(x*x+y*y);

    return (4*pow(2, 0.66666)*pow(qparam, 2.666666))
            /(pow(tau,1.333333)*pow(1 + pow(qparam,4)*pow(pow(tau,2) - pow(xperp,2),2) + 2*pow(qparam,2)*(pow(tau,2) + pow(xperp,2)),
                                       1.33333333));
}

void Evolve::flow_gubser(double tau, double x, double y, double * utau, double * ux, double * uy) {

        const double qparam=GUBSER_Q;
        const double xperp=sqrt(x*x+y*y);

        *utau=(1 + pow(qparam,2)*(pow(tau,2) + pow(xperp,2)))/ sqrt(1 + pow(qparam,4)*pow(pow(tau,2) - pow(xperp,2),2) + 2*pow(qparam,2)*(pow(tau,2) + pow(xperp,2)));
        *ux=(qparam*x)/sqrt(1 + pow(-1 + pow(qparam,2)*(tau - xperp)*(tau + xperp),2)/(4.*pow(qparam,2)*pow(tau,2)));
        *uy=(qparam*y)/sqrt(1 + pow(-1 + pow(qparam,2)*(tau - xperp)*(tau + xperp),2)/(4.*pow(qparam,2)*pow(tau,2)));

}
