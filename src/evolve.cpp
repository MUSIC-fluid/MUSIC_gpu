// Copyright 2012 Bjoern Schenke, Sangyong Jeon, and Charles Gale
#include <omp.h>
#include "./evolve.h"
#include "./util.h"
#include "./data.h"
#include "./eos.h"
#include "./advance.h"
#include "./cornelius.h"
#include "./field.h"

using namespace std;

Evolve::Evolve(EOS *eosIn, InitData *DATA_in) {
    eos = eosIn;
    grid_info = new Grid_info(DATA_in, eosIn);
    util = new Util;
    u_derivative = new U_derivative(eosIn, DATA_in);
    advance = new Advance(eosIn, DATA_in);
   
    DATA_ptr = DATA_in;
    rk_order = DATA_in->rk_order;
    grid_nx = DATA_in->nx;
    grid_ny = DATA_in->ny;
    grid_neta = DATA_in->neta;
  
    if (DATA_ptr->freezeOutMethod == 4) {
        initialize_freezeout_surface_info();
    }
}

// destructor
Evolve::~Evolve() {
    delete grid_info;
    delete util;
    delete advance;
    delete u_derivative;
}

void Evolve::clean_up_hydro_fields(Field *hydro_fields) {
    delete[] hydro_fields->e_rk0;
    delete[] hydro_fields->e_rk1;
    delete[] hydro_fields->e_prev;
    delete[] hydro_fields->rhob_rk0;
    delete[] hydro_fields->rhob_rk1;
    delete[] hydro_fields->rhob_prev;
    for (int i = 0; i < 4; i++) {
        delete[] hydro_fields->u_rk0[i];
        delete[] hydro_fields->u_rk1[i];
        delete[] hydro_fields->u_prev[i];
    }
    for (int i = 0; i < 20; i++) {
        delete[] hydro_fields->dUsup[i];
    }
    for (int i = 0; i < 14; i++) {
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
                int idx = get_indx(ieta, ix, iy);
                e_local = energy_gubser(tau, x_local, y_local);
                flow_gubser(tau, x_local, y_local, &utau_local, &ux_local,
                            &uy_local);
                hydro_fields->e_rk0[idx] = e_local;
                hydro_fields->e_rk1[idx] = e_local;
                hydro_fields->e_prev[idx] = e_local;
                hydro_fields->rhob_rk0[idx] = 0.0;
                hydro_fields->rhob_rk1[idx] = 0.0;
                hydro_fields->rhob_prev[idx] = 0.0;
                hydro_fields->u_rk0[0][idx] = utau_local;
                hydro_fields->u_rk0[1][idx] = ux_local;
                hydro_fields->u_rk0[2][idx] = uy_local;
                hydro_fields->u_rk0[3][idx] = 0.0;
                hydro_fields->u_rk1[0][idx] = utau_local;
                hydro_fields->u_rk1[1][idx] = ux_local;
                hydro_fields->u_rk1[2][idx] = uy_local;
                hydro_fields->u_rk1[3][idx] = 0.0;
                hydro_fields->u_prev[0][idx] = utau_local;
                hydro_fields->u_prev[1][idx] = ux_local;
                hydro_fields->u_prev[2][idx] = uy_local;
                hydro_fields->u_prev[3][idx] = 0.0;
                for (int ii = 0; ii < 20; ii++) {
                    hydro_fields->dUsup[ii][idx] = 0.0;
                }
                for (int ii = 0; ii < 14; ii++) {
                    hydro_fields->Wmunu_rk0[ii][idx] = 0.0;
                    hydro_fields->Wmunu_rk1[ii][idx] = 0.0;
                    hydro_fields->Wmunu_prev[ii][idx] = 0.0;
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
                int idx = get_indx(ieta, ix, iy);
                e_local = energy_gubser(tau, x_local, y_local);
                flow_gubser(tau, x_local, y_local, &utau_local, &ux_local,
                            &uy_local);
                e_diff += fabs(hydro_fields->e_rk0[idx] - e_local);
                e_total += fabs(e_local);
                ux_diff += fabs(hydro_fields->u_rk0[1][idx] - ux_local);
                ux_total += fabs(ux_local);
                uy_diff += fabs(hydro_fields->u_rk0[2][idx] - uy_local);
                uy_total += fabs(uy_local);
            }
        }
    }
    cout << "e_diff: " << e_diff/e_total << ", ux_diff: " << ux_diff/ux_total
         << ", uy_diff: " << uy_diff/uy_total << endl;
}

// master control function for hydrodynamic evolution
int Evolve::EvolveIt(InitData *DATA, Field *hydro_fields) {
    //initial_field_with_ideal_Gubser(DATA->tau0, hydro_fields);
    //check_field_with_ideal_Gubser(DATA->tau0, hydro_fields);
    // first pass some control parameters
    int facTau = DATA->facTau;
    //int Nskip_timestep = DATA->output_evolution_every_N_timesteps;
    //int outputEvo_flag = DATA->outputEvolutionData;
    //int output_movie_flag = DATA->output_movie_flag;
    int freezeout_flag = DATA->doFreezeOut;
    int freezeout_lowtemp_flag = DATA->doFreezeOut_lowtemp;
    int freezeout_method = DATA->freezeOutMethod;
    int boost_invariant_flag = DATA->boost_invariant;

    // Output information about the hydro parameters 
    // in the format of a C header file
    //if (DATA->output_hydro_params_header || outputEvo_flag == 1)
    //    grid_info->Output_hydro_information_header(DATA);

    // main loop starts ...
    DATA->delta_tau= DELTA_TAU;
    DATA->delta_x = DELTA_X;
    DATA->delta_y= DELTA_Y;
    DATA->delta_eta= DELTA_ETA;

    int itmax = static_cast<int>(DATA->tau_size/DATA->delta_tau);
    double tau0 = DATA->tau0;
    double dt = DELTA_TAU;
    DATA->delta_tau = DELTA_TAU;
    DATA->delta_x = DELTA_X;
    DATA->delta_y = DELTA_Y;
    DATA->delta_eta = DELTA_ETA;
    double tau;
    int it_start = 0;
    cout << "Pre data copy" << endl;
    #pragma acc data copyin (hydro_fields[0:1],\
                         hydro_fields->e_rk0[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA],\
                         hydro_fields->e_prev[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA],\
                         hydro_fields->rhob_rk0[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA],\
                         hydro_fields->e_rk1[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA],\
                         hydro_fields->rhob_rk1[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA],\
                         hydro_fields->rhob_prev[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA], \
                         hydro_fields->u_rk0[0:4][0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA], \
                         hydro_fields->u_rk1[0:4][0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA], \
                         hydro_fields->u_prev[0:4][0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA], \
                         hydro_fields->dUsup[0:20][0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA], \
                         hydro_fields->Wmunu_rk0[0:14][0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA], \
                         hydro_fields->Wmunu_rk1[0:14][0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA], \
                         hydro_fields->Wmunu_prev[0:14][0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA], \
                         hydro_fields->pi_b_rk0[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA], \
                         hydro_fields->pi_b_rk1[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA], \
                         hydro_fields->pi_b_prev[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA])
    {
    cout << "Post data copy" << endl;
    
    for (int it = 0; it <= itmax; it++) {
        tau = tau0 + dt*it;
        // store initial conditions
        if (it == it_start) {
            store_previous_step_for_freezeout(hydro_fields);
        }

        if (DATA->Initial_profile == 0) {
            if (fabs(tau - 1.0) < 1e-8) {
                grid_info->Gubser_flow_check_file(hydro_fields, tau);
            }
            if (fabs(tau - 1.2) < 1e-8) {
                #pragma acc update host(hydro_fields->e_rk0[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA])
                #pragma acc update host(hydro_fields->u_rk0[0:4][0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA])
                #pragma acc update host(hydro_fields->Wmunu_rk0[0:14][0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA])
                grid_info->Gubser_flow_check_file(hydro_fields, tau);
            }
            if (fabs(tau - 1.5) < 1e-8) {
                #pragma acc update host(hydro_fields->e_rk0[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA])
                #pragma acc update host(hydro_fields->u_rk0[0:4][0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA])
                #pragma acc update host(hydro_fields->Wmunu_rk0[0:14][0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA])
                grid_info->Gubser_flow_check_file(hydro_fields, tau);
            }
            if (fabs(tau - 2.0) < 1e-8) {
                #pragma acc update host(hydro_fields->e_rk0[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA])
                #pragma acc update host(hydro_fields->u_rk0[0:4][0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA])
                #pragma acc update host(hydro_fields->Wmunu_rk0[0:14][0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA])
                grid_info->Gubser_flow_check_file(hydro_fields, tau);
            }
            if (fabs(tau - 3.0) < 1e-8) {
                #pragma acc update host(hydro_fields->e_rk0[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA])
                #pragma acc update host(hydro_fields->u_rk0[0:4][0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA])
                #pragma acc update host(hydro_fields->Wmunu_rk0[0:14][0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA])
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
        
        //determine freeze-out surface
        int frozen = 0;
        if (freezeout_flag == 1) {
            if (freezeout_lowtemp_flag == 1) {
                if (it == it_start) {
                    #pragma acc update host(hydro_fields->e_rk0[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA])
                    #pragma acc update host(hydro_fields->rhob_rk0[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA])
                    #pragma acc update host(hydro_fields->u_rk0[0:4][0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA])
                    #pragma acc update host(hydro_fields->Wmunu_rk0[0:14][0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA])
                    #pragma acc update host(hydro_fields->pi_b_rk0[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA])
                    frozen = FreezeOut_equal_tau_Surface(tau, DATA,
                                                         hydro_fields);
                }
            }
            // avoid freeze-out at the first time step
            if ((it - it_start)%facTau == 0 && it > it_start) {
                #pragma acc update host(hydro_fields->e_rk0[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA])
                #pragma acc update host(hydro_fields->rhob_rk0[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA])
                #pragma acc update host(hydro_fields->u_rk0[0:4][0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA])
                #pragma acc update host(hydro_fields->Wmunu_rk0[0:14][0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA])
                #pragma acc update host(hydro_fields->pi_b_rk0[0:(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1)*GRID_SIZE_ETA])
                if (freezeout_method == 4) {
                    if (boost_invariant_flag == 0) {
                        frozen = FindFreezeOutSurface_Cornelius(tau, DATA,
                                                                hydro_fields);
                    } else {
                        frozen = FindFreezeOutSurface_boostinvariant_Cornelius(
                                                     tau, DATA, hydro_fields);
                    }
               }
               store_previous_step_for_freezeout(hydro_fields);
            }
        }/* do freeze-out determination */
    
        fprintf(stdout, "Done time step %d/%d. tau = %6.3f fm/c \n", 
                it, itmax, tau);
        if (frozen) break;
    }/* it */ 
    }

    // clean up
    clean_up_hydro_fields(hydro_fields);

    return(1); /* successful */
}/* Evolve */


// update hydro fields information for freeze-out
void Evolve::store_previous_step_for_freezeout(Field *hydro_fields) {
    for (int ieta = 0; ieta < GRID_SIZE_ETA; ieta++) {
        for (int ix = 0; ix <= GRID_SIZE_X; ix++) {
            for (int iy = 0; iy <= GRID_SIZE_Y; iy++) {
                int idx = get_indx(ieta, ix, iy);
                hydro_fields->e_prev[idx] = hydro_fields->e_rk0[idx];
                hydro_fields->rhob_prev[idx] = hydro_fields->rhob_rk0[idx];
                for (int ii = 0; ii < 4; ii++) {
                    hydro_fields->u_prev[ii][idx] = (
                                            hydro_fields->u_rk0[ii][idx]);
                }
                for (int ii = 0; ii < 14; ii++) {
                    hydro_fields->Wmunu_prev[ii][idx] = (
                                            hydro_fields->Wmunu_rk0[ii][idx]);
                }
                hydro_fields->pi_b_prev[idx] = hydro_fields->pi_b_rk0[idx];
            }
        }
    }
}


//! This is a control function for Runge-Kutta evolution in tau
int Evolve::AdvanceRK(double tau, InitData *DATA, Field *hydro_fields) {
    int flag = 0;
    // loop over Runge-Kutta steps
    for (int rk_flag = 0; rk_flag < rk_order; rk_flag++) {
        flag = advance->AdvanceIt(tau, hydro_fields, rk_flag);
    }  /* loop over rk_flag */
    return(flag);
}

      
//! This function is find freeze-out fluid cells
int Evolve::FindFreezeOutSurface_Cornelius(double tau, InitData *DATA,
                                           Field *hydro_fields) {
    // output hyper-surfaces for Cooper-Frye
    int *all_frozen = new int [n_freeze_surf];
    for(int i = 0; i < n_freeze_surf; i++)
        all_frozen[i] = 0;
      
    int neta = GRID_SIZE_ETA;
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
                            tau, DATA, ieta, hydro_fields, thread_id, epsFO);
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
                                              int ieta, Field *hydro_fields,
                                              int thread_id, double epsFO) {
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

    int facTau = DATA->facTau;   // step to skip in tau direction
    int fac_x = DATA->fac_x;
    int fac_y = DATA->fac_y;
    int fac_eta = 1;
    
    double DTAU = facTau*DELTA_TAU;
    double DX = fac_x*DELTA_X;
    double DY = fac_y*DELTA_Y;
    double DETA = fac_eta*DELTA_ETA;

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

    double eta = (ieta - static_cast<double>(GRID_SIZE_ETA)/2.0)*DELTA_ETA;
    for (int ix = 0; ix <= GRID_SIZE_X - fac_x; ix += fac_x) {
        double x = (ix - static_cast<double>(GRID_SIZE_X + 1)/2.0)*DELTA_X; 
        for (int iy = 0; iy <= GRID_SIZE_Y - fac_y; iy += fac_y) {
            double y = (iy - static_cast<double>(GRID_SIZE_Y + 1)/2.0)*DELTA_Y;

            // make sure the epsilon value is never exactly 
            // the same as epsFO...
            int idx = get_indx(ieta, ix, iy);

            if (hydro_fields->e_rk0[idx] == epsFO) {
                hydro_fields->e_rk0[idx] += 0.000001;
            }
            if (hydro_fields->e_prev[idx] == epsFO) {
                hydro_fields->e_prev[idx] += 0.000001;
            }
            double e1000 = hydro_fields->e_rk0[idx];
            double e0000 = hydro_fields->e_prev[idx];
            idx = get_indx(ieta, ix + fac_x, iy);
            if (hydro_fields->e_rk0[idx] == epsFO) {
                hydro_fields->e_rk0[idx] += 0.000001;
            }
            if (hydro_fields->e_prev[idx] == epsFO) {
                hydro_fields->e_prev[idx] += 0.000001;
            }
            double e1100 = hydro_fields->e_rk0[idx];
            double e0100 = hydro_fields->e_prev[idx];
            idx = get_indx(ieta, ix, iy + fac_y);
            if (hydro_fields->e_rk0[idx] == epsFO) {
                hydro_fields->e_rk0[idx] += 0.000001;
            }
            if (hydro_fields->e_prev[idx] == epsFO) {
                hydro_fields->e_prev[idx] += 0.000001;
            }
            double e1010 = hydro_fields->e_rk0[idx];
            double e0010 = hydro_fields->e_prev[idx];
            idx = get_indx(ieta + fac_eta, ix, iy);
            if (hydro_fields->e_rk0[idx] == epsFO) {
                hydro_fields->e_rk0[idx] += 0.000001;
            }
            if (hydro_fields->e_prev[idx] == epsFO) {
                hydro_fields->e_prev[idx] += 0.000001;
            }
            double e1001 = hydro_fields->e_rk0[idx];
            double e0001 = hydro_fields->e_prev[idx];
            idx = get_indx(ieta, ix + fac_x, iy + fac_y);
            if (hydro_fields->e_rk0[idx] == epsFO) {
                hydro_fields->e_rk0[idx] += 0.000001;
            }
            if (hydro_fields->e_prev[idx] == epsFO) {
                hydro_fields->e_prev[idx] += 0.000001;
            }
            double e1110 = hydro_fields->e_rk0[idx];
            double e0110 = hydro_fields->e_prev[idx];
            idx = get_indx(ieta + fac_eta, ix + fac_x, iy);
            if (hydro_fields->e_rk0[idx] == epsFO) {
                hydro_fields->e_rk0[idx] += 0.000001;
            }
            if (hydro_fields->e_prev[idx] == epsFO) {
                hydro_fields->e_prev[idx] += 0.000001;
            }
            double e1101 = hydro_fields->e_rk0[idx];
            double e0101 = hydro_fields->e_prev[idx];
            idx = get_indx(ieta + fac_eta, ix, iy + fac_y);
            if (hydro_fields->e_rk0[idx] == epsFO) {
                hydro_fields->e_rk0[idx] += 0.000001;
            }
            if (hydro_fields->e_prev[idx] == epsFO) {
                hydro_fields->e_prev[idx] += 0.000001;
            }
            double e1011 = hydro_fields->e_rk0[idx];
            double e0011 = hydro_fields->e_prev[idx];
            idx = get_indx(ieta + fac_eta, ix + fac_x, iy + fac_y);
            if (hydro_fields->e_rk0[idx] == epsFO) {
                hydro_fields->e_rk0[idx] += 0.000001;
            }
            if (hydro_fields->e_prev[idx] == epsFO) {
                hydro_fields->e_prev[idx] += 0.000001;
            }
            double e1111 = hydro_fields->e_rk0[idx];
            double e0111 = hydro_fields->e_prev[idx];

            // judge intersection (from Bjoern)
            intersect = 1;
            if ((e1111 - epsFO)*(e0000 - epsFO) > 0.)
                if ((e1100 - epsFO)*(e0011 - epsFO) > 0.)
                    if ((e1010 - epsFO)*(e0101 - epsFO) > 0.)
                        if ((e1001 - epsFO)*(e0110 - epsFO) > 0.)
                            if ((e1110 - epsFO)*(e0001 - epsFO) > 0.)
                                if ((e1101 - epsFO)*(e0010 - epsFO) > 0.)
                                    if ((e1011-epsFO)*(e0100 - epsFO) > 0.)
                                        if ((e1000 - epsFO)
                                            *(e0111 - epsFO) > 0.)
                                            intersect = 0;

            if (intersect==0) {
                continue;
            }

            // if intersect, prepare for the hyper-cube
            intersections++;
            prepare_freeze_out_cube(cube,
                                    hydro_fields->e_prev, hydro_fields->e_rk0,
                                    ieta, ix, iy, fac_eta, fac_x, fac_y);
    
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
                prepare_freeze_out_cube(cube, hydro_fields->u_prev[1],
                                        hydro_fields->u_rk0[1],
                                        ieta, ix, iy, fac_eta, fac_x, fac_y);
                ux_center = 
                    util->four_dimension_linear_interpolation(
                                lattice_spacing_ptr, x_fraction, cube);

                // flow velocity u^y
                prepare_freeze_out_cube(cube, hydro_fields->u_prev[2],
                                        hydro_fields->u_rk0[2],
                                        ieta, ix, iy, fac_eta, fac_x, fac_y);
                uy_center = 
                    util->four_dimension_linear_interpolation(
                                lattice_spacing_ptr, x_fraction, cube);

                // flow velocity u^eta
                prepare_freeze_out_cube(cube, hydro_fields->u_prev[3],
                                        hydro_fields->u_rk0[3],
                                        ieta, ix, iy, fac_eta, fac_x, fac_y);
                ueta_center = 
                    util->four_dimension_linear_interpolation(
                                lattice_spacing_ptr, x_fraction, cube);

                // reconstruct u^tau from u^i
                utau_center = sqrt(1. + ux_center*ux_center 
                                   + uy_center*uy_center 
                                   + ueta_center*ueta_center);

                // baryon density rho_b
                prepare_freeze_out_cube(cube, hydro_fields->rhob_prev,
                                        hydro_fields->rhob_rk0,
                                        ieta, ix, iy, fac_eta, fac_x, fac_y);
                rhob_center = 
                    util->four_dimension_linear_interpolation(
                                lattice_spacing_ptr, x_fraction, cube);
          
                // baryon diffusion current q^tau
                prepare_freeze_out_cube(cube, hydro_fields->Wmunu_prev[10],
                                        hydro_fields->Wmunu_rk0[10],
                                        ieta, ix, iy, fac_eta, fac_x, fac_y);
                qtau_center = 
                    util->four_dimension_linear_interpolation(
                                lattice_spacing_ptr, x_fraction, cube);
          
                // baryon diffusion current q^x
                prepare_freeze_out_cube(cube, hydro_fields->Wmunu_prev[11],
                                        hydro_fields->Wmunu_rk0[11],
                                        ieta, ix, iy, fac_eta, fac_x, fac_y);
                qx_center = 
                    util->four_dimension_linear_interpolation(
                                lattice_spacing_ptr, x_fraction, cube);

                // baryon diffusion current q^y
                prepare_freeze_out_cube(cube, hydro_fields->Wmunu_prev[12],
                                        hydro_fields->Wmunu_rk0[12],
                                        ieta, ix, iy, fac_eta, fac_x, fac_y);
                qy_center = 
                    util->four_dimension_linear_interpolation(
                            lattice_spacing_ptr, x_fraction, cube);
          
                // baryon diffusion current q^eta
                prepare_freeze_out_cube(cube, hydro_fields->Wmunu_prev[13],
                                        hydro_fields->Wmunu_rk0[13],
                                        ieta, ix, iy, fac_eta, fac_x, fac_y);
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
                prepare_freeze_out_cube(cube, hydro_fields->pi_b_prev,
                                        hydro_fields->pi_b_rk0,
                                        ieta, ix, iy, fac_eta, fac_x, fac_y);
                pi_b_center = 
                    util->four_dimension_linear_interpolation(
                                lattice_spacing_ptr, x_fraction, cube);

                // shear viscous tensor W^\tau\tau
                prepare_freeze_out_cube(cube, hydro_fields->Wmunu_prev[0],
                                        hydro_fields->Wmunu_rk0[0],
                                        ieta, ix, iy, fac_eta, fac_x, fac_y);
                Wtautau_center = 
                    util->four_dimension_linear_interpolation(
                                lattice_spacing_ptr, x_fraction, cube);
      
                // shear viscous tensor W^{\tau x}
                prepare_freeze_out_cube(cube, hydro_fields->Wmunu_prev[1],
                                        hydro_fields->Wmunu_rk0[1],
                                        ieta, ix, iy, fac_eta, fac_x, fac_y);
                Wtaux_center = 
                    util->four_dimension_linear_interpolation(
                                lattice_spacing_ptr, x_fraction, cube);

                // shear viscous tensor W^{\tau y}
                prepare_freeze_out_cube(cube, hydro_fields->Wmunu_prev[2],
                                        hydro_fields->Wmunu_rk0[2],
                                        ieta, ix, iy, fac_eta, fac_x, fac_y);
                Wtauy_center = 
                    util->four_dimension_linear_interpolation(
                                lattice_spacing_ptr, x_fraction, cube);
      
                // shear viscous tensor W^{\tau \eta}
                prepare_freeze_out_cube(cube, hydro_fields->Wmunu_prev[3],
                                        hydro_fields->Wmunu_rk0[3],
                                        ieta, ix, iy, fac_eta, fac_x, fac_y);
                Wtaueta_center = 
                    util->four_dimension_linear_interpolation(
                                lattice_spacing_ptr, x_fraction, cube);
      
                // shear viscous tensor W^{xx}
                prepare_freeze_out_cube(cube, hydro_fields->Wmunu_prev[4],
                                        hydro_fields->Wmunu_rk0[4],
                                        ieta, ix, iy, fac_eta, fac_x, fac_y);
                Wxx_center = 
                    util->four_dimension_linear_interpolation(
                                lattice_spacing_ptr, x_fraction, cube);

                // shear viscous tensor W^{xy}
                prepare_freeze_out_cube(cube, hydro_fields->Wmunu_prev[5],
                                        hydro_fields->Wmunu_rk0[5],
                                        ieta, ix, iy, fac_eta, fac_x, fac_y);
                Wxy_center = 
                    util->four_dimension_linear_interpolation(
                                lattice_spacing_ptr, x_fraction, cube);

                // shear viscous tensor W^{x\eta}
                prepare_freeze_out_cube(cube, hydro_fields->Wmunu_prev[6],
                                        hydro_fields->Wmunu_rk0[6],
                                        ieta, ix, iy, fac_eta, fac_x, fac_y);
                Wxeta_center = 
                    util->four_dimension_linear_interpolation(
                                lattice_spacing_ptr, x_fraction, cube);
      
                // shear viscous tensor W^{yy}
                prepare_freeze_out_cube(cube, hydro_fields->Wmunu_prev[7],
                                        hydro_fields->Wmunu_rk0[7],
                                        ieta, ix, iy, fac_eta, fac_x, fac_y);
                Wyy_center = 
                    util->four_dimension_linear_interpolation(
                                lattice_spacing_ptr, x_fraction, cube);
      
                // shear viscous tensor W^{y\eta}
                prepare_freeze_out_cube(cube, hydro_fields->Wmunu_prev[8],
                                        hydro_fields->Wmunu_rk0[8],
                                        ieta, ix, iy, fac_eta, fac_x, fac_y);
                Wyeta_center = 
                    util->four_dimension_linear_interpolation(
                                lattice_spacing_ptr, x_fraction, cube);
      
                // shear viscous tensor W^{\eta\eta}
                prepare_freeze_out_cube(cube, hydro_fields->Wmunu_prev[9],
                                        hydro_fields->Wmunu_rk0[9],
                                        ieta, ix, iy, fac_eta, fac_x, fac_y);
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
                if (DATA->turn_on_bulk) {
                    s_file << pi_b_center << " " ;
                }
                if (DATA->turn_on_rhob) {
                    s_file << rhob_center << " " ;
                }
                if (DATA->turn_on_diff) {
                    s_file << qtau_center << " " << qx_center << " " 
                           << qy_center << " " << qeta_center << " " ;
                }
                s_file << endl;
            }
        }
    }
    s_file.close();

    // clean up
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for(int k = 0; k < 2; k++) {
                delete [] cube[i][j][k];
            }
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


//! This function is a shell function to freeze-out fluid cells
//! outside the freeze-out energy density at the first time step
//! of the evolution
int Evolve::FreezeOut_equal_tau_Surface(double tau, InitData *DATA,
                                        Field *hydro_fields) {
    // this function freeze-out fluid cells between epsFO and epsFO_low
    // on an equal time hyper-surface at the first time step
    // this function will be trigged if freezeout_lowtemp_flag == 1
    int neta = GRID_SIZE_ETA;
    int fac_eta = 1;
   
    for (int i_freezesurf = 0; i_freezesurf < n_freeze_surf; i_freezesurf++) {
        double epsFO = epsFO_list[i_freezesurf]/hbarc;
        int ieta;
        #pragma omp parallel private(ieta)
        {
            #pragma omp for
            for (ieta = 0; ieta < neta - fac_eta; ieta += fac_eta) {
                int thread_id = omp_get_thread_num();
                FreezeOut_equal_tau_Surface_XY(tau, DATA, ieta, hydro_fields,
                                               thread_id, epsFO);
            }
            #pragma omp barrier
        }
    }
    return(0);
}


//! This function freeze-outs fluid cells
//! outside the freeze-out energy density at the first time step
//! of the evolution in the transverse plane
void Evolve::FreezeOut_equal_tau_Surface_XY(double tau, InitData *DATA,
                                            int ieta, Field *hydro_fields,
                                            int thread_id, double epsFO) {

    double epsFO_low = 0.05/hbarc;        // 1/fm^4

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
    
    double DX = fac_x*DELTA_X;
    double DY = fac_y*DELTA_Y;
    double DETA = fac_eta*DELTA_ETA;

    double eta = (ieta - static_cast<double>(GRID_SIZE_ETA)/2.0)*DELTA_ETA;
    for (int ix = 0; ix <= GRID_SIZE_X - fac_x; ix += fac_x) {
        double x = (ix - static_cast<double>(GRID_SIZE_X)/2.0)*DELTA_X;
        for (int iy = 0; iy <= GRID_SIZE_Y - fac_y; iy += fac_y) {
            double y = (iy - static_cast<double>(GRID_SIZE_Y)/2.0)*DELTA_Y;
            int idx = get_indx(ieta, ix, iy);

            // judge intersection
            intersect = 0;
            if (hydro_fields->e_rk0[idx] < epsFO
                && hydro_fields->e_rk0[idx] > epsFO_low) {
                intersect = 1;
            }

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
            ux_center = hydro_fields->u_rk0[1][idx];
            uy_center = hydro_fields->u_rk0[2][idx];
            ueta_center = hydro_fields->u_rk0[3][idx];
            // reconstruct u^tau from u^i
            utau_center = sqrt(1. + ux_center*ux_center 
                               + uy_center*uy_center 
                               + ueta_center*ueta_center);

            // baryon density rho_b
            rhob_center = hydro_fields->rhob_rk0[idx];

            // baryon diffusion current
            qtau_center = hydro_fields->Wmunu_rk0[10][idx];
            qx_center = hydro_fields->Wmunu_rk0[11][idx];
            qy_center = hydro_fields->Wmunu_rk0[12][idx];
            qeta_center = hydro_fields->Wmunu_rk0[13][idx];
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
            pi_b_center = hydro_fields->pi_b_rk0[idx];

            // shear viscous tensor
            Wtautau_center = hydro_fields->Wmunu_rk0[0][idx];
            Wtaux_center = hydro_fields->Wmunu_rk0[1][idx];
            Wtauy_center = hydro_fields->Wmunu_rk0[2][idx];
            Wtaueta_center = hydro_fields->Wmunu_rk0[3][idx];
            Wxx_center = hydro_fields->Wmunu_rk0[4][idx];
            Wxy_center = hydro_fields->Wmunu_rk0[5][idx];
            Wxeta_center = hydro_fields->Wmunu_rk0[6][idx];
            Wyy_center = hydro_fields->Wmunu_rk0[7][idx];
            Wyeta_center = hydro_fields->Wmunu_rk0[8][idx];
            Wetaeta_center = hydro_fields->Wmunu_rk0[9][idx];

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
            double e_local = hydro_fields->e_rk0[idx];
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


//! This function freeze-out fluid cell for boost-invarinat simulations
int Evolve::FindFreezeOutSurface_boostinvariant_Cornelius(
                            double tau, InitData *DATA, Field *hydro_fields) {
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

        int facTau = DATA->facTau;   // step to skip in tau direction
        int fac_x = DATA->fac_x;
        int fac_y = DATA->fac_y;

        double DX = fac_x*DELTA_X;
        double DY = fac_y*DELTA_Y;
        double DETA = 1.0;
        double DTAU = facTau*DELTA_TAU;

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
  
        int ieta = 0;
        for (int ix=0; ix <= GRID_SIZE_X - fac_x; ix += fac_x) {
            double x = (ix - static_cast<double>(GRID_SIZE_X + 1)/2.0)*DELTA_X;
            for (int iy=0; iy <= GRID_SIZE_Y - fac_y; iy += fac_y) {
                double y = ((iy - static_cast<double>(GRID_SIZE_Y + 1)/2.0)
                            *DELTA_Y);
    
                // make sure the epsilon value is never 
                // exactly the same as epsFO...
                int idx = get_indx(ieta, ix, iy);
                if (hydro_fields->e_rk0[idx] == epsFO) {
                    hydro_fields->e_rk0[idx] += 0.000001;
                }
                if (hydro_fields->e_prev[idx] == epsFO) {
                    hydro_fields->e_prev[idx] += 0.000001;
                }
                double e100 = hydro_fields->e_rk0[idx];
                double e000 = hydro_fields->e_prev[idx];
                idx = get_indx(ieta, ix + fac_x, iy);
                if (hydro_fields->e_rk0[idx] == epsFO) {
                    hydro_fields->e_rk0[idx] += 0.000001;
                }
                if (hydro_fields->e_prev[idx] == epsFO) {
                    hydro_fields->e_prev[idx] += 0.000001;
                }
                double e110 = hydro_fields->e_rk0[idx];
                double e010 = hydro_fields->e_prev[idx];
                idx = get_indx(ieta, ix, iy + fac_y);
                if (hydro_fields->e_rk0[idx] == epsFO) {
                    hydro_fields->e_rk0[idx] += 0.000001;
                }
                if (hydro_fields->e_prev[idx] == epsFO) {
                    hydro_fields->e_prev[idx] += 0.000001;
                }
                double e101 = hydro_fields->e_rk0[idx];
                double e001 = hydro_fields->e_prev[idx];
                idx = get_indx(ieta, ix + fac_x, iy + fac_y);
                if (hydro_fields->e_rk0[idx] == epsFO) {
                    hydro_fields->e_rk0[idx] += 0.000001;
                }
                if (hydro_fields->e_prev[idx] == epsFO) {
                    hydro_fields->e_prev[idx] += 0.000001;
                }
                double e111 = hydro_fields->e_rk0[idx];
                double e011 = hydro_fields->e_prev[idx];
                
                // judge intersection (from Bjoern)
                intersect = 1;
                if ((e111 - epsFO)*(e000 - epsFO) > 0.)
                    if ((e110 - epsFO)*(e001 - epsFO) > 0.)
                        if ((e101 - epsFO)*(e010 - epsFO) > 0.)
                            if ((e100 - epsFO)*(e011 - epsFO) > 0.)
                                intersect = 0;
               
                if (intersect == 0) {
                    continue;
                }

                // if intersect, prepare for the hyper-cube
                intersections++;

                prepare_freeze_out_cube_boost_invariant(cube,
                        hydro_fields->e_prev, hydro_fields->e_rk0,
                        ix, iy, fac_x, fac_y);
           
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
                    prepare_freeze_out_cube_boost_invariant(cube,
                            hydro_fields->u_prev[0], hydro_fields->u_rk0[0],
                            ix, iy, fac_x, fac_y);
                    utau_center = util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);

                    // flow velocity u^x
                    prepare_freeze_out_cube_boost_invariant(cube,
                            hydro_fields->u_prev[1], hydro_fields->u_rk0[1],
                            ix, iy, fac_x, fac_y);
                    ux_center = util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);

                    // flow velocity u^y
                    prepare_freeze_out_cube_boost_invariant(cube,
                            hydro_fields->u_prev[2], hydro_fields->u_rk0[2],
                            ix, iy, fac_x, fac_y);
                    uy_center = util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);

                    // flow velocity u^eta
                    prepare_freeze_out_cube_boost_invariant(cube,
                            hydro_fields->u_prev[3], hydro_fields->u_rk0[3],
                            ix, iy, fac_x, fac_y);
                    ueta_center = util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);

                    // baryon density rho_b
                    prepare_freeze_out_cube_boost_invariant(cube,
                        hydro_fields->rhob_prev, hydro_fields->rhob_rk0,
                        ix, iy, fac_x, fac_y);
                    rhob_center = util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);

                    // bulk viscous pressure pi_b
                    prepare_freeze_out_cube_boost_invariant(cube,
                        hydro_fields->pi_b_prev, hydro_fields->pi_b_rk0,
                        ix, iy, fac_x, fac_y);
                    pi_b_center = util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);
               
                    // baryon diffusion current q^\tau
                    prepare_freeze_out_cube_boost_invariant(cube,
                                            hydro_fields->Wmunu_prev[10],
                                            hydro_fields->Wmunu_rk0[10],
                                            ix, iy, fac_x, fac_y);
                    qtau_center = util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);
               
                    // baryon diffusion current q^x
                    prepare_freeze_out_cube_boost_invariant(cube,
                                            hydro_fields->Wmunu_prev[11],
                                            hydro_fields->Wmunu_rk0[11],
                                            ix, iy, fac_x, fac_y);
                    qx_center = util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);
               
                    // baryon diffusion current q^y
                    prepare_freeze_out_cube_boost_invariant(cube,
                                            hydro_fields->Wmunu_prev[12],
                                            hydro_fields->Wmunu_rk0[12],
                                            ix, iy, fac_x, fac_y);
                    qy_center = util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);
               
                    // baryon diffusion current q^eta
                    prepare_freeze_out_cube_boost_invariant(cube,
                                            hydro_fields->Wmunu_prev[13],
                                            hydro_fields->Wmunu_rk0[13],
                                            ix, iy, fac_x, fac_y);
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
                    prepare_freeze_out_cube_boost_invariant(cube,
                                            hydro_fields->Wmunu_prev[0],
                                            hydro_fields->Wmunu_rk0[0],
                                            ix, iy, fac_x, fac_y);
                    Wtautau_center = 
                        util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);
                  
                    // shear viscous tensor W^{\tau x}
                    prepare_freeze_out_cube_boost_invariant(cube,
                                            hydro_fields->Wmunu_prev[1],
                                            hydro_fields->Wmunu_rk0[1],
                                            ix, iy, fac_x, fac_y);
                    Wtaux_center = 
                        util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);

                    // shear viscous tensor W^{\tau y}
                    prepare_freeze_out_cube_boost_invariant(cube,
                                            hydro_fields->Wmunu_prev[2],
                                            hydro_fields->Wmunu_rk0[2],
                                            ix, iy, fac_x, fac_y);
                    Wtauy_center = 
                        util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);
                  
                    // shear viscous tensor W^{\tau \eta}
                    prepare_freeze_out_cube_boost_invariant(cube,
                                            hydro_fields->Wmunu_prev[3],
                                            hydro_fields->Wmunu_rk0[3],
                                            ix, iy, fac_x, fac_y);
                    Wtaueta_center = 
                        util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);
                  
                    // shear viscous tensor W^{xx}
                    prepare_freeze_out_cube_boost_invariant(cube,
                                            hydro_fields->Wmunu_prev[4],
                                            hydro_fields->Wmunu_rk0[4],
                                            ix, iy, fac_x, fac_y);
                    Wxx_center = 
                        util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);

                    // shear viscous tensor W^{xy}
                    prepare_freeze_out_cube_boost_invariant(cube,
                                            hydro_fields->Wmunu_prev[5],
                                            hydro_fields->Wmunu_rk0[5],
                                            ix, iy, fac_x, fac_y);
                    Wxy_center = 
                        util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);

                    // shear viscous tensor W^{x \eta}
                    prepare_freeze_out_cube_boost_invariant(cube,
                                            hydro_fields->Wmunu_prev[6],
                                            hydro_fields->Wmunu_rk0[6],
                                            ix, iy, fac_x, fac_y);
                    Wxeta_center = 
                        util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);
                  
                    // shear viscous tensor W^{yy}
                    prepare_freeze_out_cube_boost_invariant(cube,
                                            hydro_fields->Wmunu_prev[7],
                                            hydro_fields->Wmunu_rk0[7],
                                            ix, iy, fac_x, fac_y);
                    Wyy_center = 
                        util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);
                  
                    // shear viscous tensor W^{yeta}
                    prepare_freeze_out_cube_boost_invariant(cube,
                                            hydro_fields->Wmunu_prev[8],
                                            hydro_fields->Wmunu_rk0[8],
                                            ix, iy, fac_x, fac_y);
                    Wyeta_center = 
                        util->three_dimension_linear_interpolation(
                                        lattice_spacing_ptr, x_fraction, cube);
                  
                    // shear viscous tensor W^{\eta\eta}
                    prepare_freeze_out_cube_boost_invariant(cube,
                                            hydro_fields->Wmunu_prev[9],
                                            hydro_fields->Wmunu_rk0[9],
                                            ix, iy, fac_x, fac_y);
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
        cout << "Boost_invariant: All cells are frozen out. Exiting." << endl;

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


//! This is a function to prepare the freeze-out cube
void Evolve::prepare_freeze_out_cube(double ****cube,
                                     double* data_prev, double* data_array,
                                     int ieta, int ix, int iy,
                                     int fac_eta, int fac_x, int fac_y) {
    int idx = get_indx(ieta, ix, iy);
    cube[0][0][0][0] = data_prev[idx];
    cube[1][0][0][0] = data_array[idx];
    idx = (iy + (ix + fac_x)*(GRID_SIZE_Y + 1)
               + ieta*(GRID_SIZE_Y + 1)*(GRID_SIZE_X + 1));
    cube[0][1][0][0] = data_prev[idx];
    cube[1][1][0][0] = data_array[idx];
    idx = (iy + fac_y + ix*(GRID_SIZE_Y + 1)
               + ieta*(GRID_SIZE_Y + 1)*(GRID_SIZE_X + 1));
    cube[0][0][1][0] = data_prev[idx];
    cube[1][0][1][0] = data_array[idx];
    idx = (iy + ix*(GRID_SIZE_Y + 1)
               + (ieta + fac_eta)*(GRID_SIZE_Y + 1)*(GRID_SIZE_X + 1));
    cube[0][0][0][1] = data_prev[idx];
    cube[1][0][0][1] = data_array[idx];
    idx = (iy + fac_y + (ix + fac_x)*(GRID_SIZE_Y + 1)
               + ieta*(GRID_SIZE_Y + 1)*(GRID_SIZE_X + 1));
    cube[0][1][1][0] = data_prev[idx];
    cube[1][1][1][0] = data_array[idx];
    idx = (iy + fac_y + ix*(GRID_SIZE_Y + 1)
               + (ieta + fac_eta)*(GRID_SIZE_Y + 1)*(GRID_SIZE_X + 1));
    cube[0][0][1][1] = data_prev[idx];
    cube[1][0][1][1] = data_array[idx];
    idx = (iy + (ix + fac_x)*(GRID_SIZE_Y + 1)
               + (ieta + fac_eta)*(GRID_SIZE_Y + 1)*(GRID_SIZE_X + 1));
    cube[0][1][0][1] = data_prev[idx];
    cube[1][1][0][1] = data_array[idx];
    idx = (iy + fac_y + (ix + fac_x)*(GRID_SIZE_Y + 1)
               + (ieta + fac_eta)*(GRID_SIZE_Y + 1)*(GRID_SIZE_X + 1));
    cube[0][1][1][1] = data_prev[idx];
    cube[1][1][1][1] = data_array[idx];
}


//! This is a function to prepare the freeze-out cube for boost-invariant
//! surface
void Evolve::prepare_freeze_out_cube_boost_invariant(
                    double ***cube, double* data_prev, double* data_array,
                    int ix, int iy, int fac_x, int fac_y) {
    int idx = get_indx(0, ix, iy);
    cube[0][0][0] = data_prev[idx];
    cube[1][0][0] = data_array[idx];
    idx = iy + (ix + fac_x)*(GRID_SIZE_Y + 1);
    cube[0][1][0] = data_prev[idx];
    cube[1][1][0] = data_array[idx];
    idx = iy + fac_y + ix*(GRID_SIZE_Y + 1);
    cube[0][0][1] = data_prev[idx];
    cube[1][0][1] = data_array[idx];
    idx = iy + fac_y + (ix + fac_x)*(GRID_SIZE_Y + 1);
    cube[0][1][1] = data_prev[idx];
    cube[1][1][1] = data_array[idx];
}
