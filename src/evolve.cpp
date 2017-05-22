// Copyright 2012 Bjoern Schenke, Sangyong Jeon, and Charles Gale
#include "./evolve.h"
#include "./util.h"
#include "./data.h"
#include "./grid.h"
#include "./eos.h"
#include "./advance.h"
#include "./cornelius.h"

using namespace std;

Evolve::Evolve(EOS *eosIn, InitData *DATA_in) {
    eos = eosIn;
    grid = new Grid;
    grid_info = new Grid_info(DATA_in, eosIn);
    util = new Util;
    advance = new Advance(eosIn, DATA_in);
    u_derivative = new U_derivative(eosIn, DATA_in);
   
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
    delete grid;
    delete grid_info;
    delete util;
    delete advance;
    delete u_derivative;
}

// master control function for hydrodynamic evolution
int Evolve::EvolveIt(InitData *DATA, Grid ***arena) {
    // first pass some control parameters
    facTau = DATA->facTau;
    int output_hydro_debug_flag = DATA->output_hydro_debug_info;
    int Nskip_timestep = DATA->output_evolution_every_N_timesteps;
    int outputEvo_flag = DATA->outputEvolutionData;
    int output_movie_flag = DATA->output_movie_flag;
    int freezeout_flag = DATA->doFreezeOut;
    int freezeout_lowtemp_flag = DATA->doFreezeOut_lowtemp;
    int freezeout_method = DATA->freezeOutMethod;
    int boost_invariant_flag = DATA->boost_invariant;

    // Output information about the hydro parameters 
    // in the format of a C header file
    if (DATA->output_hydro_params_header || outputEvo_flag == 1)
        grid_info->Output_hydro_information_header(DATA);

    // main loop starts ...
    int itmax = DATA->nt;
    double tau0 = DATA->tau0;
    double dt = DATA->delta_tau;

    weirdCases=0;
    SUM = 0.;
    warnings = 0;
    cells = 0;
      
    double tau;
    int it_start = 0;
    for (int it = 0; it <= itmax; it++) {
        tau = tau0 + dt*it;
        // store initial conditions
        if (it == it_start) {
            store_previous_step_for_freezeout(arena);
        }
        
        if (DATA->Initial_profile == 0) {
            if (fabs(tau - 1.0) < 1e-8) {
                grid_info->Gubser_flow_check_file(arena, tau);
            }
            if (fabs(tau - 1.2) < 1e-8) {
                grid_info->Gubser_flow_check_file(arena, tau);
            }
            if (fabs(tau - 1.5) < 1e-8) {
                grid_info->Gubser_flow_check_file(arena, tau);
            }
            if (fabs(tau - 2.0) < 1e-8) {
                grid_info->Gubser_flow_check_file(arena, tau);
            }
            if (fabs(tau - 3.0) < 1e-8) {
                grid_info->Gubser_flow_check_file(arena, tau);
            }
        }

        if (DATA->Initial_profile == 1) {
            if (fabs(tau - 1.0) < 1e-8) {
                grid_info->output_1p1D_check_file(arena, tau);
            }
            if (fabs(tau - 2.0) < 1e-8) {
                grid_info->output_1p1D_check_file(arena, tau);
            }
            if (fabs(tau - 5.0) < 1e-8) {
                grid_info->output_1p1D_check_file(arena, tau);
            }
            if (fabs(tau - 10.) < 1e-8) {
                grid_info->output_1p1D_check_file(arena, tau);
            }
            if (fabs(tau - 20.) < 1e-8) {
                grid_info->output_1p1D_check_file(arena, tau);
            }
        }

        if (it % Nskip_timestep == 0) {
            if (outputEvo_flag == 1) {
                grid_info->OutputEvolutionDataXYEta(arena, DATA, tau);
            } else if (outputEvo_flag == 2) {
                grid_info->OutputEvolutionDataXYEta_chun(arena, DATA, tau);
            }
            if (output_movie_flag == 1) {
                grid_info->output_evolution_for_movie(arena, tau);
            }
        }

        // grid_info->output_average_phase_diagram_trajectory(tau, -0.5, 0.5,
        //                                                    arena);
        // grid_info->output_average_phase_diagram_trajectory(tau, 0.5, 1.5,
        //                                                    arena);
        // grid_info->output_average_phase_diagram_trajectory(tau, -1.5, -0.5,
        //                                                    arena);
        // grid_info->output_average_phase_diagram_trajectory(tau, 1.5, 2.5,
        //                                                    arena);
        // grid_info->output_average_phase_diagram_trajectory(tau, -2.5, -1.5,
        //                                                    arena);

        // check energy conservation
        if (boost_invariant_flag == 0)
            grid_info->check_conservation_law(arena, DATA, tau);
        grid_info->get_maximum_energy_density(arena);

        /* execute rk steps */
        // all the evolution are at here !!!
        AdvanceRK(tau, DATA, arena);
        
        // output evolution information for debug
        if (output_hydro_debug_flag == 1) {
            if ((it%Nskip_timestep) == 0) {
                grid_info->check_velocity_shear_tensor(arena, tau);
            }
        }
        UpdateArena(arena);
   
        //determine freeze-out surface
        int frozen = 0;
        if (freezeout_flag == 1) {
            if (freezeout_lowtemp_flag == 1) {
                if (it == it_start) {
                    frozen = FreezeOut_equal_tau_Surface(tau, DATA, arena);
                }
            }
            // avoid freeze-out at the first time step
            if ((it - it_start)%facTau == 0 && it > it_start) {
               //if (freezeout_method == 1)
               //    FindFreezeOutSurface(tau, DATA, arena);
               //else if (freezeout_method == 2)
               //    FindFreezeOutSurface2(tau, DATA, arena);
               //else if (freezeout_method == 3)
               //    frozen = FindFreezeOutSurface3(tau, DATA, arena);
               if (freezeout_method == 4) {
                   if (boost_invariant_flag == 0) {
                       frozen = FindFreezeOutSurface_Cornelius(tau, DATA,
                                                               arena);
                   } else {
                       frozen = FindFreezeOutSurface_boostinvariant_Cornelius(
                                                            tau, DATA, arena);
                   }
               }
               store_previous_step_for_freezeout(arena);
            }
        }/* do freeze-out determination */
    
        fprintf(stdout, "Done time step %d/%d. tau = %6.3f fm/c \n", 
                it, itmax, tau);
        if (frozen) break;
    }/* it */ 

    // clean up
    for (int ieta=0; ieta < DATA->neta; ieta++) {
        for (int ix=0; ix <= DATA->nx; ix++) {
            for (int iy=0; iy <= DATA->ny; iy++) {
                util->cube_free(arena[ieta][ix][iy].TJb, rk_order+1, 5, 4);
                util->cube_free(arena[ieta][ix][iy].dUsup, 1, 5, 4);
                util->mtx_free(arena[ieta][ix][iy].sigma, 1, 10);
                util->mtx_free(arena[ieta][ix][iy].Wmunu, rk_order+1, 14);
                util->mtx_free(arena[ieta][ix][iy].prevWmunu, rk_order, 14);
                util->mtx_free(arena[ieta][ix][iy].u, rk_order+1, 4);
                util->mtx_free(arena[ieta][ix][iy].a, 1, 5);
                util->mtx_free(arena[ieta][ix][iy].prev_u, rk_order, 4);
                util->vector_free(arena[ieta][ix][iy].theta_u);
                util->vector_free(arena[ieta][ix][iy].pi_b);
                util->vector_free(arena[ieta][ix][iy].prev_pi_b);
                
                util->vector_free(arena[ieta][ix][iy].W_prev);
                delete[] arena[ieta][ix][iy].nbr_p_1;
                delete[] arena[ieta][ix][iy].nbr_p_2;
                delete[] arena[ieta][ix][iy].nbr_m_1;
                delete[] arena[ieta][ix][iy].nbr_m_2;
            }
        }
    }

    fprintf(stderr,"SUM=%f\n", SUM);
    return 1; /* successful */
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
int Evolve::UpdateArena(Grid ***arena) {
    int neta = grid_neta;
    int ieta;
    #pragma omp parallel private(ieta)
    {
        #pragma omp for
        for (ieta = 0; ieta < neta; ieta++) {
            //printf("Evolve::UpdateArena:");
            //printf("Thread %d executes loop iteraction %d\n",
            //       omp_get_thread_num(), ieta);
            UpdateArena_XY(ieta, arena);
        } /* ieta */
    }
    return 1;
}/* UpdateArena */

void Evolve::UpdateArena_XY(int ieta, Grid ***arena) {
    int nx = grid_nx;
    int ny = grid_ny;
    for (int ix = 0; ix <= nx; ix++) {
        for (int iy = 0; iy <= ny; iy++) {
            arena[ieta][ix][iy].prev_epsilon = arena[ieta][ix][iy].epsilon;
            arena[ieta][ix][iy].prev_rhob = arena[ieta][ix][iy].rhob;
            arena[ieta][ix][iy].p = arena[ieta][ix][iy].p_t;
            arena[ieta][ix][iy].epsilon = arena[ieta][ix][iy].epsilon_t;
            arena[ieta][ix][iy].rhob = arena[ieta][ix][iy].rhob_t;
     
            // previous pi_b is stored in prevPimunu
            arena[ieta][ix][iy].prev_pi_b[0] = arena[ieta][ix][iy].pi_b[0];
            arena[ieta][ix][iy].prev_pi_b[1] = arena[ieta][ix][iy].pi_b[1];
            arena[ieta][ix][iy].pi_b[0] = arena[ieta][ix][iy].pi_b[rk_order];
            for (int ii = 0; ii < 14; ii++) {
                for (int rkstep = 0; rkstep < 2; rkstep++) {
                    /* this was the previous value */
                    arena[ieta][ix][iy].prevWmunu[rkstep][ii] = (
                                    arena[ieta][ix][iy].Wmunu[rkstep][ii]); 
                }
                /* this is the new value */
                arena[ieta][ix][iy].Wmunu[0][ii] = (
                                arena[ieta][ix][iy].Wmunu[rk_order][ii]); 
            }
            for (int mu = 0; mu < 4; mu++) {
                /* this was the previous value */
                arena[ieta][ix][iy].prev_u[0][mu] = (
                                        arena[ieta][ix][iy].u[0][mu]); 
                arena[ieta][ix][iy].prev_u[1][mu] = (
                                        arena[ieta][ix][iy].u[1][mu]); 
                /* this is the new value */
                arena[ieta][ix][iy].u[0][mu] = (
                                arena[ieta][ix][iy].u[rk_order][mu]); 

                for (int alpha = 0; alpha < 5; alpha++) {
                    /* this is the new value */
                    arena[ieta][ix][iy].TJb[0][alpha][mu] = (
                        arena[ieta][ix][iy].TJb[rk_order][alpha][mu]);
                    
                }
            }/* mu, alpha */
        }
    }
}


int Evolve::AdvanceRK(double tau, InitData *DATA, Grid ***arena) {
    // control function for Runge-Kutta evolution in tau
    int flag = 0;
    // loop over Runge-Kutta steps
    for (int rk_flag = 0; rk_flag < rk_order; rk_flag++) {
        flag = u_derivative->MakedU(tau, DATA, arena, rk_flag);
        if (flag == 0) return 0;
        flag = advance->AdvanceIt(tau, DATA, arena, rk_flag);
    }  /* loop over rk_flag */
    return(flag);
}  /* AdvanceRK */
      

void Evolve::FindFreezeOutSurface2(double tau, InitData *DATA, Grid ***arena, int size, int rank)
{   
  ofstream t_file;
  t_file.open("tauf.dat" , ios::out | ios::app );
  ofstream t2_file;
  t2_file.open("taufx.dat" , ios::out | ios::app );
  ofstream t3_file;
  t3_file.open("taufy.dat" , ios::out | ios::app );

  stringstream strs_name;
  strs_name << "surface" << rank << ".dat";
  string s_name = strs_name.str();
  
  ofstream s_file;
  s_file.open(s_name.c_str() , ios::out | ios::app );

  int ix, iy, ieta, nx, ny, neta;
  double x, y, eta;
  double epsFO;
  double tauf, xf, yf, etaf;
  nx = grid_nx;
  ny = grid_ny;
  neta = grid_neta;
  double cube[16];
  double EK, EL, DEK, DEL, ADEK, ADEL, ELowerSum, EHigherSum;
  double cuts[32][4]; // a 4d hypercube has (2^(n-1)*n=32) edges.
  double VLower0, VHigher0, VLower1, VHigher1, VLower2, VHigher2, VLower3, VHigher3;
  double V0, V1, V2, V3, VD0, VD1, VD2, VD3;
  double VMID[4], AD[32][4], BD[4], DD[32][4], SU[32][4], FULLSU[4];
//   double CD[4];
  int NSurfaces;
  int iEdge[32];
  int prevEdge[32];
  int neighbors[32][6];
  int neighborsDone[32][6];
  int additionalNeighbors[32][6];
  int edge1[32];
  int edge2[32];
  int is, intersect, IE, JE, IS, i, j, k;
//   int KE, MINPTS, JMIN, IPTS, NSE, NSM, M, M2;
  double DX, DY, DETA, DTAU;
//   double APU, SIG;
  int countEdges, skip;
  int previousEdges[32], usedEdges1[32],usedEdges2[32],usedEdges3[32],usedEdges4[32], countAdditionalEdges;
  int fac,l, COUNTER, additionalEdges, temp, m, m2;
  int group[32][3];
  int tries, tries3;
//   int tries2;
  int shift;
//   int addCOUNTER;
//   int ISID[32];
  int countSingleEdges;
  int singleConnections[32][2];
  int singleConnectionsUsed[32];
  int numberConnectionIsUsed[32][32]; // should be 2 in the end
  double Wtautau, Wtaux, Wtauy, Wtaueta, Wxx, Wxy, Wxeta, Wyy, Wyeta, Wetaeta;
  double Wtautau1, Wtaux1, Wtauy1, Wtaueta1, Wxx1, Wxy1, Wxeta1, Wyy1, Wyeta1, Wetaeta1;
  double Wtautau2, Wtaux2, Wtauy2, Wtaueta2, Wxx2, Wxy2, Wxeta2, Wyy2, Wyeta2, Wetaeta2;
  double WX1, WX2, WX3, WX4, WY1, WY2;
  double rhob, utau, ux, uy, ueta, TFO, muB;
  double utauX1, utauX2, utauX3, utauX4, utauY1, utauY2, utau1, utau2;
  double rhobX1, rhobX2, rhobX3, rhobX4, rhobY1, rhobY2;
//   double rhob1, rhob2;
  double uxX1, uxX2, uxX3, uxX4, uxY1, uxY2, ux1, ux2;
  double uyX1, uyX2, uyX3, uyX4, uyY1, uyY2, uy1, uy2;
  double uetaX1, uetaX2, uetaX3, uetaX4, uetaY1, uetaY2, ueta1, ueta2;
  double xfrac, yfrac, etafrac, taufrac;
  // who is direct neighbor (1) or neighbor across the plane (2), more distant neighbor (3), (4)
  int intersections;
  int maxEta;
  double eps_plus_p_over_T_FO, P;
  int const IBIT[32][32] = 
    { {0,1,2,1,1,1,3,3,2,3,4,3,2,3,4,3,3,3,0,0,4,0,0,0,1,1,3,3,3,3,0,0},
      {1,0,1,2,3,1,1,3,3,2,3,4,3,2,3,4,3,3,0,0,0,4,0,0,3,1,1,3,0,3,3,0},
      {2,1,0,1,3,3,1,1,4,3,2,3,4,3,2,3,0,0,3,3,0,0,4,0,3,3,1,1,0,0,3,3},
      {1,2,1,0,1,3,3,1,3,4,3,2,3,4,3,2,3,0,0,3,0,4,0,0,1,3,3,1,3,0,0,3},
      {1,3,3,1,0,2,4,2,1,3,3,1,3,0,0,3,2,4,0,4,3,0,0,3,1,3,0,3,1,3,0,3},
      {1,1,3,3,2,0,2,4,1,1,3,3,3,3,0,0,4,2,4,0,3,3,0,0,3,1,3,0,3,1,3,0},
      {3,1,1,3,4,2,0,2,3,1,1,3,0,3,3,0,0,4,2,4,0,3,3,0,0,3,1,3,0,3,1,3},
      {3,3,1,1,2,4,2,0,3,3,1,1,0,0,3,3,4,0,4,2,0,0,3,3,3,0,3,1,3,0,3,1},
      {2,3,4,3,1,1,3,3,0,1,2,1,4,0,0,0,3,3,0,0,2,3,4,3,3,3,0,0,1,1,3,3},
      {3,2,3,4,3,1,1,3,1,0,1,2,0,4,0,0,0,3,3,0,3,2,3,4,0,3,3,0,3,1,1,3},
      {4,3,2,3,3,3,1,1,2,1,0,1,0,0,4,0,0,0,3,3,4,3,2,3,0,0,3,3,3,3,1,1},
      {3,4,3,2,1,3,3,1,1,2,1,0,0,0,0,4,3,0,0,3,3,4,3,2,3,0,0,3,1,3,3,1},
      {2,3,4,3,3,3,0,0,4,0,0,0,0,1,2,1,1,1,3,3,2,3,4,3,1,1,3,3,3,3,0,0},
      {3,2,3,4,0,3,3,0,0,4,0,0,1,0,1,2,3,1,1,3,3,2,3,4,3,1,1,3,0,3,3,0},
      {4,3,2,3,0,0,3,3,0,0,4,0,2,1,0,1,3,3,1,1,4,3,2,3,3,3,1,1,0,0,3,3},
      {3,4,3,2,3,0,0,3,0,0,0,4,1,2,1,0,1,3,3,1,3,4,3,2,1,3,3,1,3,0,0,3},
      {3,0,0,3,2,4,0,4,3,0,0,3,1,3,3,1,0,2,4,2,1,3,3,1,1,3,0,3,1,3,0,3},
      {3,3,0,0,4,2,4,0,3,3,0,0,1,1,3,3,2,0,2,4,1,1,3,3,3,1,3,0,3,1,3,0},
      {0,3,3,0,0,4,2,4,0,3,3,0,3,1,1,3,4,2,0,2,3,1,1,3,0,3,1,3,0,3,1,3},
      {0,0,3,3,4,0,4,2,0,0,3,3,3,3,1,1,2,4,2,0,3,3,1,1,3,0,3,1,3,0,3,1},
      {4,0,0,0,3,3,0,0,2,3,4,3,2,3,4,3,1,1,3,3,0,1,2,1,3,3,0,0,1,1,3,3},
      {0,4,0,0,0,3,3,0,3,2,3,4,3,2,3,4,3,1,1,3,1,0,1,2,0,3,3,0,3,1,1,3},
      {0,0,4,0,0,0,3,3,4,3,2,3,4,3,2,3,3,3,1,1,2,1,0,1,0,0,3,3,3,3,1,1},
      {0,0,0,4,3,0,0,3,3,4,3,2,3,4,3,2,1,3,3,1,1,2,1,0,3,0,0,3,1,3,3,1},
      {1,3,3,1,1,3,0,3,3,0,0,3,1,3,3,1,1,3,0,3,3,0,0,3,0,2,4,2,2,4,0,4},
      {1,1,3,3,3,1,3,0,3,3,0,0,1,1,3,3,3,1,3,0,3,3,0,0,2,0,2,4,4,2,4,0},
      {3,1,1,3,0,3,1,3,0,3,3,0,3,1,1,3,0,3,1,3,0,3,3,0,4,2,0,2,0,4,2,4},
      {3,3,1,1,3,0,3,1,0,0,3,3,3,3,1,1,3,0,3,1,0,0,3,3,2,4,2,0,4,0,4,2},
      {3,0,0,3,1,3,0,3,1,3,3,1,3,0,0,3,1,3,0,3,1,3,3,1,2,4,0,4,0,2,4,2},
      {3,3,0,0,3,1,3,0,1,1,3,3,3,3,0,0,3,1,3,0,1,1,3,3,4,2,4,0,2,0,2,4},
      {0,3,3,0,0,3,1,3,3,1,1,3,0,3,3,0,0,3,1,3,3,1,1,3,0,4,2,4,4,2,0,2},
      {0,0,3,3,3,0,3,1,3,3,1,1,0,0,3,3,3,0,3,1,3,3,1,1,4,0,4,2,2,4,2,0},
    };
  
  fac=1; // keep fac=1 in this version for now (parallel needs it to be)
  facTau = DATA->facTau;
  DX=fac*DATA->delta_x;
  DY=fac*DATA->delta_y;
  DETA=fac*DATA->delta_eta;
  DTAU=facTau*DATA->delta_tau;
  intersections=0;

  if (DATA->useEpsFO)
    {
      epsFO=DATA->epsilonFreeze/hbarc;
    }
  else 
    {
      rhob = 0.;
      cout << "[evolve.cpp:FindFreezeoutSurface2]: Using T_freeze works for rhob=0 only" << endl;
      epsFO= eos->findRoot(&EOS::Tsolve, rhob, DATA->TFO/hbarc, 1.15*rhob+0.001, 300.,0.001);
      cout << "T_freeze=" << DATA->TFO << ", epsFO=" << epsFO*hbarc << endl;
    }

  //  int ix, iy, ieta, nx, ny, neta, i, alpha, iflag;
  int sizeOfData = (nx+1)*(ny+1);
  int position;
  double *package;
  double *package_prev;
  double *packageutau;
  double *packageux;
  double *packageuy;
  double *packageueta;
  double *packagerhob;

  double *packageutau_prev;
  double *packageux_prev;
  double *packageuy_prev;
  double *packageueta_prev;
  double *packagerhob_prev;
  
  double **Rneighbor_eps;
  double **Rneighbor_eps_prev;
  double **Rneighbor_utau;
  double **Rneighbor_ux;
  double **Rneighbor_uy;
  double **Rneighbor_ueta;
  double **Rneighbor_rhob;
 
  double **Rneighbor_utau_prev;
  double **Rneighbor_ux_prev;
  double **Rneighbor_uy_prev;
  double **Rneighbor_ueta_prev;
  double **Rneighbor_rhob_prev;

  double *packageWtautau;
  double *packageWtaux;
  double *packageWtauy;
  double *packageWtaueta;
  double *packageWxx;
  double *packageWxy;
  double *packageWxeta;
  double *packageWyy;
  double *packageWyeta;
  double *packageWetaeta;

  double *packageWtautau_prev;
  double *packageWtaux_prev;
  double *packageWtauy_prev;
  double *packageWtaueta_prev;
  double *packageWxx_prev;
  double *packageWxy_prev;
  double *packageWxeta_prev;
  double *packageWyy_prev;
  double *packageWyeta_prev;
  double *packageWetaeta_prev;

  double **Rneighbor_Wtautau_prev;
  double **Rneighbor_Wtaux_prev;
  double **Rneighbor_Wtauy_prev;
  double **Rneighbor_Wtaueta_prev;
  double **Rneighbor_Wxx_prev;
  double **Rneighbor_Wxy_prev;
  double **Rneighbor_Wxeta_prev;
  double **Rneighbor_Wyy_prev;
  double **Rneighbor_Wyeta_prev;
  double **Rneighbor_Wetaeta_prev;

  double **Rneighbor_Wtautau;
  double **Rneighbor_Wtaux;
  double **Rneighbor_Wtauy;
  double **Rneighbor_Wtaueta;
  double **Rneighbor_Wxx;
  double **Rneighbor_Wxy;
  double **Rneighbor_Wxeta;
  double **Rneighbor_Wyy;
  double **Rneighbor_Wyeta;
  double **Rneighbor_Wetaeta;

  package = new double[sizeOfData];
  package_prev = new double[sizeOfData];
  packageutau = new double[sizeOfData];
  packageux = new double[sizeOfData];
  packageuy = new double[sizeOfData];
  packageueta = new double[sizeOfData];
  packagerhob = new double[sizeOfData];

  packageWtautau = new double[sizeOfData];
  packageWtaux = new double[sizeOfData];
  packageWtauy = new double[sizeOfData];
  packageWtaueta = new double[sizeOfData];
  packageWxx = new double[sizeOfData];
  packageWxy = new double[sizeOfData];
  packageWxeta = new double[sizeOfData];
  packageWyy = new double[sizeOfData];
  packageWyeta = new double[sizeOfData];
  packageWetaeta = new double[sizeOfData];
  
  packageWtautau_prev = new double[sizeOfData];
  packageWtaux_prev = new double[sizeOfData];
  packageWtauy_prev = new double[sizeOfData];
  packageWtaueta_prev = new double[sizeOfData];
  packageWxx_prev = new double[sizeOfData];
  packageWxy_prev = new double[sizeOfData];
  packageWxeta_prev = new double[sizeOfData];
  packageWyy_prev = new double[sizeOfData];
  packageWyeta_prev = new double[sizeOfData];
  packageWetaeta_prev = new double[sizeOfData];
  
  packageutau_prev = new double[sizeOfData];
  packageux_prev = new double[sizeOfData];
  packageuy_prev = new double[sizeOfData];
  packageueta_prev = new double[sizeOfData];
  packagerhob_prev = new double[sizeOfData];

  Rneighbor_eps = util->mtx_malloc(nx+1,ny+1);
  Rneighbor_eps_prev = util->mtx_malloc(nx+1,ny+1);
  Rneighbor_utau = util->mtx_malloc(nx+1,ny+1);
  Rneighbor_ux = util->mtx_malloc(nx+1,ny+1);
  Rneighbor_uy = util->mtx_malloc(nx+1,ny+1);
  Rneighbor_ueta = util->mtx_malloc(nx+1,ny+1);
  Rneighbor_rhob = util->mtx_malloc(nx+1,ny+1);
 
  Rneighbor_Wtautau = util->mtx_malloc(nx+1,ny+1);
  Rneighbor_Wtaux = util->mtx_malloc(nx+1,ny+1);
  Rneighbor_Wtauy = util->mtx_malloc(nx+1,ny+1);
  Rneighbor_Wtaueta = util->mtx_malloc(nx+1,ny+1);
  Rneighbor_Wxx = util->mtx_malloc(nx+1,ny+1);
  Rneighbor_Wxy = util->mtx_malloc(nx+1,ny+1);
  Rneighbor_Wxeta = util->mtx_malloc(nx+1,ny+1);
  Rneighbor_Wyy = util->mtx_malloc(nx+1,ny+1);
  Rneighbor_Wyeta = util->mtx_malloc(nx+1,ny+1);
  Rneighbor_Wetaeta = util->mtx_malloc(nx+1,ny+1);

  Rneighbor_Wtautau_prev = util->mtx_malloc(nx+1,ny+1);
  Rneighbor_Wtaux_prev = util->mtx_malloc(nx+1,ny+1);
  Rneighbor_Wtauy_prev = util->mtx_malloc(nx+1,ny+1);
  Rneighbor_Wtaueta_prev = util->mtx_malloc(nx+1,ny+1);
  Rneighbor_Wxx_prev = util->mtx_malloc(nx+1,ny+1);
  Rneighbor_Wxy_prev = util->mtx_malloc(nx+1,ny+1);
  Rneighbor_Wxeta_prev = util->mtx_malloc(nx+1,ny+1);
  Rneighbor_Wyy_prev = util->mtx_malloc(nx+1,ny+1);
  Rneighbor_Wyeta_prev = util->mtx_malloc(nx+1,ny+1);
  Rneighbor_Wetaeta_prev = util->mtx_malloc(nx+1,ny+1);
 
  Rneighbor_utau_prev = util->mtx_malloc(nx+1,ny+1);
  Rneighbor_ux_prev = util->mtx_malloc(nx+1,ny+1);
  Rneighbor_uy_prev = util->mtx_malloc(nx+1,ny+1);
  Rneighbor_ueta_prev = util->mtx_malloc(nx+1,ny+1);
  Rneighbor_rhob_prev = util->mtx_malloc(nx+1,ny+1);
  
  // receive from the right / send to the left
  int from = rank+1;
  int to = rank-1;

  // get cells from neighboring processors
  if ( rank != 0 )
    {
      //      cout << " sending to the left on rank " << rank << endl;
      for(ix=0; ix<=nx; ix++)
    {
      for(iy=0; iy<=ny; iy++)
        {
          //if (ix ==nx/2 && iy==ny/2 && alpha == 0 && i == 0 )
          //cout << "arena[ix][iy][0].TJb[i][alpha][0]=" << arena[ix][iy][0].TJb[i][alpha][0] << endl;
          position = ix + ((nx+1)*iy);
          package[position] = arena[ix][iy][0].epsilon;
          package_prev[position] = arena[ix][iy][0].epsilon_prev;
          packageutau[position] = arena[ix][iy][0].u[0][0];
          packageux[position] = arena[ix][iy][0].u[0][1];
          packageuy[position] = arena[ix][iy][0].u[0][2];
          packageueta[position] = arena[ix][iy][0].u[0][3];
          
          packageWtautau[position] = arena[ix][iy][0].Wmunu[0][0][0];
          packageWtaux[position] = arena[ix][iy][0].Wmunu[0][0][1];
          packageWtauy[position] = arena[ix][iy][0].Wmunu[0][0][2];
          packageWtaueta[position] = arena[ix][iy][0].Wmunu[0][0][3];
          packageWxx[position] = arena[ix][iy][0].Wmunu[0][1][1];
          packageWxy[position] = arena[ix][iy][0].Wmunu[0][1][2];
          packageWxeta[position] = arena[ix][iy][0].Wmunu[0][1][3];
          packageWyy[position] = arena[ix][iy][0].Wmunu[0][2][2];
          packageWyeta[position] = arena[ix][iy][0].Wmunu[0][2][3];
          packageWetaeta[position] = arena[ix][iy][0].Wmunu[0][3][3];
      
          packageWtautau_prev[position] = arena[ix][iy][0].W_prev[0][0];
          packageWtaux_prev[position] = arena[ix][iy][0].W_prev[0][1];
          packageWtauy_prev[position] = arena[ix][iy][0].W_prev[0][2];
          packageWtaueta_prev[position] = arena[ix][iy][0].W_prev[0][3];
          packageWxx_prev[position] = arena[ix][iy][0].W_prev[1][1];
          packageWxy_prev[position] = arena[ix][iy][0].W_prev[1][2];
          packageWxeta_prev[position] = arena[ix][iy][0].W_prev[1][3];
          packageWyy_prev[position] = arena[ix][iy][0].W_prev[2][2];
          packageWyeta_prev[position] = arena[ix][iy][0].W_prev[2][3];
          packageWetaeta_prev[position] = arena[ix][iy][0].W_prev[3][3];
      
          packagerhob[position] = arena[ix][iy][0].rhob;
          packageutau_prev[position] = arena[ix][iy][0].u_prev[0];
          packageux_prev[position] = arena[ix][iy][0].u_prev[1];
          packageuy_prev[position] = arena[ix][iy][0].u_prev[2];
          packageueta_prev[position] = arena[ix][iy][0].u_prev[3];
        }
    }
      MPI::COMM_WORLD.Send(package,sizeOfData,MPI::DOUBLE,to,1);
      MPI::COMM_WORLD.Send(package_prev,sizeOfData,MPI::DOUBLE,to,2);
      MPI::COMM_WORLD.Send(packageutau,sizeOfData,MPI::DOUBLE,to,3);
      MPI::COMM_WORLD.Send(packageux,sizeOfData,MPI::DOUBLE,to,4);
      MPI::COMM_WORLD.Send(packageuy,sizeOfData,MPI::DOUBLE,to,5);
      MPI::COMM_WORLD.Send(packageueta,sizeOfData,MPI::DOUBLE,to,6);
      MPI::COMM_WORLD.Send(packagerhob,sizeOfData,MPI::DOUBLE,to,7);
      MPI::COMM_WORLD.Send(packageutau_prev,sizeOfData,MPI::DOUBLE,to,8);
      MPI::COMM_WORLD.Send(packageux_prev,sizeOfData,MPI::DOUBLE,to,9);
      MPI::COMM_WORLD.Send(packageuy_prev,sizeOfData,MPI::DOUBLE,to,10);
      MPI::COMM_WORLD.Send(packageueta_prev,sizeOfData,MPI::DOUBLE,to,11);
      MPI::COMM_WORLD.Send(packageWtautau,sizeOfData,MPI::DOUBLE,to,12);
      MPI::COMM_WORLD.Send(packageWtaux,sizeOfData,MPI::DOUBLE,to,13);
      MPI::COMM_WORLD.Send(packageWtauy,sizeOfData,MPI::DOUBLE,to,14);
      MPI::COMM_WORLD.Send(packageWtaueta,sizeOfData,MPI::DOUBLE,to,15);
      MPI::COMM_WORLD.Send(packageWxx,sizeOfData,MPI::DOUBLE,to,16);
      MPI::COMM_WORLD.Send(packageWxy,sizeOfData,MPI::DOUBLE,to,17);
      MPI::COMM_WORLD.Send(packageWxeta,sizeOfData,MPI::DOUBLE,to,18);
      MPI::COMM_WORLD.Send(packageWyy,sizeOfData,MPI::DOUBLE,to,19);
      MPI::COMM_WORLD.Send(packageWyeta,sizeOfData,MPI::DOUBLE,to,20);
      MPI::COMM_WORLD.Send(packageWetaeta,sizeOfData,MPI::DOUBLE,to,21);
      MPI::COMM_WORLD.Send(packageWtautau_prev,sizeOfData,MPI::DOUBLE,to,22);
      MPI::COMM_WORLD.Send(packageWtaux_prev,sizeOfData,MPI::DOUBLE,to,23);
      MPI::COMM_WORLD.Send(packageWtauy_prev,sizeOfData,MPI::DOUBLE,to,24);
      MPI::COMM_WORLD.Send(packageWtaueta_prev,sizeOfData,MPI::DOUBLE,to,25);
      MPI::COMM_WORLD.Send(packageWxx_prev,sizeOfData,MPI::DOUBLE,to,26);
      MPI::COMM_WORLD.Send(packageWxy_prev,sizeOfData,MPI::DOUBLE,to,27);
      MPI::COMM_WORLD.Send(packageWxeta_prev,sizeOfData,MPI::DOUBLE,to,28);
      MPI::COMM_WORLD.Send(packageWyy_prev,sizeOfData,MPI::DOUBLE,to,29);
      MPI::COMM_WORLD.Send(packageWyeta_prev,sizeOfData,MPI::DOUBLE,to,30);
      MPI::COMM_WORLD.Send(packageWetaeta_prev,sizeOfData,MPI::DOUBLE,to,31);
        //cout << " done sending to the left on rank " << rank << endl;
    }
  // receiving and unwrapping the package
  if ( rank != size-1 )
    {  
      MPI::COMM_WORLD.Recv(package,sizeOfData,MPI::DOUBLE,from,1);
      MPI::COMM_WORLD.Recv(package_prev,sizeOfData,MPI::DOUBLE,from,2);
      MPI::COMM_WORLD.Recv(packageutau,sizeOfData,MPI::DOUBLE,from,3);
      MPI::COMM_WORLD.Recv(packageux,sizeOfData,MPI::DOUBLE,from,4);
      MPI::COMM_WORLD.Recv(packageuy,sizeOfData,MPI::DOUBLE,from,5);
      MPI::COMM_WORLD.Recv(packageueta,sizeOfData,MPI::DOUBLE,from,6);
      MPI::COMM_WORLD.Recv(packagerhob,sizeOfData,MPI::DOUBLE,from,7);
      MPI::COMM_WORLD.Recv(packageutau_prev,sizeOfData,MPI::DOUBLE,from,8);
      MPI::COMM_WORLD.Recv(packageux_prev,sizeOfData,MPI::DOUBLE,from,9);
      MPI::COMM_WORLD.Recv(packageuy_prev,sizeOfData,MPI::DOUBLE,from,10);
      MPI::COMM_WORLD.Recv(packageueta_prev,sizeOfData,MPI::DOUBLE,from,11);
      MPI::COMM_WORLD.Recv(packageWtautau,sizeOfData,MPI::DOUBLE,from,12);
      MPI::COMM_WORLD.Recv(packageWtaux,sizeOfData,MPI::DOUBLE,from,13);
      MPI::COMM_WORLD.Recv(packageWtauy,sizeOfData,MPI::DOUBLE,from,14);
      MPI::COMM_WORLD.Recv(packageWtaueta,sizeOfData,MPI::DOUBLE,from,15);
      MPI::COMM_WORLD.Recv(packageWxx,sizeOfData,MPI::DOUBLE,from,16);
      MPI::COMM_WORLD.Recv(packageWxy,sizeOfData,MPI::DOUBLE,from,17);
      MPI::COMM_WORLD.Recv(packageWxeta,sizeOfData,MPI::DOUBLE,from,18);
      MPI::COMM_WORLD.Recv(packageWyy,sizeOfData,MPI::DOUBLE,from,19);
      MPI::COMM_WORLD.Recv(packageWyeta,sizeOfData,MPI::DOUBLE,from,20);
      MPI::COMM_WORLD.Recv(packageWetaeta,sizeOfData,MPI::DOUBLE,from,21);
      MPI::COMM_WORLD.Recv(packageWtautau_prev,sizeOfData,MPI::DOUBLE,from,22);
      MPI::COMM_WORLD.Recv(packageWtaux_prev,sizeOfData,MPI::DOUBLE,from,23);
      MPI::COMM_WORLD.Recv(packageWtauy_prev,sizeOfData,MPI::DOUBLE,from,24);
      MPI::COMM_WORLD.Recv(packageWtaueta_prev,sizeOfData,MPI::DOUBLE,from,25);
      MPI::COMM_WORLD.Recv(packageWxx_prev,sizeOfData,MPI::DOUBLE,from,26);
      MPI::COMM_WORLD.Recv(packageWxy_prev,sizeOfData,MPI::DOUBLE,from,27);
      MPI::COMM_WORLD.Recv(packageWxeta_prev,sizeOfData,MPI::DOUBLE,from,28);
      MPI::COMM_WORLD.Recv(packageWyy_prev,sizeOfData,MPI::DOUBLE,from,29);
      MPI::COMM_WORLD.Recv(packageWyeta_prev,sizeOfData,MPI::DOUBLE,from,30);
      MPI::COMM_WORLD.Recv(packageWetaeta_prev,sizeOfData,MPI::DOUBLE,from,31);
      //cout << " receiving from the right on rank " << rank << endl;
      for(ix=0; ix<=nx; ix++)
    {
      for(iy=0; iy<=ny; iy++)
        {
          position = ix + ((nx+1)*iy);
          //if (ix ==nx/2 && iy==ny/2 && alpha == 0 && i == 0 )
          //cout << "Rneighbor[ix][iy][0].TJb[i][alpha][0]=" << package[position] << endl;
          Rneighbor_eps[ix][iy] = package[position];
          Rneighbor_eps_prev[ix][iy] = package_prev[position];
          Rneighbor_utau[ix][iy] = packageutau[position];
          Rneighbor_ux[ix][iy] = packageux[position];
          Rneighbor_uy[ix][iy] = packageuy[position];
          Rneighbor_ueta[ix][iy] = packageueta[position];
          Rneighbor_rhob[ix][iy] = packagerhob[position];
          Rneighbor_utau_prev[ix][iy] = packageutau_prev[position];
          Rneighbor_ux_prev[ix][iy] = packageux_prev[position];
          Rneighbor_uy_prev[ix][iy] = packageuy_prev[position];
          Rneighbor_ueta_prev[ix][iy] = packageueta_prev[position];
          Rneighbor_Wtautau[ix][iy] = packageWtautau[position];
          Rneighbor_Wtaux[ix][iy] = packageWtaux[position];
          Rneighbor_Wtauy[ix][iy] = packageWtauy[position];
          Rneighbor_Wtaueta[ix][iy] = packageWtaueta[position];
          Rneighbor_Wxx[ix][iy] = packageWxx[position];
          Rneighbor_Wxy[ix][iy] = packageWxy[position];
          Rneighbor_Wxeta[ix][iy] = packageWxeta[position];
          Rneighbor_Wyy[ix][iy] = packageWyy[position];
          Rneighbor_Wyeta[ix][iy] = packageWyeta[position];
          Rneighbor_Wetaeta[ix][iy] = packageWetaeta[position];
          Rneighbor_Wtautau_prev[ix][iy] = packageWtautau_prev[position];
          Rneighbor_Wtaux_prev[ix][iy] = packageWtaux_prev[position];
          Rneighbor_Wtauy_prev[ix][iy] = packageWtauy_prev[position];
          Rneighbor_Wtaueta_prev[ix][iy] = packageWtaueta_prev[position];
          Rneighbor_Wxx_prev[ix][iy] = packageWxx_prev[position];
          Rneighbor_Wxy_prev[ix][iy] = packageWxy_prev[position];
          Rneighbor_Wxeta_prev[ix][iy] = packageWxeta_prev[position];
          Rneighbor_Wyy_prev[ix][iy] = packageWyy_prev[position];
          Rneighbor_Wyeta_prev[ix][iy] = packageWyeta_prev[position];
          Rneighbor_Wetaeta_prev[ix][iy] = packageWetaeta_prev[position];
        }
    }
      //cout << " done receiving from the right on rank " << rank << endl;
    }

  if (rank == size-1) maxEta = neta-fac;
  else maxEta = neta;
  
  for(ix=0; ix<=nx-fac; ix+=fac)
    {
      // fprintf(stderr,"IBIT[%d][%d]=%d\n",0,0,IBIT[0][0]);
      // fprintf(stderr,"IBIT[%d][%d]=%d\n",20,6,IBIT[20-1][6-1]);
      x = ix*(DATA->delta_x) - (DATA->x_size/2.0); 
      for(iy=0; iy<=ny-fac; iy+=fac)
    {
      y = iy*(DATA->delta_y) - (DATA->y_size/2.0);
      for(ieta=0; ieta<maxEta; ieta+=fac)
        {
          eta = (DATA->delta_eta)*(ieta+DATA->neta*rank) - (DATA->eta_size)/2.0;
          //fprintf(stderr, "%d, %d, %d\n",ix,iy,ieta);
          

          // make sure the epsilon value is never exactly the same as epsFO...
          if(arena[ix+fac][iy+fac][ieta+fac].epsilon==epsFO)
        arena[ix+fac][iy+fac][ieta+fac].epsilon+=0.000001;

          if(arena[ix][iy][ieta].epsilon_prev==epsFO)
        arena[ix][iy][ieta].epsilon_prev+=0.000001;
          
          if(arena[ix+fac][iy][ieta].epsilon==epsFO)
        arena[ix+fac][iy][ieta].epsilon+=0.000001;
          
          if(arena[ix][iy+fac][ieta+fac].epsilon_prev==epsFO)
        arena[ix][iy+fac][ieta+fac].epsilon_prev+=0.000001;
          
          if(arena[ix][iy+fac][ieta].epsilon==epsFO)
        arena[ix][iy+fac][ieta].epsilon+=0.000001;
          
          if(arena[ix+fac][iy][ieta+fac].epsilon_prev==epsFO)
        arena[ix+fac][iy][ieta+fac].epsilon_prev+=0.000001;
          
          if(arena[ix][iy][ieta+fac].epsilon==epsFO)
        arena[ix][iy][ieta+fac].epsilon+=0.000001;
          
          if(arena[ix+fac][iy+fac][ieta].epsilon_prev==epsFO)
        arena[ix+fac][iy+fac][ieta].epsilon_prev+=0.000001;
          
          if(arena[ix+fac][iy+fac][ieta].epsilon==epsFO)
        arena[ix+fac][iy+fac][ieta].epsilon+=0.000001;
          
          if(arena[ix][iy][ieta+fac].epsilon_prev==epsFO)
        arena[ix][iy][ieta+fac].epsilon_prev+=0.000001;
          
          if(arena[ix+fac][iy][ieta+fac].epsilon==epsFO)
        arena[ix+fac][iy][ieta+fac].epsilon+=0.000001;
          
          if(arena[ix][iy+fac][ieta].epsilon_prev==epsFO)
        arena[ix][iy+fac][ieta].epsilon_prev+=0.000001;
          
          if(arena[ix][iy+fac][ieta+fac].epsilon==epsFO)
        arena[ix][iy+fac][ieta+fac].epsilon+=0.000001;
          
          if(arena[ix+fac][iy][ieta].epsilon_prev==epsFO)
        arena[ix+fac][iy][ieta].epsilon_prev+=0.000001;
          
          if(arena[ix][iy][ieta].epsilon==epsFO)
        arena[ix][iy][ieta].epsilon+=0.000001;
          
          if(arena[ix+fac][iy+fac][ieta+fac].epsilon_prev==epsFO)
        arena[ix+fac][iy+fac][ieta+fac].epsilon_prev+=0.000001;

          
          if (ieta<neta-fac)
        {
          intersect=1;
          if((arena[ix+fac][iy+fac][ieta+fac].epsilon-epsFO)*(arena[ix][iy][ieta].epsilon_prev-epsFO)>0.)
            if((arena[ix+fac][iy][ieta].epsilon-epsFO)*(arena[ix][iy+fac][ieta+fac].epsilon_prev-epsFO)>0.)
              if((arena[ix][iy+fac][ieta].epsilon-epsFO)*(arena[ix+fac][iy][ieta+fac].epsilon_prev-epsFO)>0.)
            if((arena[ix][iy][ieta+fac].epsilon-epsFO)*(arena[ix+fac][iy+fac][ieta].epsilon_prev-epsFO)>0.)
              if((arena[ix+fac][iy+fac][ieta].epsilon-epsFO)*(arena[ix][iy][ieta+fac].epsilon_prev-epsFO)>0.)
                if((arena[ix+fac][iy][ieta+fac].epsilon-epsFO)*(arena[ix][iy+fac][ieta].epsilon_prev-epsFO)>0.)
                  if((arena[ix][iy+fac][ieta+fac].epsilon-epsFO)*(arena[ix+fac][iy][ieta].epsilon_prev-epsFO)>0.)
                if((arena[ix][iy][ieta].epsilon-epsFO)*(arena[ix+fac][iy+fac][ieta+fac].epsilon_prev-epsFO)>0.)
                  intersect=0;
        }
          else // if this is the right most edge
        {
          intersect=1;
          if((Rneighbor_eps[ix+fac][iy+fac] - epsFO)*(arena[ix][iy][ieta].epsilon_prev-epsFO)>0.)
            if((arena[ix+fac][iy][ieta].epsilon-epsFO)*(Rneighbor_eps_prev[ix][iy+fac]-epsFO)>0.)
              if((arena[ix][iy+fac][ieta].epsilon-epsFO)*(Rneighbor_eps_prev[ix+fac][iy]-epsFO)>0.)
            if((Rneighbor_eps[ix][iy]-epsFO)*(arena[ix+fac][iy+fac][ieta].epsilon_prev-epsFO)>0.)
              if((arena[ix+fac][iy+fac][ieta].epsilon-epsFO)*(Rneighbor_eps_prev[ix][iy]-epsFO)>0.)
                if((Rneighbor_eps[ix+fac][iy]-epsFO)*(arena[ix][iy+fac][ieta].epsilon_prev-epsFO)>0.)
                  if((Rneighbor_eps[ix][iy+fac]-epsFO)*(arena[ix+fac][iy][ieta].epsilon_prev-epsFO)>0.)
                if((arena[ix][iy][ieta].epsilon-epsFO)*(Rneighbor_eps_prev[ix+fac][iy+fac]-epsFO)>0.)
                  intersect=0;
        }

          //fprintf(stderr, "done checks\n");
          if (intersect==0)
        {
          //fprintf(stderr, "continue...\n");
          continue;
        }
          else       
        {
          //fprintf(stderr, "starting freeze out\n");
          intersections++;
          if (ieta<neta-fac)
            {
              cube[0]=arena[ix][iy][ieta].epsilon_prev;
              cube[1]=arena[ix+fac][iy][ieta].epsilon_prev;
              cube[2]=arena[ix+fac][iy+fac][ieta].epsilon_prev;
              cube[3]=arena[ix][iy+fac][ieta].epsilon_prev;
              cube[4]=arena[ix][iy][ieta].epsilon;
              cube[5]=arena[ix+fac][iy][ieta].epsilon;
              cube[6]=arena[ix+fac][iy+fac][ieta].epsilon;
              cube[7]=arena[ix][iy+fac][ieta].epsilon;
              cube[8]=arena[ix][iy][ieta+fac].epsilon_prev;
              cube[9]=arena[ix+fac][iy][ieta+fac].epsilon_prev;
              cube[10]=arena[ix+fac][iy+fac][ieta+fac].epsilon_prev;
              cube[11]=arena[ix][iy+fac][ieta+fac].epsilon_prev;
              cube[12]=arena[ix][iy][ieta+fac].epsilon;
              cube[13]=arena[ix+fac][iy][ieta+fac].epsilon;
              cube[14]=arena[ix+fac][iy+fac][ieta+fac].epsilon;
              cube[15]=arena[ix][iy+fac][ieta+fac].epsilon;
            }
          else
            {
              cube[0]=arena[ix][iy][ieta].epsilon_prev;
              cube[1]=arena[ix+fac][iy][ieta].epsilon_prev;
              cube[2]=arena[ix+fac][iy+fac][ieta].epsilon_prev;
              cube[3]=arena[ix][iy+fac][ieta].epsilon_prev;
              cube[4]=arena[ix][iy][ieta].epsilon;
              cube[5]=arena[ix+fac][iy][ieta].epsilon;
              cube[6]=arena[ix+fac][iy+fac][ieta].epsilon;
              cube[7]=arena[ix][iy+fac][ieta].epsilon;
              cube[8]=Rneighbor_eps_prev[ix][iy];
              cube[9]=Rneighbor_eps_prev[ix+fac][iy];
              cube[10]=Rneighbor_eps_prev[ix+fac][iy+fac];
              cube[11]=Rneighbor_eps_prev[ix][iy+fac];
              cube[12]=Rneighbor_eps[ix][iy];
              cube[13]=Rneighbor_eps[ix+fac][iy];
              cube[14]=Rneighbor_eps[ix+fac][iy+fac];
              cube[15]=Rneighbor_eps[ix][iy+fac];
            }

          /*      // set values for testing: */
/*        cube[0]=epsFO+0.01; // 0    0   0   0 */
/*        cube[1]=epsFO-0.01; // 0    dx  0   0 */
/*        cube[2]=epsFO-0.01; // 0    dx  dy  0 */
/*        cube[3]=epsFO+0.01; // 0    0   dy  0 */
/*        cube[4]=epsFO+0.01; // dtau 0   0   0 */
/*        cube[5]=epsFO-0.01; // dtau dx  0   0 */
/*        cube[6]=epsFO-0.01; // dtau dx  dy  0 */
/*        cube[7]=epsFO+0.01; // dtau 0   dy  0 */
/*        cube[8]=epsFO+0.01; // 0    0   0   deta */
/*        cube[9]=epsFO-0.01; // 0    dx  0   deta */
/*        cube[10]=epsFO-0.01;// 0    dx  dy  deta */
/*        cube[11]=epsFO+0.01;// 0    0   dy  deta */
/*        cube[12]=epsFO+0.01;// dtau 0   0   deta */
/*        cube[13]=epsFO-0.01;// dtau dx  0   deta */
/*        cube[14]=epsFO-0.01;// dtau dx  dy  deta */
/*        cube[15]=epsFO+0.01;// dtau 0   dy  deta */
          
                  // set values for testing:
/*        cube[0]=epsFO+0.01; // 0    0   0   0 */
/*        cube[1]=epsFO-0.01; // 0    dx  0   0 */
/*        cube[2]=epsFO-0.01; // 0    dx  dy  0 */
/*        cube[3]=epsFO-0.01; // 0    0   dy  0 */
/*        cube[4]=epsFO+0.01; // dtau 0   0   0 */
/*        cube[5]=epsFO+0.01; // dtau dx  0   0 */
/*        cube[6]=epsFO+0.01; // dtau dx  dy  0 */
/*        cube[7]=epsFO+0.01; // dtau 0   dy  0 */
/*        cube[8]=epsFO+0.01; // 0    0   0   deta */
/*        cube[9]=epsFO-0.01; // 0    dx  0   deta */
/*        cube[10]=epsFO-0.01;// 0    dx  dy  deta */
/*        cube[11]=epsFO-0.01;// 0    0   dy  deta */
/*        cube[12]=epsFO+0.01;// dtau 0   0   deta */
/*        cube[13]=epsFO+0.01;// dtau dx  0   deta */
/*        cube[14]=epsFO+0.01;// dtau dx  dy  deta */
/*        cube[15]=epsFO+0.01;// dtau 0   dy  deta */
          
          NSurfaces = 0;
          ELowerSum = 0.;
          EHigherSum = 0.;
          VLower0=0.;
          VHigher0=0.;
          VLower1=0.;
          VHigher1=0.;
          VLower2=0.;
          VHigher2=0.;
          VLower3=0.;
          VHigher3=0.;


          // edge 1:
          EK = cube[0];
          EL = cube[1];
          DEK = epsFO-EK;
          DEL = epsFO-EL;
          ADEK = fabs(DEK);
          if( DEK*DEL<0 )
            {
              NSurfaces += 1;
              iEdge[NSurfaces-1] = 1;
              cuts[NSurfaces-1][0] = 0.;
              cuts[NSurfaces-1][1] = DEK/(EL-EK)*DX;
              cuts[NSurfaces-1][2] = 0.;
              cuts[NSurfaces-1][3] = 0.;
            }
          if ( DEK > 0 )
            ELowerSum+=ADEK;
          else
            EHigherSum+=ADEK;

          // edge 2: v for point 1
          EK = cube[1];
          EL = cube[2];
          DEK = epsFO-EK;
          DEL = epsFO-EL;
          ADEK = fabs(DEK);
          if( DEK*DEL<0 )
            {
              NSurfaces += 1;
              iEdge[NSurfaces-1] = 2;
              cuts[NSurfaces-1][0] = 0.;
              cuts[NSurfaces-1][1] = DX;
              cuts[NSurfaces-1][2] = DEK/(EL-EK)*DY;
              cuts[NSurfaces-1][3] = 0.;
            }
          if ( DEK > 0 )
            {
              ELowerSum+=ADEK;
              VLower1+=ADEK;
            }
          else
            {
              EHigherSum+=ADEK;
              VHigher1+=ADEK;
            }
          
          // edge 3: v for point 2
          EK = cube[3];
          EL = cube[2];
          DEK = epsFO-EK;
          DEL = epsFO-EL;
          ADEL = fabs(DEL);
          if( DEK*DEL<0 )
            {
              NSurfaces += 1;
              iEdge[NSurfaces-1] = 3;
              cuts[NSurfaces-1][0] = 0.;
              cuts[NSurfaces-1][1] = DEK/(EL-EK)*DX;
              cuts[NSurfaces-1][2] = DY;
              cuts[NSurfaces-1][3] = 0.;
            }
          if ( DEL > 0 )
            {
              ELowerSum+=ADEL;
              VLower1+=ADEL;
              VLower2+=ADEL;
            }
          else
            {
              EHigherSum+=ADEL;
              VHigher1+=ADEL;
              VHigher2+=ADEL;
            }

          // edge 4: v for point 3
          EK = cube[0];
          EL = cube[3];
          DEK = epsFO-EK;
          DEL = epsFO-EL;
          ADEL = fabs(DEL);
          if( DEK*DEL<0 )
            {
              NSurfaces += 1;
              iEdge[NSurfaces-1] = 4;
              cuts[NSurfaces-1][0] = 0.;
              cuts[NSurfaces-1][1] = 0.;
              cuts[NSurfaces-1][2] = DEK/(EL-EK)*DY;
              cuts[NSurfaces-1][3] = 0.;
            }
          if ( DEL > 0 )
            {
              ELowerSum+=ADEL;
              VLower2+=ADEL;
            }
          else
            {
              EHigherSum+=ADEL;
              VHigher2+=ADEL;
            }
        
          // edge 5: v for point 4
          EK = cube[0];
          EL = cube[4];
          DEK = epsFO-EK;
          DEL = epsFO-EL;
          ADEL = fabs(DEL);
          if( DEK*DEL<0 )
            {
              NSurfaces += 1;
              iEdge[NSurfaces-1] = 5;
              cuts[NSurfaces-1][0] = DEK/(EL-EK)*DTAU;
              cuts[NSurfaces-1][1] = 0.;
              cuts[NSurfaces-1][2] = 0.;
              cuts[NSurfaces-1][3] = 0.;
            }
          if ( DEL > 0 )
            {
              ELowerSum+=ADEL;
              VLower0+=ADEL;
            }
          else
            {
              EHigherSum+=ADEL;
              VHigher0+=ADEL;
            }


          // edge 6: v for point 5
          EK = cube[1];
          EL = cube[5];
          DEK = epsFO-EK;
          DEL = epsFO-EL;
          ADEL = fabs(DEL);
          if( DEK*DEL<0 )
            {
              NSurfaces += 1;
              iEdge[NSurfaces-1] = 6;
              cuts[NSurfaces-1][0] = DEK/(EL-EK)*DTAU;
              cuts[NSurfaces-1][1] = DX;
              cuts[NSurfaces-1][2] = 0.;
              cuts[NSurfaces-1][3] = 0.;
            }
          if ( DEL > 0 )
            {
              ELowerSum+=ADEL;
              VLower0+=ADEL;
              VLower1+=ADEL;
            }
          else
            {
              EHigherSum+=ADEL;
              VHigher0+=ADEL;
              VHigher1+=ADEL;
            }

          // edge 7: v for point 6
          EK = cube[2];
          EL = cube[6];
          DEK = epsFO-EK;
          DEL = epsFO-EL;
          ADEL = fabs(DEL);
          if( DEK*DEL<0 )
            {
              NSurfaces += 1;
              iEdge[NSurfaces-1] = 7;
              cuts[NSurfaces-1][0] = DEK/(EL-EK)*DTAU;
              cuts[NSurfaces-1][1] = DX;
              cuts[NSurfaces-1][2] = DY;
              cuts[NSurfaces-1][3] = 0.;
            }
          if ( DEL > 0 )
            {
              ELowerSum+=ADEL;
              VLower0+=ADEL;
              VLower1+=ADEL;
              VLower2+=ADEL;
            }
          else
            {
              EHigherSum+=ADEL;
              VHigher0+=ADEL;
              VHigher1+=ADEL;
              VHigher2+=ADEL;
            }

          // edge 8: v for point 7
          EK = cube[3];
          EL = cube[7];
          DEK = epsFO-EK;
          DEL = epsFO-EL;
          ADEL = fabs(DEL);
          if( DEK*DEL<0 )
            {
              NSurfaces += 1;
              iEdge[NSurfaces-1] = 8;
              cuts[NSurfaces-1][0] = DEK/(EL-EK)*DTAU;
              cuts[NSurfaces-1][1] = 0.;
              cuts[NSurfaces-1][2] = DY;
              cuts[NSurfaces-1][3] = 0.;
            }
          if ( DEL > 0 )
            {
              ELowerSum+=ADEL;
              VLower0+=ADEL;
              VLower2+=ADEL;
            }
          else
            {
              EHigherSum+=ADEL;
              VHigher0+=ADEL;
              VHigher2+=ADEL;
            }

          // edge 9:
          EK = cube[4];
          EL = cube[5];
          DEK = epsFO-EK;
          DEL = epsFO-EL;
          ADEL = fabs(DEL);
          if( DEK*DEL<0 )
            {
              NSurfaces += 1;
              iEdge[NSurfaces-1] = 9;
              cuts[NSurfaces-1][0] = DTAU;
              cuts[NSurfaces-1][1] = DEK/(EL-EK)*DX;
              cuts[NSurfaces-1][2] = 0.;
              cuts[NSurfaces-1][3] = 0.;
            }

          // edge 10:
          EK = cube[5];
          EL = cube[6];
          DEK = epsFO-EK;
          DEL = epsFO-EL;
          ADEL = fabs(DEL);
          if( DEK*DEL<0 )
            {
              NSurfaces += 1;
              iEdge[NSurfaces-1] = 10;
              cuts[NSurfaces-1][0] = DTAU;
              cuts[NSurfaces-1][1] = DX;
              cuts[NSurfaces-1][2] = DEK/(EL-EK)*DY;
              cuts[NSurfaces-1][3] = 0.;
            }

          // edge 11:
          EK = cube[7];
          EL = cube[6];
          DEK = epsFO-EK;
          DEL = epsFO-EL;
          ADEL = fabs(DEL);
          if( DEK*DEL<0 )
            {
              NSurfaces += 1;
              iEdge[NSurfaces-1] = 11;
              cuts[NSurfaces-1][0] = DTAU;
              cuts[NSurfaces-1][1] = DEK/(EL-EK)*DX;
              cuts[NSurfaces-1][2] = DY;
              cuts[NSurfaces-1][3] = 0.;
            }

          // edge 12:
          EK = cube[4];
          EL = cube[7];
          DEK = epsFO-EK;
          DEL = epsFO-EL;
          ADEL = fabs(DEL);
          if( DEK*DEL<0 )
            {
              NSurfaces += 1;
              iEdge[NSurfaces-1] = 12;
              cuts[NSurfaces-1][0] = DTAU;
              cuts[NSurfaces-1][1] = 0.;
              cuts[NSurfaces-1][2] = DEK/(EL-EK)*DY;
              cuts[NSurfaces-1][3] = 0.;
            }

          // edge 13: v for point 8
          EK = cube[8];
          EL = cube[9];
          DEK = epsFO-EK;
          DEL = epsFO-EL;
          ADEK = fabs(DEK);
          if( DEK*DEL<0 )
            {
              NSurfaces += 1;
              iEdge[NSurfaces-1] = 13;
              cuts[NSurfaces-1][0] = 0.;
              cuts[NSurfaces-1][1] = DEK/(EL-EK)*DX;
              cuts[NSurfaces-1][2] = 0.;
              cuts[NSurfaces-1][3] = DETA;
            }
          if ( DEK > 0 )
            {
              VLower3+=ADEK;
              ELowerSum+=ADEK;
            }
          else
            {
              VHigher3+=ADEK;
              EHigherSum+=ADEK;
            }

          // edge 14: v for point 9
          EK = cube[9];
          EL = cube[10];
          DEK = epsFO-EK;
          DEL = epsFO-EL;
          ADEK = fabs(DEK);
          if( DEK*DEL<0 )
            {
              NSurfaces += 1;
              iEdge[NSurfaces-1] = 14;
              cuts[NSurfaces-1][0] = 0.;
              cuts[NSurfaces-1][1] = DX;
              cuts[NSurfaces-1][2] = DEK/(EL-EK)*DY;
              cuts[NSurfaces-1][3] = DETA;
            }
          if ( DEK > 0 )
            {
              ELowerSum+=ADEK;
              VLower1+=ADEK;
              VLower3+=ADEK;
            }
          else
            {
              EHigherSum+=ADEK;
              VHigher1+=ADEK;
              VHigher3+=ADEK;
            }
          
          // edge 15: v for point 10
          EK = cube[11];
          EL = cube[10];
          DEK = epsFO-EK;
          DEL = epsFO-EL;
          ADEL = fabs(DEL);
          if( DEK*DEL<0 )
            {
              NSurfaces += 1;
              iEdge[NSurfaces-1] = 15;
              cuts[NSurfaces-1][0] = 0.;
              cuts[NSurfaces-1][1] = DEK/(EL-EK)*DX;
              cuts[NSurfaces-1][2] = DY;
              cuts[NSurfaces-1][3] = DETA;
            }
          if ( DEL > 0 )
            {
              ELowerSum+=ADEL;
              VLower1+=ADEL;
              VLower2+=ADEL;
              VLower3+=ADEL;
            }
          else
            {
              EHigherSum+=ADEL;
              VHigher1+=ADEL;
              VHigher2+=ADEL;
              VHigher3+=ADEL;
            }

          // edge 16: v for point 11
          EK = cube[8];
          EL = cube[11];
          DEK = epsFO-EK;
          DEL = epsFO-EL;
          ADEL = fabs(DEL);
          if( DEK*DEL<0 )
            {
              NSurfaces += 1;
              iEdge[NSurfaces-1] = 16;
              cuts[NSurfaces-1][0] = 0.;
              cuts[NSurfaces-1][1] = 0.;
              cuts[NSurfaces-1][2] = DEK/(EL-EK)*DY;
              cuts[NSurfaces-1][3] = DETA;
            }
          if ( DEL > 0 )
            {
              ELowerSum+=ADEL;
              VLower2+=ADEL;
              VLower3+=ADEL;
            }
          else
            {
              EHigherSum+=ADEL;
              VHigher2+=ADEL;
              VHigher3+=ADEL;
            }
        
          // edge 17: v for point 12
          EK = cube[8];
          EL = cube[12];
          DEK = epsFO-EK;
          DEL = epsFO-EL;
          ADEL = fabs(DEL);
          if( DEK*DEL<0 )
            {
              NSurfaces += 1;
              iEdge[NSurfaces-1] = 17;
              cuts[NSurfaces-1][0] = DEK/(EL-EK)*DTAU;
              cuts[NSurfaces-1][1] = 0.;
              cuts[NSurfaces-1][2] = 0.;
              cuts[NSurfaces-1][3] = DETA;
            }
          if ( DEL > 0 )
            {
              ELowerSum+=ADEL;
              VLower0+=ADEL;
              VLower3+=ADEL;
            }
          else
            {
              EHigherSum+=ADEL;
              VHigher0+=ADEL;
              VHigher3+=ADEL;
            }


          // edge 18: v for point 13
          EK = cube[9];
          EL = cube[13];
          DEK = epsFO-EK;
          DEL = epsFO-EL;
          ADEL = fabs(DEL);
          if( DEK*DEL<0 )
            {
              NSurfaces += 1;
              iEdge[NSurfaces-1] = 18;
              cuts[NSurfaces-1][0] = DEK/(EL-EK)*DTAU;
              cuts[NSurfaces-1][1] = DX;
              cuts[NSurfaces-1][2] = 0.;
              cuts[NSurfaces-1][3] = DETA;
            }
          if ( DEL > 0 )
            {
              ELowerSum+=ADEL;
              VLower0+=ADEL;
              VLower1+=ADEL;
              VLower3+=ADEL;
            }
          else
            {
              EHigherSum+=ADEL;
              VHigher0+=ADEL;
              VHigher1+=ADEL;
              VHigher3+=ADEL;
            }

          // edge 19: v for point 14
          EK = cube[10];
          EL = cube[14];
          DEK = epsFO-EK;
          DEL = epsFO-EL;
          ADEL = fabs(DEL);
          if( DEK*DEL<0 )
            {
              NSurfaces += 1;
              iEdge[NSurfaces-1] = 19;
              cuts[NSurfaces-1][0] = DEK/(EL-EK)*DTAU;
              cuts[NSurfaces-1][1] = DX;
              cuts[NSurfaces-1][2] = DY;
              cuts[NSurfaces-1][3] = DETA;
            }
          if ( DEL > 0 )
            {
              ELowerSum+=ADEL;
              VLower0+=ADEL;
              VLower1+=ADEL;
              VLower2+=ADEL;
              VLower3+=ADEL;
            }
          else
            {
              EHigherSum+=ADEL;
              VHigher0+=ADEL;
              VHigher1+=ADEL;
              VHigher2+=ADEL;
              VHigher3+=ADEL;
            }

          // edge 20: v for point 15
          EK = cube[11];
          EL = cube[15];
          DEK = epsFO-EK;
          DEL = epsFO-EL;
          ADEL = fabs(DEL);
          if( DEK*DEL<0 )
            {
              NSurfaces += 1;
              iEdge[NSurfaces-1] = 20;
              cuts[NSurfaces-1][0] = DEK/(EL-EK)*DTAU;
              cuts[NSurfaces-1][1] = 0.;
              cuts[NSurfaces-1][2] = DY;
              cuts[NSurfaces-1][3] = DETA;
            }
          if ( DEL > 0 )
            {
              ELowerSum+=ADEL;
              VLower0+=ADEL;
              VLower2+=ADEL;
              VLower3+=ADEL;
            }
          else
            {
              EHigherSum+=ADEL;
              VHigher0+=ADEL;
              VHigher2+=ADEL;
              VHigher3+=ADEL;
            }

          // edge 21:
          EK = cube[12];
          EL = cube[13];
          DEK = epsFO-EK;
          DEL = epsFO-EL;
          ADEL = fabs(DEL);
          if( DEK*DEL<0 )
            {
              NSurfaces += 1;
              iEdge[NSurfaces-1] = 21;
              cuts[NSurfaces-1][0] = DTAU;
              cuts[NSurfaces-1][1] = DEK/(EL-EK)*DX;
              cuts[NSurfaces-1][2] = 0.;
              cuts[NSurfaces-1][3] = DETA;
            }

          // edge 22:
          EK = cube[13];
          EL = cube[14];
          DEK = epsFO-EK;
          DEL = epsFO-EL;
          ADEL = fabs(DEL);
          if( DEK*DEL<0 )
            {
              NSurfaces += 1;
              iEdge[NSurfaces-1] = 22;
              cuts[NSurfaces-1][0] = DTAU;
              cuts[NSurfaces-1][1] = DX;
              cuts[NSurfaces-1][2] = DEK/(EL-EK)*DY;
              cuts[NSurfaces-1][3] = DETA;
            }

          // edge 23:
          EK = cube[15];
          EL = cube[14];
          DEK = epsFO-EK;
          DEL = epsFO-EL;
          ADEL = fabs(DEL);
          if( DEK*DEL<0 )
            {
              NSurfaces += 1;
              iEdge[NSurfaces-1] = 23;
              cuts[NSurfaces-1][0] = DTAU;
              cuts[NSurfaces-1][1] = DEK/(EL-EK)*DX;
              cuts[NSurfaces-1][2] = DY;
              cuts[NSurfaces-1][3] = DETA;
            }

          // edge 24:
          EK = cube[12];
          EL = cube[15];
          DEK = epsFO-EK;
          DEL = epsFO-EL;
          ADEL = fabs(DEL);
          if( DEK*DEL<0 )
            {
              NSurfaces += 1;
              iEdge[NSurfaces-1] = 24;
              cuts[NSurfaces-1][0] = DTAU;
              cuts[NSurfaces-1][1] = 0.;
              cuts[NSurfaces-1][2] = DEK/(EL-EK)*DY;
              cuts[NSurfaces-1][3] = DETA;
            }

          // edge 25:
          EK = cube[0];
          EL = cube[8];
          DEK = epsFO-EK;
          DEL = epsFO-EL;
          ADEL = fabs(DEL);
          if( DEK*DEL<0 )
            {
              NSurfaces += 1;
              iEdge[NSurfaces-1] = 25;
              cuts[NSurfaces-1][0] = 0.;
              cuts[NSurfaces-1][1] = 0.;
              cuts[NSurfaces-1][2] = 0.;
              cuts[NSurfaces-1][3] = DEK/(EL-EK)*DETA;
            }

          // edge 26:
          EK = cube[1];
          EL = cube[9];
          DEK = epsFO-EK;
          DEL = epsFO-EL;
          ADEL = fabs(DEL);
          if( DEK*DEL<0 )
            {
              NSurfaces += 1;
              iEdge[NSurfaces-1] = 26;
              cuts[NSurfaces-1][0] = 0.;
              cuts[NSurfaces-1][1] = DX;
              cuts[NSurfaces-1][2] = 0.;
              cuts[NSurfaces-1][3] = DEK/(EL-EK)*DETA;
            }

          // edge 27:
          EK = cube[2];
          EL = cube[10];
          DEK = epsFO-EK;
          DEL = epsFO-EL;
          ADEL = fabs(DEL);
          if( DEK*DEL<0 )
            {
              NSurfaces += 1;
              iEdge[NSurfaces-1] = 27;
              cuts[NSurfaces-1][0] = 0.;
              cuts[NSurfaces-1][1] = DX;
              cuts[NSurfaces-1][2] = DY;
              cuts[NSurfaces-1][3] = DEK/(EL-EK)*DETA;
            }

          // edge 28:
          EK = cube[3];
          EL = cube[11];
          DEK = epsFO-EK;
          DEL = epsFO-EL;
          ADEL = fabs(DEL);
          if( DEK*DEL<0 )
            {
              NSurfaces += 1;
              iEdge[NSurfaces-1] = 28;
              cuts[NSurfaces-1][0] = 0.;
              cuts[NSurfaces-1][1] = 0.;
              cuts[NSurfaces-1][2] = DY;
              cuts[NSurfaces-1][3] = DEK/(EL-EK)*DETA;
            }

          // edge 29:
          EK = cube[4];
          EL = cube[12];
          DEK = epsFO-EK;
          DEL = epsFO-EL;
          ADEL = fabs(DEL);
          if( DEK*DEL<0 )
            {
              NSurfaces += 1;
              iEdge[NSurfaces-1] = 29;
              cuts[NSurfaces-1][0] = DTAU;
              cuts[NSurfaces-1][1] = 0.;
              cuts[NSurfaces-1][2] = 0.;
              cuts[NSurfaces-1][3] = DEK/(EL-EK)*DETA;
            }

          // edge 30:
          EK = cube[5];
          EL = cube[13];
          DEK = epsFO-EK;
          DEL = epsFO-EL;
          ADEL = fabs(DEL);
          if( DEK*DEL<0 )
            {
              NSurfaces += 1;
              iEdge[NSurfaces-1] = 30;
              cuts[NSurfaces-1][0] = DTAU;
              cuts[NSurfaces-1][1] = DX;
              cuts[NSurfaces-1][2] = 0.;
              cuts[NSurfaces-1][3] = DEK/(EL-EK)*DETA;
            }

          // edge 31:
          EK = cube[6];
          EL = cube[14];
          DEK = epsFO-EK;
          DEL = epsFO-EL;
          ADEL = fabs(DEL);
          if( DEK*DEL<0 )
            {
              NSurfaces += 1;
              iEdge[NSurfaces-1] = 31;
              cuts[NSurfaces-1][0] = DTAU;
              cuts[NSurfaces-1][1] = DX;
              cuts[NSurfaces-1][2] = DY;
              cuts[NSurfaces-1][3] = DEK/(EL-EK)*DETA;
            }

          // edge 32:
          EK = cube[7];
          EL = cube[15];
          DEK = epsFO-EK;
          DEL = epsFO-EL;
          ADEL = fabs(DEL);
          if( DEK*DEL<0 )
            {
              NSurfaces += 1;
              iEdge[NSurfaces-1] = 32;
              cuts[NSurfaces-1][0] = DTAU;
              cuts[NSurfaces-1][1] = 0.;
              cuts[NSurfaces-1][2] = DY;
              cuts[NSurfaces-1][3] = DEK/(EL-EK)*DETA;
            }


//______________________got all intersection points_________________________

          if (ELowerSum!=0)
            { 
              VLower0=VLower0/ELowerSum*DTAU;
              VLower1=VLower1/ELowerSum*DX;
              VLower2=VLower2/ELowerSum*DY;
              VLower3=VLower3/ELowerSum*DETA;
            }
          if (EHigherSum!=0)
            { 
              VHigher0=VHigher0/EHigherSum*DTAU;
              VHigher1=VHigher1/EHigherSum*DX;
              VHigher2=VHigher2/EHigherSum*DY;
              VHigher3=VHigher3/EHigherSum*DETA;
            }

          VD0 = VLower0-VHigher0;
          VD1 = VLower1-VHigher1;
          VD2 = VLower2-VHigher2;
          VD3 = VLower3-VHigher3;
        
          // compute the mean vector of intersection points:
          
          V0=0.0;
          V1=0.0;
          V2=0.0;
          V3=0.0;
 
          for (is=0; is<NSurfaces; is++)
            {
              V0+=cuts[is][0];
              V1+=cuts[is][1];
              V2+=cuts[is][2];
              V3+=cuts[is][3];
            }

          if (NSurfaces==0)
            continue;

          VMID[0]=V0/NSurfaces;
          VMID[1]=V1/NSurfaces;
          VMID[2]=V2/NSurfaces;
          VMID[3]=V3/NSurfaces;
 
          xf = x + VMID[1];
          yf = y + VMID[2];
          etaf = eta + VMID[3];
          tauf = tau - DTAU + VMID[0];
          
          if (tauf!=tauf)
            {
              cout << "ERROR: tauf nan!. exiting." << endl;
              cout << "DTAU=" << DTAU << endl;
              cout << "VMID[0]=" << VMID[0] << endl;
              cout << "VMID[1]=" << VMID[1] << endl;
              cout << "VMID[2]=" << VMID[2] << endl;
              cout << "VMID[3]=" << VMID[3] << endl;
              cout << "tau=" << tau << endl;
              cout << "V0=" << V0 << endl;
              cout << "NSurfaces=" << NSurfaces << endl;
              cout << "ieta=" << ieta << endl;
              cout << "neta-fac=" << neta-fac << endl;
              cout << (arena[ix+fac][iy+fac][ieta+fac].epsilon-epsFO)*(arena[ix][iy][ieta].epsilon_prev-epsFO) << endl;
              cout << (arena[ix+fac][iy][ieta].epsilon-epsFO)*(arena[ix][iy+fac][ieta+fac].epsilon_prev-epsFO) << endl;
              cout << (arena[ix][iy+fac][ieta].epsilon-epsFO)*(arena[ix+fac][iy][ieta+fac].epsilon_prev-epsFO) << endl;
              cout << (arena[ix][iy][ieta+fac].epsilon-epsFO)*(arena[ix+fac][iy+fac][ieta].epsilon_prev-epsFO) << endl;
              cout << (arena[ix+fac][iy+fac][ieta].epsilon-epsFO)*(arena[ix][iy][ieta+fac].epsilon_prev-epsFO) << endl;
              cout << (arena[ix+fac][iy][ieta+fac].epsilon-epsFO)*(arena[ix][iy+fac][ieta].epsilon_prev-epsFO) << endl;
              cout << (arena[ix][iy+fac][ieta+fac].epsilon-epsFO)*(arena[ix+fac][iy][ieta].epsilon_prev-epsFO) << endl;
              cout << (arena[ix][iy][ieta].epsilon-epsFO)*(arena[ix+fac][iy+fac][ieta+fac].epsilon_prev-epsFO) << endl;
              
              cout << "epsFO=" << epsFO*hbarc << endl;
              cout << "value=" << arena[ix+fac][iy+fac][ieta].epsilon_prev*hbarc << endl;
              cout << (arena[ix][iy][ieta+fac].epsilon-epsFO) << " " << (arena[ix+fac][iy+fac][ieta].epsilon_prev-epsFO) << endl;
              cout << "ix=" << ix << ", iy=" << iy << endl;
              exit(1);   
            }

          //fprintf(stderr, "NUMBER OF HYPERSURFACES = %d\n",NSurfaces);
          if (NSurfaces>12 || NSurfaces%2>0) fprintf(stderr, "********************NUMBER OF HYPERSURFACES = %d\n",NSurfaces);
          
          // find the neighbors
          tries3=0;
          COUNTER=0;
          while(((NSurfaces==10 && (COUNTER!=16 && COUNTER!=15)) || (NSurfaces==12 && (COUNTER!=19 && COUNTER!=20 && COUNTER!=21))
            ||(NSurfaces==4 && (COUNTER!=4)) || (NSurfaces==6 && (COUNTER!=8))
             ||(NSurfaces==8 && (COUNTER!=12))) && tries3<2*NSurfaces)
            {
         
              for (i=0; i<NSurfaces; i++)
            {
              for (j=0; j<6; j++)
                {
                  additionalNeighbors[i][j]=0;
                  neighborsDone[i][j]=0;
                }
            }
             
              for(is=0; is<32; is++)
            for(i=0; i<32; i++)
              numberConnectionIsUsed[is][i] = -1;
    
              //if (tries3>0) sleep(1);
              tries3+=1;
              //fprintf(stderr,"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++tries3=%d\n",tries3);
              for (m=0; m<NSurfaces; m++)
            {
              i=abs((m+tries3)%NSurfaces);
              IE = iEdge[i];
              IS = 0;
              for (m2=0; m2<NSurfaces; m2++)
                {
                  if (tries3<10) j=m2;//abs((m2+tries)%NSurfaces);
                  else 
                j=abs(m2+2*tries3)%NSurfaces;
                  JE = iEdge[j];
                  if ( IBIT[IE-1][JE-1]==1 ) 
                {
                  IS = IS+1;
//                ISID[IS-1] = j;
                  neighbors[i][IS-1]=JE; // determines the neighbors of ith edge (which is iEdge[i])
                }
                }
              for (m2=0; m2<NSurfaces; m2++)
                {
                  j=abs((m2+3*tries3)%NSurfaces);
                  JE = iEdge[j];
                  if ( IS<6 && IBIT[IE-1][JE-1]==2 ) 
                {
                  IS = IS+1;
//                ISID[IS-1] = j;
                  neighbors[i][IS-1]=JE; // determines the neighbors of ith edge (which is iEdge[i])
                }
                }
              if (IS!=3) 
                {
                  //sleep(1);
                  if (tries3>10)
                {
                  cerr << "WARNING: total number of neighbors=" << IS << endl;
                  break;
                }
                }
            }
              
              
              FULLSU[0]=0.;
              FULLSU[1]=0.;
              FULLSU[2]=0.;
              FULLSU[3]=0.;
              
          
              // do the first group of tetraedra that do not share any edges:
              COUNTER=0;
              countEdges=0;
              for (is=0; is<NSurfaces; is++)
            {
              skip=0;
              if (is>0)
                for (j=0; j<countEdges; j++)
                  {
                if (iEdge[is]==neighbors[previousEdges[j]][0] || iEdge[is]==neighbors[previousEdges[j]][1] 
                    || iEdge[is]==neighbors[previousEdges[j]][2])
                  {
                    //fprintf(stderr,"************skipping edge %d\n",iEdge[is]);
                    skip=1;
                    break;
                  }
                  }
              if (skip==1) continue;
              // fprintf(stderr,"is=%d, doing edge %d\n",is, iEdge[is]);
              
              previousEdges[countEdges]=is;
              countEdges+=1;
              IS=0;
              for(k=0;k<4;k++)
                {
                  BD[k] = VMID[k]-cuts[is][k];
                }
              for (j=0; j<3; j++)
                {
                  for (i=0; i<NSurfaces; i++)
                {
                  if (neighbors[is][j]==iEdge[i])
                    {
                      neighborsDone[is][j]=1;
                      for (l=0; l<3; l++)
                    {
                      if (neighbors[i][l] == iEdge[is]) 
                        {
                          neighborsDone[i][l]=1;
                        }
                    }
                      IS = IS+1;
                      prevEdge[IS-1]=iEdge[i];
                      if (iEdge[is]>iEdge[i])
                    numberConnectionIsUsed[iEdge[is]-1][iEdge[i]-1]+=3;
                      else
                    numberConnectionIsUsed[iEdge[i]-1][iEdge[is]-1]+=3;
                      
                      
                      for(k=0;k<4;k++)
                    {
                      AD[IS-1][k] = cuts[is][k]-cuts[i][k];
                    }
                    }
                }
                }
              // update the number of connections for every used edge:
              for (m=0; m<IS; m++)
                {
                  for (m2=0; m2<m; m2++)
                {
                  if (prevEdge[m]>prevEdge[m2])
                    if (numberConnectionIsUsed[prevEdge[m]-1][prevEdge[m2]-1]==-1)
                      numberConnectionIsUsed[prevEdge[m]-1][prevEdge[m2]-1]+=2;
                    else
                      numberConnectionIsUsed[prevEdge[m]-1][prevEdge[m2]-1]+=1;
                  else
                    if (numberConnectionIsUsed[prevEdge[m2]-1][prevEdge[m]-1]==-1)
                      numberConnectionIsUsed[prevEdge[m2]-1][prevEdge[m]-1]+=2;
                    else
                      numberConnectionIsUsed[prevEdge[m2]-1][prevEdge[m]-1]+=1;
                }
                }
              
              if (IS==3)
                {
                  COUNTER+=3;
                  
                  SU[COUNTER-3][0] = AD[0][1]*AD[1][2]*BD[3] - AD[0][1]*AD[1][3]*BD[2] + AD[0][3]*AD[1][1]*BD[2]
                -  AD[0][3]*AD[1][2]*BD[1] + AD[0][2]*AD[1][3]*BD[1] - AD[0][2]*AD[1][1]*BD[3];
                  SU[COUNTER-3][1] = AD[0][2]*AD[1][0]*BD[3] - AD[0][0]*AD[1][2]*BD[3] + AD[0][0]*AD[1][3]*BD[2]
                -  AD[0][3]*AD[1][0]*BD[2] + AD[0][3]*AD[1][2]*BD[0] - AD[0][2]*AD[1][3]*BD[0];
                  SU[COUNTER-3][2] = AD[0][0]*AD[1][1]*BD[3] - AD[0][0]*AD[1][3]*BD[1] + AD[0][3]*AD[1][0]*BD[1]
                -  AD[0][3]*AD[1][1]*BD[0] + AD[0][1]*AD[1][3]*BD[0] - AD[0][1]*AD[1][0]*BD[3];
                  SU[COUNTER-3][3] = AD[0][2]*AD[1][1]*BD[0] - AD[0][1]*AD[1][2]*BD[0] + AD[0][0]*AD[1][2]*BD[1]
                -  AD[0][2]*AD[1][0]*BD[1] + AD[0][1]*AD[1][0]*BD[2] - AD[0][0]*AD[1][1]*BD[2];
                  
                  SU[COUNTER-2][0] = AD[0][1]*AD[2][2]*BD[3] - AD[0][1]*AD[2][3]*BD[2] + AD[0][3]*AD[2][1]*BD[2]
                -  AD[0][3]*AD[2][2]*BD[1] + AD[0][2]*AD[2][3]*BD[1] - AD[0][2]*AD[2][1]*BD[3];
                  SU[COUNTER-2][1] = AD[0][2]*AD[2][0]*BD[3] - AD[0][0]*AD[2][2]*BD[3] + AD[0][0]*AD[2][3]*BD[2]
                -  AD[0][3]*AD[2][0]*BD[2] + AD[0][3]*AD[2][2]*BD[0] - AD[0][2]*AD[2][3]*BD[0];
                  SU[COUNTER-2][2] = AD[0][0]*AD[2][1]*BD[3] - AD[0][0]*AD[2][3]*BD[1] + AD[0][3]*AD[2][0]*BD[1]
                -  AD[0][3]*AD[2][1]*BD[0] + AD[0][1]*AD[2][3]*BD[0] - AD[0][1]*AD[2][0]*BD[3];
                  SU[COUNTER-2][3] = AD[0][2]*AD[2][1]*BD[0] - AD[0][1]*AD[2][2]*BD[0] + AD[0][0]*AD[2][2]*BD[1]
                -  AD[0][2]*AD[2][0]*BD[1] + AD[0][1]*AD[2][0]*BD[2] - AD[0][0]*AD[2][1]*BD[2];
                  
                  SU[COUNTER-1][0] = AD[1][1]*AD[2][2]*BD[3] - AD[1][1]*AD[2][3]*BD[2] + AD[1][3]*AD[2][1]*BD[2]
                -  AD[1][3]*AD[2][2]*BD[1] + AD[1][2]*AD[2][3]*BD[1] - AD[1][2]*AD[2][1]*BD[3];
                  SU[COUNTER-1][1] = AD[1][2]*AD[2][0]*BD[3] - AD[1][0]*AD[2][2]*BD[3] + AD[1][0]*AD[2][3]*BD[2]
                -  AD[1][3]*AD[2][0]*BD[2] + AD[1][3]*AD[2][2]*BD[0] - AD[1][2]*AD[2][3]*BD[0];
                  SU[COUNTER-1][2] = AD[1][0]*AD[2][1]*BD[3] - AD[1][0]*AD[2][3]*BD[1] + AD[1][3]*AD[2][0]*BD[1]
                -  AD[1][3]*AD[2][1]*BD[0] + AD[1][1]*AD[2][3]*BD[0] - AD[1][1]*AD[2][0]*BD[3];
                  SU[COUNTER-1][3] = AD[1][2]*AD[2][1]*BD[0] - AD[1][1]*AD[2][2]*BD[0] + AD[1][0]*AD[2][2]*BD[1]
                -  AD[1][2]*AD[2][0]*BD[1] + AD[1][1]*AD[2][0]*BD[2] - AD[1][0]*AD[2][1]*BD[2];
                  
                  for (k=0; k<4;k++)
                {
                  SU[COUNTER-3][k]/=6.; // to get tetraedron's volume
                  SU[COUNTER-2][k]/=6.; // to get tetraedron's volume
                  SU[COUNTER-1][k]/=6.; // to get tetraedron's volume
                }
                  
                }
            }
              
              for(j=0; j<32; j++)
            {
              singleConnections[j][0]=0;
              singleConnections[j][1]=0;
              singleConnectionsUsed[j]=0;
            }

              countSingleEdges=0;
              for(j=1; j<=32; j++)
            for (k=1; k<j; k++)
              {
                if ( numberConnectionIsUsed[j-1][k-1]>-1 )
                  {
                // fprintf(stderr,"connection between %d and %d is used %d times.\n", j, k, numberConnectionIsUsed[j-1][k-1]);
                if (numberConnectionIsUsed[j-1][k-1]<2)
                  {
                    if (numberConnectionIsUsed[j-1][k-1]==0)
                      countSingleEdges+=2;
                    else
                      countSingleEdges+=1;
                    singleConnections[countSingleEdges-1][0]=j;
                    singleConnections[countSingleEdges-1][1]=k;
                    //fprintf(stderr,"-------------------------------------------\n");
                  }
                if (numberConnectionIsUsed[j-1][k-1]>2)
                  {
                    //fprintf(stderr,"connection between %d and %d is used %d times.\n", j, k, numberConnectionIsUsed[j-1][k-1]);
                    //fprintf(stderr,"Three connections is messed up. That can't happen.\n");
                    break;
                  }
                  }
              }

                 
              // find additional connections between edges of the previously defined tetraedra
              additionalEdges=1; // just to get started in the while loop at all
              // do the missed edges:
              countSingleEdges=1; // just to get started in the while loop at all
              shift=0;
              //addCOUNTER=0;
//            tries2=0;
           /*    for(j=1; j<=32; j++) */
/*          for (k=1; k<j; k++) */
/*            { */
/*              if ( numberConnectionIsUsed[j-1][k-1]>-1 ) */
/*                { */
/*              //fprintf(stderr,"BEFORE - connection between %d and %d is used %d times.\n", j, k, numberConnectionIsUsed[j-1][k-1]); */
/*              if (numberConnectionIsUsed[j-1][k-1]<2) */
/*                { */
/*                  fprintf(stderr,"-------------------------------------------\n"); */
/*                } */
/*                } */
/*            } */
              
              for (i=0; i<NSurfaces; i++)
            {
              for (j=0; j<3; j++)
                {
                  if (additionalNeighbors[i][j]==1) neighborsDone[i][j]=0;
                  additionalNeighbors[i][j]=0;
                }
            }
              //fprintf(stderr,"NUMBER OF SURFACES=%d\n",NSurfaces);
              //fprintf(stderr,"COUNTER=%d\n",COUNTER);
              //fprintf(stderr,"additionalEdges=%d\n",additionalEdges);
              //fprintf(stderr,"countSingleEdges=%d\n",countSingleEdges);
              //fprintf(stderr,"shift=%d\n",shift);
              additionalEdges=0;
              IS=0;
              for (m=0; m<NSurfaces; m++)
            {
              is = abs((m-shift)%NSurfaces);
              for (j=0; j<3; j++)
                {
                  //fprintf(stderr,"%d's neighbor %d done = %d\n",iEdge[is],neighbors[is][j],neighborsDone[is][j]);
                  if (neighborsDone[is][j] == 0)
                {
                  for (m2=0; m2<NSurfaces; m2++)
                    {
                      i = (tries3+shift+m2)%NSurfaces;
                      for (l=0; l<3; l++)
                    {
                      if (neighborsDone[i][l] == 0 && neighbors[is][j]==iEdge[i] && neighbors[i][l]==iEdge[is])
                        {
                          additionalEdges+=1;
                          additionalNeighbors[is][j]=1;
                          additionalNeighbors[i][j]=1;
                          neighborsDone[is][j]=1;
                          neighborsDone[i][l]=1;
                          IS=IS+1;
                          edge1[IS-1]=iEdge[is];
                          edge2[IS-1]=iEdge[i];
                          //edge1Done[IS-1]=0;
                          //edge2Done[IS-1]=0;
                          //fprintf(stderr,"doing for edge %d and %d\n",iEdge[is],iEdge[i]);
                          //fprintf(stderr,"is=%d, i=%d, IS=%d\n",is,i,IS);
                          //fprintf(stderr,"edge1=%d, edge2=%d \n",edge1[IS-1],edge2[IS-1]);
                          
                          for(k=0;k<4;k++)
                        {
                          DD[IS-1][k] = VMID[k]-cuts[is][k];
                          AD[IS-1][k] = cuts[is][k]-cuts[i][k];
                        }
                        }
                    }
                    }
                }
                }
            }
              
              countAdditionalEdges=0;
              //fprintf(stderr,"IS=%d\n",IS);
              
              
              for (m=0; m<IS; m++)
            {
              srand(time(NULL));
              if (tries3<5) i=m;
              else
                i=abs(m-tries3*2)%IS;
              for (m2=0; m2<IS; m2++)
                {
                  if (tries3==0) j=m2;
                  else
                j=abs(m2+tries3)%IS;
                  //fprintf(stderr,"j=%d\n",j);
                  if ( additionalEdges>0 &&
                   (
                    ((edge1[i] == edge2[j]) || (edge2[i] == edge1[j]) || (edge1[i] == edge1[j])|| (edge2[i] == edge2[j]))))
                {
                  if (edge1[i] == edge1[j] && edge2[i] == edge2[j]) continue;
                  if (edge1[i] == edge2[j] && edge2[i] == edge1[j]) continue;
                  
                  usedEdges1[countAdditionalEdges]=edge1[i];
                  usedEdges2[countAdditionalEdges]=edge1[j];
                  usedEdges3[countAdditionalEdges]=edge2[i];
                  usedEdges4[countAdditionalEdges]=edge2[j];
                  
                  // sort them
                  if (usedEdges3[countAdditionalEdges]>usedEdges4[countAdditionalEdges])
                    {
                      temp = usedEdges3[countAdditionalEdges];
                      usedEdges3[countAdditionalEdges] = usedEdges4[countAdditionalEdges];
                      usedEdges4[countAdditionalEdges] = temp;
                    }
                  if (usedEdges2[countAdditionalEdges]>usedEdges3[countAdditionalEdges])
                    {
                      temp = usedEdges2[countAdditionalEdges];
                      usedEdges2[countAdditionalEdges] = usedEdges3[countAdditionalEdges];
                      usedEdges3[countAdditionalEdges] = temp;
                    }
                  if (usedEdges1[countAdditionalEdges]>usedEdges2[countAdditionalEdges])
                    {
                      temp = usedEdges1[countAdditionalEdges];
                      usedEdges1[countAdditionalEdges] = usedEdges2[countAdditionalEdges];
                      usedEdges2[countAdditionalEdges] = temp;
                    }
                  
                  // remove double one
                  if (usedEdges2[countAdditionalEdges] == usedEdges3[countAdditionalEdges])
                    usedEdges3[countAdditionalEdges] = usedEdges4[countAdditionalEdges];
                  if (usedEdges1[countAdditionalEdges] == usedEdges2[countAdditionalEdges])
                    {
                      usedEdges2[countAdditionalEdges] = usedEdges3[countAdditionalEdges];
                      usedEdges3[countAdditionalEdges] = usedEdges4[countAdditionalEdges];
                    }
                  
                 /*  fprintf(stderr,"usedEdges1[%d]=%d\n",countAdditionalEdges, usedEdges1[countAdditionalEdges]); */
/*                fprintf(stderr,"usedEdges2[%d]=%d\n",countAdditionalEdges, usedEdges2[countAdditionalEdges]); */
/*                fprintf(stderr,"usedEdges3[%d]=%d\n",countAdditionalEdges, usedEdges3[countAdditionalEdges]); */
/*                fprintf(stderr,"usedEdges4[%d]=%d\n",countAdditionalEdges, usedEdges4[countAdditionalEdges]); */
/*                fprintf(stderr,"edge1[%d]=%d\n",i, edge1[i]); */
/*                fprintf(stderr,"edge2[%d]=%d\n",i, edge2[i]); */
/*                fprintf(stderr,"edge1[%d]=%d\n",j, edge1[j]); */
/*                fprintf(stderr,"edge2[%d]=%d\n",j, edge2[j]); */
                  
                  if (countAdditionalEdges>0)
                    {
                      for (l=0; l<countAdditionalEdges; l++)
                    {
                      skip = 0;
                      if ( usedEdges1[countAdditionalEdges] == usedEdges1[l]
                           && usedEdges2[countAdditionalEdges] == usedEdges2[l]
                           && usedEdges3[countAdditionalEdges] == usedEdges3[l]
                           )
                        {
                          //fprintf(stderr,"usedEdges1[%d]=%d\n",countAdditionalEdges, usedEdges1[countAdditionalEdges]);
                          //fprintf(stderr,"usedEdges2[%d]=%d\n",countAdditionalEdges, usedEdges2[countAdditionalEdges]);
                          //fprintf(stderr,"usedEdges3[%d]=%d\n",countAdditionalEdges, usedEdges3[countAdditionalEdges]);
                          // fprintf(stderr,"usedEdges1[%d]=%d\n",l, usedEdges1[l]);
                          //fprintf(stderr,"usedEdges2[%d]=%d\n",l, usedEdges2[l]);
                          //fprintf(stderr,"usedEdges3[%d]=%d\n",l, usedEdges3[l]);
                          skip=1;
                          //      fprintf(stderr,"skipping surface\n");
                          break;
                        }
                    }
                      if (skip==1) continue;
                    }
                  
                  skip=0;
                  countAdditionalEdges+=1;
                  //fprintf(stderr,"BEFORE adding\n");
                      if (edge1[i]!=edge2[i]) 
                    {
                      //fprintf(stderr,"1 adding 1 for %d and %d\n",edge1[i], edge2[i]);
                      if (edge1[i]>edge2[i])
                        {
                          if (numberConnectionIsUsed[edge1[i]-1][edge2[i]-1]==2) continue;
                          if (numberConnectionIsUsed[edge1[i]-1][edge2[i]-1]==-1)
                        numberConnectionIsUsed[edge1[i]-1][edge2[i]-1]+=2;
                          else
                        numberConnectionIsUsed[edge1[i]-1][edge2[i]-1]+=1;
                        }
                      else
                        {
                          if (numberConnectionIsUsed[edge2[i]-1][edge1[i]-1]==2) continue;
                          if (numberConnectionIsUsed[edge2[i]-1][edge1[i]-1]==-1)
                        numberConnectionIsUsed[edge2[i]-1][edge1[i]-1]+=2;
                          else
                        numberConnectionIsUsed[edge2[i]-1][edge1[i]-1]+=1;
                        }
                    }
                      if (edge1[j]!=edge2[i])
                    if (edge1[i]!=edge1[j]) 
                      {
                        //fprintf(stderr,"2 adding 1 for %d and %d\n",edge1[i], edge1[j]);
                        if(edge1[i]>edge1[j])
                          {
                        if (numberConnectionIsUsed[edge1[i]-1][edge1[j]-1]==2) continue;
                        if (numberConnectionIsUsed[edge1[i]-1][edge1[j]-1]==-1)
                          numberConnectionIsUsed[edge1[i]-1][edge1[j]-1]+=2;
                        else
                          numberConnectionIsUsed[edge1[i]-1][edge1[j]-1]+=1;
                          }
                        else
                          {
                        if (numberConnectionIsUsed[edge1[j]-1][edge1[i]-1]==2) continue;
                        if (numberConnectionIsUsed[edge1[j]-1][edge1[i]-1]==-1)
                          numberConnectionIsUsed[edge1[j]-1][edge1[i]-1]+=2;
                        else
                          numberConnectionIsUsed[edge1[j]-1][edge1[i]-1]+=1;
                          }  
                      }
                      
                      if (edge1[j]!=edge2[j])
                    {           
                      //fprintf(stderr,"3 adding 1 for %d and %d\n",edge1[j], edge2[j]);
                      if(edge1[j]>edge2[j])
                        {
                          if (numberConnectionIsUsed[edge1[j]-1][edge2[j]-1]==2) continue;
                          if (numberConnectionIsUsed[edge1[j]-1][edge2[j]-1]==-1)
                        numberConnectionIsUsed[edge1[j]-1][edge2[j]-1]+=2;
                          else
                        numberConnectionIsUsed[edge1[j]-1][edge2[j]-1]+=1;
                        }
                      else
                        {
                          if (numberConnectionIsUsed[edge2[j]-1][edge1[j]-1]==2) continue;
                          if (numberConnectionIsUsed[edge2[j]-1][edge1[j]-1]==-1)
                        numberConnectionIsUsed[edge2[j]-1][edge1[j]-1]+=2;
                          else
                        numberConnectionIsUsed[edge2[j]-1][edge1[j]-1]+=1;
                        }
                    }
                      
                      if (edge1[j]!=edge2[i])
                    if (edge2[i]!=edge2[j]) 
                      {
                        //fprintf(stderr,"4 adding 1 for %d and %d\n",edge2[i], edge2[j]);
                        if(edge2[i]>edge2[j])
                          {
                        if (numberConnectionIsUsed[edge2[i]-1][edge2[j]-1]==2) continue;
                        if (numberConnectionIsUsed[edge2[i]-1][edge2[j]-1]==-1)
                          numberConnectionIsUsed[edge2[i]-1][edge2[j]-1]+=2;
                        else
                          numberConnectionIsUsed[edge2[i]-1][edge2[j]-1]+=1;
                          }
                        else
                          {
                        if (numberConnectionIsUsed[edge2[j]-1][edge2[i]-1]==2) continue;
                        if (numberConnectionIsUsed[edge2[j]-1][edge2[i]-1]==-1)
                          numberConnectionIsUsed[edge2[j]-1][edge2[i]-1]+=2;
                        else
                          numberConnectionIsUsed[edge2[j]-1][edge2[i]-1]+=1;
                          }
                      }
                      
                      if (edge2[j]!=edge1[j] && edge2[j]!=edge2[i] && edge1[i]!=edge1[j])
                    if (edge1[i]!=edge2[j]) 
                      {
                        //fprintf(stderr,"5 adding 1 for %d and %d\n",edge1[i], edge2[j]);
                        if(edge1[i]>edge1[j])
                          {
                        if (numberConnectionIsUsed[edge1[i]-1][edge2[j]-1]==2) continue;
                        if (numberConnectionIsUsed[edge1[i]-1][edge2[j]-1]==-1)
                          numberConnectionIsUsed[edge1[i]-1][edge2[j]-1]+=2;
                        else
                          numberConnectionIsUsed[edge1[i]-1][edge2[j]-1]+=1;
                          }
                        else
                          {
                        if (numberConnectionIsUsed[edge2[j]-1][edge1[i]-1]==2) continue;
                        if (numberConnectionIsUsed[edge2[j]-1][edge1[i]-1]==-1)
                          numberConnectionIsUsed[edge2[j]-1][edge1[i]-1]+=2;
                        else
                          numberConnectionIsUsed[edge2[j]-1][edge1[i]-1]+=1;
                          }
                      }
                      
                      if (edge1[j]!=edge2[j] && edge2[i]!=edge1[i] && edge2[i]!=edge2[j] && edge1[j]!=edge1[i])
                    if (edge1[j]!=edge2[i])
                      {
                        //fprintf(stderr,"6 adding 1 for %d and %d\n",edge1[j], edge2[i]);
                        if(edge1[j]>edge2[i])
                          {
                        if (numberConnectionIsUsed[edge1[j]-1][edge2[i]-1]==2) continue;
                        if (numberConnectionIsUsed[edge1[j]-1][edge2[i]-1]==-1)
                          numberConnectionIsUsed[edge1[j]-1][edge2[i]-1]+=2;
                        else
                          numberConnectionIsUsed[edge1[j]-1][edge2[i]-1]+=1;
                          }
                        else
                          {                   
                        if (numberConnectionIsUsed[edge2[i]-1][edge1[j]-1]==2) continue;
                        if (numberConnectionIsUsed[edge2[i]-1][edge1[j]-1]==-1)
                          numberConnectionIsUsed[edge2[i]-1][edge1[j]-1]+=2;
                        else
                          numberConnectionIsUsed[edge2[i]-1][edge1[j]-1]+=1;
                          }
                      }
                      
                      COUNTER+=1;
                      //addCOUNTER+=1;
                      //fprintf(stderr,"counter=%d\n",COUNTER);
                      //fprintf(stderr,"edge1[%d]=%d\n",i,edge1[i]);
                      //fprintf(stderr,"edge2[%d]=%d\n",i,edge2[i]);
                      //fprintf(stderr,"edge1[%d]=%d\n",j,edge1[j]);
                      //fprintf(stderr,"edge2[%d]=%d\n",j,edge2[j]);
                      SU[COUNTER-1][0] = AD[i][1]*AD[j][2]*DD[i][3] - AD[i][1]*AD[j][3]*DD[i][2] + AD[i][3]*AD[j][1]*DD[i][2]
                    -  AD[i][3]*AD[j][2]*DD[i][1] + AD[i][2]*AD[j][3]*DD[i][1] - AD[i][2]*AD[j][1]*DD[i][3];
                      SU[COUNTER-1][1] = AD[i][2]*AD[j][0]*DD[i][3] - AD[i][0]*AD[j][2]*DD[i][3] + AD[i][0]*AD[j][3]*DD[i][2]
                    -  AD[i][3]*AD[j][0]*DD[i][2] + AD[i][3]*AD[j][2]*DD[i][0] - AD[i][2]*AD[j][3]*DD[i][0];
                      SU[COUNTER-1][2] = AD[i][0]*AD[j][1]*DD[i][3] - AD[i][0]*AD[j][3]*DD[i][1] + AD[i][3]*AD[j][0]*DD[i][1]
                    -  AD[i][3]*AD[j][1]*DD[i][0] + AD[i][1]*AD[j][3]*DD[i][0] - AD[i][1]*AD[j][0]*DD[i][3];
                      SU[COUNTER-1][3] = AD[i][2]*AD[j][1]*DD[i][0] - AD[i][1]*AD[j][2]*DD[i][0] + AD[i][0]*AD[j][2]*DD[i][1]
                    -  AD[i][2]*AD[j][0]*DD[i][1] + AD[i][1]*AD[j][0]*DD[i][2] - AD[i][0]*AD[j][1]*DD[i][2];
                      
                      for (k=0; k<4; k++)
                    {
                      SU[COUNTER-1][k]/=6.; // to get tetraedron's volume
                    }
                }
                }
            }
              
/*            countSingleEdges=0; */
/*            for(j=1; j<=32; j++) */
/*          for (k=1; k<j; k++) */
/*            { */
/*              if ( numberConnectionIsUsed[j-1][k-1]>-1 ) */
/*                { */
/*              //fprintf(stderr,"connection between %d and %d is used %d times.\n", j, k, numberConnectionIsUsed[j-1][k-1]); */
/*              if (numberConnectionIsUsed[j-1][k-1]<2) */
/*                { */
/*                  if (numberConnectionIsUsed[j-1][k-1]==0) */
/*                    countSingleEdges+=2; */
/*                  else */
/*                    countSingleEdges+=1; */
/*                  singleConnections[countSingleEdges-1][0]=j; */
/*                  singleConnections[countSingleEdges-1][1]=k; */
/*                  //fprintf(stderr,"-------------------------------------------\n"); */
/*                } */
/*              if (numberConnectionIsUsed[j-1][k-1]>2) */
/*                { */
/*                  fprintf(stderr,"connection between %d and %d is used %d times.\n", j, k, numberConnectionIsUsed[j-1][k-1]); */
/*                  fprintf(stderr,"Three connections is messed up. That can't happen.\n"); */
/*                  exit(1); */
/*                } */
/*                } */
/*            } */


              for(j=0; j<32; j++)
            {
              singleConnections[j][0]=0;
              singleConnections[j][1]=0;
              singleConnectionsUsed[j]=0;
            }
              
              
              countSingleEdges=0;
              for(j=1; j<=32; j++)
                for (k=1; k<j; k++)
                  {
                if ( numberConnectionIsUsed[j-1][k-1]>-1 )
                  {
                    //fprintf(stderr,"connection between %d and %d is used %d times.\n", j, k, numberConnectionIsUsed[j-1][k-1]);
                    if (numberConnectionIsUsed[j-1][k-1]<2)
                      {
                    if (numberConnectionIsUsed[j-1][k-1]==0)
                      countSingleEdges+=2;
                    else
                      countSingleEdges+=1;
                    singleConnections[countSingleEdges-1][0]=j;
                    singleConnections[countSingleEdges-1][1]=k;
                    //fprintf(stderr,"-------------------------------------------\n");
                      }
                    if (numberConnectionIsUsed[j-1][k-1]>2)
                      {
                    //fprintf(stderr,"connection between %d and %d is used %d times.\n", j, k, numberConnectionIsUsed[j-1][k-1]);
                    //fprintf(stderr,"Three connections is messed up. That can't happen.\n");
                    break;
                      }
                  }
                  }
              
              
              /*              countSingleEdges=0; */
              /*              for(j=1; j<=32; j++) */
              /*            for (k=1; k<j; k++) */
              /*              { */
              /*                if ( numberConnectionIsUsed[j-1][k-1]>0 ) */
              /*                  { */
              /*                if (numberConnectionIsUsed[j-1][k-1]<2) */
              /*                  { */
              /*                    countSingleEdges+=1; */
              /*                    singleConnections[countSingleEdges-1][0]=j; */
              /*                    singleConnections[countSingleEdges-1][1]=k; */
              /*                  } */
              /*                  } */
              /*              } */
              
              shift+=1;
              
              
              if (countSingleEdges>0)
                {
                  if (1==1)//countSingleEdges%3==0) //if we have a number of single edges that is a multiple of 3
                {
                  IS=0;
                  for (j=0; j<countSingleEdges; j++)
                    {
                      if(NSurfaces==12)
                    {
                      // fprintf(stderr,"singleConnections[%d][0]=%d\n",j,singleConnections[j][0]);
                      //fprintf(stderr,"singleConnections[%d][1]=%d\n",j,singleConnections[j][1]);
                    }
                      for (i=0; i<countSingleEdges; i++)
                    {
                      if (i!=j && singleConnectionsUsed[j]==0 && singleConnectionsUsed[i]==0 )
                        {
                          for ( k=0; k<countSingleEdges; k++ )
                        {
                          if(k!=j && k!=i)
                            {
                              if ( ((singleConnections[j][0] == singleConnections[i][0]
                                 && singleConnections[j][1] == singleConnections[k][0]
                                 && singleConnections[k][1] == singleConnections[i][1])
                                ||
                                (singleConnections[j][0] == singleConnections[i][1]
                                 && singleConnections[j][1] == singleConnections[k][0]
                                 && singleConnections[k][1] == singleConnections[i][0])
                                ||
                                (singleConnections[j][0] == singleConnections[i][0]
                                 && singleConnections[j][1] == singleConnections[k][1]
                                 && singleConnections[k][0] == singleConnections[i][1])
                                ||
                                (singleConnections[j][0] == singleConnections[i][1]
                                 && singleConnections[j][1] == singleConnections[k][1]
                                 && singleConnections[k][0] == singleConnections[i][0])
                                ||
                                (singleConnections[j][1] == singleConnections[i][0]
                                 && singleConnections[j][0] == singleConnections[k][0]
                                 && singleConnections[k][1] == singleConnections[i][1])
                                ||
                                (singleConnections[j][1] == singleConnections[i][1]
                                 && singleConnections[j][0] == singleConnections[k][0]
                                 && singleConnections[k][1] == singleConnections[i][0])
                                ||
                                (singleConnections[j][1] == singleConnections[i][0]
                                 && singleConnections[j][0] == singleConnections[k][1]
                                 && singleConnections[k][0] == singleConnections[i][1])
                                ||
                                (singleConnections[j][1] == singleConnections[i][1]
                                 && singleConnections[j][0] == singleConnections[k][1]
                                 && singleConnections[k][0] == singleConnections[i][0])
                                )&& singleConnectionsUsed[k]==0 )
                            {
                              singleConnectionsUsed[k]+=1;
                              singleConnectionsUsed[i]+=1;
                              singleConnectionsUsed[j]+=1;
                              group[IS][0]=singleConnections[j][0];
                              group[IS][1]=singleConnections[j][1];
                              if (singleConnections[i][0]!=singleConnections[j][0] 
                              && singleConnections[i][0]!=singleConnections[j][1] )
                            group[IS][2]=singleConnections[i][0];
                              else
                            group[IS][2]=singleConnections[i][1];
                              IS+=1;
                            }
                            }
                        }
                        }
                    }
                    }
                  
                  if (IS!=countSingleEdges/3) 
                    {
                      srand(time(NULL));
                      m=0;
                      tries=0;
                      while(IS!=countSingleEdges/3 && tries<5) 
                    {
                      tries+=1;
                      for(j=0; j<32; j++)
                        {
                          singleConnectionsUsed[j]=0;
                        }
                      //fprintf(stderr,"MISSED SOME TETRAEDRA. got %d out of %d\n",IS, countSingleEdges/3);
                      IS=0;
                      for (j=0; j<countSingleEdges; j++)
                        {
                          m=abs(rand()%countSingleEdges);
                          //fprintf(stderr,"singleConnections[%d][0]=%d\n",j,singleConnections[j][0]);
                          //fprintf(stderr,"singleConnections[%d][1]=%d\n",j,singleConnections[j][1]);
                          for (l=0; l<countSingleEdges; l++)
                        {
                          i=(l+m)%countSingleEdges;
                          //RANDOM ORDERING     
                          if (i!=j && singleConnectionsUsed[j]==0 && singleConnectionsUsed[i]==0 )
                            {
                              for ( k=0; k<countSingleEdges; k++ )
                            {
                              if(k!=j && k!=i)
                                {
                                  if ( ((singleConnections[j][0] == singleConnections[i][0]
                                     && singleConnections[j][1] == singleConnections[k][0]
                                     && singleConnections[k][1] == singleConnections[i][1])
                                    ||
                                    (singleConnections[j][0] == singleConnections[i][1]
                                     && singleConnections[j][1] == singleConnections[k][0]
                                     && singleConnections[k][1] == singleConnections[i][0])
                                    ||
                                    (singleConnections[j][0] == singleConnections[i][0]
                                     && singleConnections[j][1] == singleConnections[k][1]
                                     && singleConnections[k][0] == singleConnections[i][1])
                                    ||
                                    (singleConnections[j][0] == singleConnections[i][1]
                                     && singleConnections[j][1] == singleConnections[k][1]
                                     && singleConnections[k][0] == singleConnections[i][0])
                                    ||
                                    (singleConnections[j][1] == singleConnections[i][0]
                                     && singleConnections[j][0] == singleConnections[k][0]
                                     && singleConnections[k][1] == singleConnections[i][1])
                                    ||
                                    (singleConnections[j][1] == singleConnections[i][1]
                                     && singleConnections[j][0] == singleConnections[k][0]
                                     && singleConnections[k][1] == singleConnections[i][0])
                                    ||
                                    (singleConnections[j][1] == singleConnections[i][0]
                                     && singleConnections[j][0] == singleConnections[k][1]
                                     && singleConnections[k][0] == singleConnections[i][1])
                                    ||
                                    (singleConnections[j][1] == singleConnections[i][1]
                                     && singleConnections[j][0] == singleConnections[k][1]
                                     && singleConnections[k][0] == singleConnections[i][0])
                                    )&& singleConnectionsUsed[k]==0 )
                                {
                                  singleConnectionsUsed[k]+=1;
                                  singleConnectionsUsed[i]+=1;
                                  singleConnectionsUsed[j]+=1;
                                  group[IS][0]=singleConnections[j][0];
                                  group[IS][1]=singleConnections[j][1];
                                  if (singleConnections[i][0]!=singleConnections[j][0] 
                                      && singleConnections[i][0]!=singleConnections[j][1] )
                                    group[IS][2]=singleConnections[i][0];
                                  else
                                    group[IS][2]=singleConnections[i][1];
                                  IS+=1;
                                  //fprintf(stderr,"IS=%d\n",IS);
                                }
                                }
                            }
                            }
                        }
                        }
                    /*   for (j=0; j<IS; j++) */
/*                      { */
/*                        fprintf(stderr,"group[%d]=(%d,%d,%d)\n",j,group[j][0],group[j][1],group[j][2]); */
/*                      } */
                    }
                      //sleep(50);
                    }
                }
                  
                  if (countSingleEdges>0)
                {
                  if (1==1)//countSingleEdges%3==0) //if we have a number of single edges that is a multiple of 3
                    {
                      for (j=0; j<IS; j++)
                    {
                      skip=0;
                      for(is=0; is<NSurfaces; is++)
                        {
                          if (iEdge[is]==group[j][0])
                        {
                          for(i=0; i<NSurfaces; i++)
                            {
                              if (iEdge[i]==group[j][1])
                            {
                              if(iEdge[is]>iEdge[i])
                                numberConnectionIsUsed[iEdge[is]-1][iEdge[i]-1]+=1;
                              else
                                numberConnectionIsUsed[iEdge[i]-1][iEdge[is]-1]+=1;
                              if(iEdge[is]>group[j][2])
                                numberConnectionIsUsed[iEdge[is]-1][group[j][2]-1]+=1;
                              else
                                numberConnectionIsUsed[group[j][2]-1][iEdge[is]-1]+=1;
                              if(iEdge[i]>group[j][2])
                                numberConnectionIsUsed[iEdge[i]-1][group[j][2]-1]+=1;
                              else
                                numberConnectionIsUsed[group[j][2]-1][iEdge[i]-1]+=1;
                              
                              //fprintf(stderr,"IN THE LOOP 1 and j=%d\n",j);
                              //fprintf(stderr,"edges= %d, %d, %d\n",iEdge[is], iEdge[i], group[j][2]);
                              for(k=0;k<4;k++)
                                {
                                  AD[0][k] = cuts[is][k]-cuts[i][k];
                                  BD[k] = VMID[k]-cuts[is][k];
                                }
                              skip=1;
                              continue;
                            }
                            }
                          for(i=0; i<NSurfaces; i++)
                            {
                              if (iEdge[i]==group[j][2])
                            {
                              for(k=0;k<4;k++)
                                {
                                  AD[1][k] = cuts[is][k]-cuts[i][k];
                                }
                              skip=1;
                              continue;
                            }
                            }
                        }
                          if ( skip==1 ) continue;
                        }
                      COUNTER+=1;
                      SU[COUNTER-1][0] = AD[0][1]*AD[1][2]*BD[3] - AD[0][1]*AD[1][3]*BD[2] + AD[0][3]*AD[1][1]*BD[2]
                        -  AD[0][3]*AD[1][2]*BD[1] + AD[0][2]*AD[1][3]*BD[1] - AD[0][2]*AD[1][1]*BD[3];
                      SU[COUNTER-1][1] = AD[0][2]*AD[1][0]*BD[3] - AD[0][0]*AD[1][2]*BD[3] + AD[0][0]*AD[1][3]*BD[2]
                        -  AD[0][3]*AD[1][0]*BD[2] + AD[0][3]*AD[1][2]*BD[0] - AD[0][2]*AD[1][3]*BD[0];
                      SU[COUNTER-1][2] = AD[0][0]*AD[1][1]*BD[3] - AD[0][0]*AD[1][3]*BD[1] + AD[0][3]*AD[1][0]*BD[1]
                        -  AD[0][3]*AD[1][1]*BD[0] + AD[0][1]*AD[1][3]*BD[0] - AD[0][1]*AD[1][0]*BD[3];
                      SU[COUNTER-1][3] = AD[0][2]*AD[1][1]*BD[0] - AD[0][1]*AD[1][2]*BD[0] + AD[0][0]*AD[1][2]*BD[1]
                        -  AD[0][2]*AD[1][0]*BD[1] + AD[0][1]*AD[1][0]*BD[2] - AD[0][0]*AD[1][1]*BD[2];
                      
                      for (k=0; k<4;k++)
                        {
                          SU[COUNTER-1][k]/=6.; // to get tetraedron's volume
                        }
                      
                    }
                      
/*                    for (j=0; j<IS; j++) */
/*                  { */
/*                    fprintf(stderr,"group[%d]=(%d,%d,%d)\n",j,group[j][0],group[j][1],group[j][2]); */
/*                  } */
                      // if it is not a multiple of 3 try something else...
                    }
                }
                }

              //check if there are still issues
          
              countSingleEdges=0;
              for(j=1; j<=32; j++)
                for (k=1; k<j; k++)
                  {
                if ( numberConnectionIsUsed[j-1][k-1]>-1 )
                  {
                    //fprintf(stderr,"connection between %d and %d is used %d times.\n", j, k, numberConnectionIsUsed[j-1][k-1]);
                    if (numberConnectionIsUsed[j-1][k-1]<2)
                      {
                    if (numberConnectionIsUsed[j-1][k-1]==0)
                      countSingleEdges+=2;
                    else
                      countSingleEdges+=1;
                    singleConnections[countSingleEdges-1][0]=j;
                    singleConnections[countSingleEdges-1][1]=k;
                    //fprintf(stderr,"-------------------------------------------\n");
                      }
                    if (numberConnectionIsUsed[j-1][k-1]>2)
                      {
                    //fprintf(stderr,"connection between %d and %d is used %d times.\n", j, k, numberConnectionIsUsed[j-1][k-1]);
                    //fprintf(stderr,"Three connections is messed up. That can't happen.\n");
                    break;
                      }
                  }
                  }
        




              if (countSingleEdges>0)
                {
                  IS=0;
                  for (i=0; i<NSurfaces; i++)
                for (j=0; j<NSurfaces; j++)
                  {
                    if (numberConnectionIsUsed[iEdge[i]-1][iEdge[j]-1]==1)
                      {
                    IS+=1;
                    //fprintf(stderr,"Missing ones: %d-%d \n",iEdge[i],iEdge[j]);
                    edge1[IS-1]=iEdge[i];
                    edge2[IS-1]=iEdge[j];
                      }
                  }
                  for (is=0; is<IS; is++)
                {
                  //                  fprintf(stderr,"edge1[%d]=%d edge2[%d]=%d \n",is,edge1[is],is,edge2[is]);
                  for (m=0; m<IS; m++)
                    {
                      if (edge1[is]==edge1[m])
                    {
                      if(IBIT[edge2[is]-1][edge2[m]-1]==1 || IBIT[edge2[is]-1][edge2[m]-1]==2 
                         || IBIT[edge2[is]-1][edge2[m]-1]==3 || IBIT[edge2[is]-1][edge2[m]-1]==4)
                        {
                          //      fprintf(stderr,"add %d - %d \n",edge2[is],edge2[m]);
                          if (numberConnectionIsUsed[edge2[is]-1][edge2[m]-1]==-1)
                        numberConnectionIsUsed[edge2[is]-1][edge2[m]-1]=0;
                        }
                    }
                      if (edge1[is]==edge2[m])
                    {
                      if(IBIT[edge2[is]-1][edge1[m]-1]==1 || IBIT[edge2[is]-1][edge1[m]-1]==2
                         || IBIT[edge2[is]-1][edge1[m]-1]==3 || IBIT[edge2[is]-1][edge1[m]-1]==4)
                        {
                          //fprintf(stderr,"add %d - %d \n",edge2[is],edge1[m]);
                          if (numberConnectionIsUsed[edge2[is]-1][edge1[m]-1]==-1)
                        numberConnectionIsUsed[edge2[is]-1][edge1[m]-1]=0;
                        }
                    }
                      if (edge2[is]==edge2[m])
                    {
                      if(IBIT[edge1[is]-1][edge1[m]-1]==1 || IBIT[edge1[is]-1][edge1[m]-1]==2
                         || IBIT[edge1[is]-1][edge1[m]-1]==3 || IBIT[edge1[is]-1][edge1[m]-1]==4)
                        {
                          //fprintf(stderr,"add %d - %d \n",edge1[is],edge1[m]);
                          if (numberConnectionIsUsed[edge1[is]-1][edge1[m]-1]==-1)
                        numberConnectionIsUsed[edge1[is]-1][edge1[m]-1]=0;
                        }
                    }
                      
                    }
                }
                  
                  
                  
                  countSingleEdges=0;
                  for(j=1; j<=32; j++)
                for (k=1; k<j; k++)
                  {
                    if ( numberConnectionIsUsed[j-1][k-1]>-1 )
                      {
                    if (numberConnectionIsUsed[j-1][k-1]<2)
                      {
                        if (numberConnectionIsUsed[j-1][k-1]==0)
                          {
                        countSingleEdges+=2;
                        singleConnections[countSingleEdges-1][0]=j;
                        singleConnections[countSingleEdges-1][1]=k;
                        singleConnectionsUsed[countSingleEdges-1]=-1;
                          }
                        else
                          {
                        countSingleEdges+=1;
                        singleConnections[countSingleEdges-1][0]=j;
                        singleConnections[countSingleEdges-1][1]=k;
                        singleConnectionsUsed[countSingleEdges-1]=0;
                          }
                        //  fprintf(stderr,"-------------------------------------------\n");
                      }
                    if (numberConnectionIsUsed[j-1][k-1]>2)
                      {
                        //fprintf(stderr,"connection between %d and %d is used %d times.\n", j, k, numberConnectionIsUsed[j-1][k-1]);
                        //fprintf(stderr,"Three connections is messed up. That can't happen.\n");
                        break;
                      }
                      }
                  }
                  
                  //fprintf(stderr,"DO THE VERY LAST CHECK, single edges=%d, tries2=%d\n",countSingleEdges,tries2);
                  
                  if (countSingleEdges>0)
                {
                  if (countSingleEdges%3==0 || NSurfaces>8) //if we have a number of single edges that is a multiple of 3
                    {
                      IS=0;
                      for (j=0; j<countSingleEdges; j++)
                    {
                      if(NSurfaces==12)
                        {
                          //          fprintf(stderr,"singleConnections[%d][0]=%d\n",j,singleConnections[j][0]);
                          //fprintf(stderr,"singleConnections[%d][1]=%d\n",j,singleConnections[j][1]);
                        }
                      for (i=0; i<countSingleEdges; i++)
                        {
                          if (i!=j && singleConnectionsUsed[j]<1 && singleConnectionsUsed[i]<1 )
                        {
                          for ( k=0; k<countSingleEdges; k++ )
                            {
                              if(k!=j && k!=i)
                            {
                              if ( ((singleConnections[j][0] == singleConnections[i][0]
                                 && singleConnections[j][1] == singleConnections[k][0]
                                 && singleConnections[k][1] == singleConnections[i][1])
                                ||
                                (singleConnections[j][0] == singleConnections[i][1]
                                 && singleConnections[j][1] == singleConnections[k][0]
                                 && singleConnections[k][1] == singleConnections[i][0])
                                ||
                                (singleConnections[j][0] == singleConnections[i][0]
                                 && singleConnections[j][1] == singleConnections[k][1]
                                 && singleConnections[k][0] == singleConnections[i][1])
                                ||
                                (singleConnections[j][0] == singleConnections[i][1]
                                 && singleConnections[j][1] == singleConnections[k][1]
                                 && singleConnections[k][0] == singleConnections[i][0])
                                ||
                                (singleConnections[j][1] == singleConnections[i][0]
                                 && singleConnections[j][0] == singleConnections[k][0]
                                 && singleConnections[k][1] == singleConnections[i][1])
                                ||
                                (singleConnections[j][1] == singleConnections[i][1]
                                 && singleConnections[j][0] == singleConnections[k][0]
                                 && singleConnections[k][1] == singleConnections[i][0])
                                ||
                                (singleConnections[j][1] == singleConnections[i][0]
                                 && singleConnections[j][0] == singleConnections[k][1]
                                 && singleConnections[k][0] == singleConnections[i][1])
                                ||
                                (singleConnections[j][1] == singleConnections[i][1]
                                 && singleConnections[j][0] == singleConnections[k][1]
                                 && singleConnections[k][0] == singleConnections[i][0])
                                )&& singleConnectionsUsed[k]==0 )
                                {
                                  singleConnectionsUsed[k]+=1;
                                  singleConnectionsUsed[i]+=1;
                                  singleConnectionsUsed[j]+=1;
                                  group[IS][0]=singleConnections[j][0];
                                  group[IS][1]=singleConnections[j][1];
                                  if (singleConnections[i][0]!=singleConnections[j][0] 
                                  && singleConnections[i][0]!=singleConnections[j][1] )
                                group[IS][2]=singleConnections[i][0];
                                  else
                                group[IS][2]=singleConnections[i][1];
                                  IS+=1;
                                }
                            }
                            }
                        }
                        }
                    }
                      
                      if (IS!=countSingleEdges/3) 
                    {
                      srand(time(NULL));
                      m=0;
                      tries=0;
                      while(IS!=countSingleEdges/3 && tries<5) 
                        {
                          tries+=1;
                          
                          
                          
                          for(j=1; j<=32; j++)
                        for (k=1; k<j; k++)
                          {
                            if (numberConnectionIsUsed[j-1][k-1]==0)
                              {
                            singleConnectionsUsed[countSingleEdges-1]=-1;
                              }
                            else
                              {
                            singleConnectionsUsed[countSingleEdges-1]=0;
                              }
                          }
                          
                          
                          //fprintf(stderr,"MISSED SOME TETRAEDRA. got %d out of %d\n",IS, countSingleEdges/3);
                          IS=0;
                          for (j=0; j<countSingleEdges; j++)
                        {
                          m=abs(rand()%countSingleEdges);
                          //  fprintf(stderr,"singleConnections[%d][0]=%d\n",j,singleConnections[j][0]);
                          //fprintf(stderr,"singleConnections[%d][1]=%d\n",j,singleConnections[j][1]);
                          for (l=0; l<countSingleEdges; l++)
                            {
                              i=(l+m)%countSingleEdges;
                              //RANDOM ORDERING     
                              if (i!=j && singleConnectionsUsed[j]<1 && singleConnectionsUsed[i]<1 )
                            {
                              for ( k=0; k<countSingleEdges; k++ )
                                {
                                  if(k!=j && k!=i)
                                {
                                  if ( ((singleConnections[j][0] == singleConnections[i][0]
                                     && singleConnections[j][1] == singleConnections[k][0]
                                     && singleConnections[k][1] == singleConnections[i][1])
                                    ||
                                    (singleConnections[j][0] == singleConnections[i][1]
                                     && singleConnections[j][1] == singleConnections[k][0]
                                     && singleConnections[k][1] == singleConnections[i][0])
                                    ||
                                    (singleConnections[j][0] == singleConnections[i][0]
                                     && singleConnections[j][1] == singleConnections[k][1]
                                     && singleConnections[k][0] == singleConnections[i][1])
                                    ||
                                    (singleConnections[j][0] == singleConnections[i][1]
                                     && singleConnections[j][1] == singleConnections[k][1]
                                     && singleConnections[k][0] == singleConnections[i][0])
                                    ||
                                    (singleConnections[j][1] == singleConnections[i][0]
                                     && singleConnections[j][0] == singleConnections[k][0]
                                     && singleConnections[k][1] == singleConnections[i][1])
                                    ||
                                    (singleConnections[j][1] == singleConnections[i][1]
                                     && singleConnections[j][0] == singleConnections[k][0]
                                     && singleConnections[k][1] == singleConnections[i][0])
                                    ||
                                    (singleConnections[j][1] == singleConnections[i][0]
                                     && singleConnections[j][0] == singleConnections[k][1]
                                     && singleConnections[k][0] == singleConnections[i][1])
                                    ||
                                    (singleConnections[j][1] == singleConnections[i][1]
                                     && singleConnections[j][0] == singleConnections[k][1]
                                     && singleConnections[k][0] == singleConnections[i][0])
                                    )&& singleConnectionsUsed[k]==0 )
                                    {
                                      if (singleConnectionsUsed[k]<1)
                                    singleConnectionsUsed[k]+=1;
                                      if (singleConnectionsUsed[i]<1)
                                    singleConnectionsUsed[i]+=1;
                                      if (singleConnectionsUsed[j]<1)
                                    singleConnectionsUsed[j]+=1;
                                      
                                      group[IS][0]=singleConnections[j][0];
                                      group[IS][1]=singleConnections[j][1];
                                      if (singleConnections[i][0]!=singleConnections[j][0] 
                                      && singleConnections[i][0]!=singleConnections[j][1] )
                                    group[IS][2]=singleConnections[i][0];
                                      else
                                    group[IS][2]=singleConnections[i][1];
                                      IS+=1;
                                      //      fprintf(stderr,"IS=%d\n",IS);
                                    }
                                }
                                }
                            }
                            }
                        }
/*                        for (j=0; j<IS; j++) */
/*                      { */
/*                        fprintf(stderr,"group[%d]=(%d,%d,%d)\n",j,group[j][0],group[j][1],group[j][2]); */
/*                      } */
                        }
                      //sleep(50);
                    }
                    }
                }
                  
                  if (countSingleEdges>0)
                {
                  if (countSingleEdges%3==0 || NSurfaces>8) //if we have a number of single edges that is a multiple of 3
                    {
                      for (j=0; j<IS; j++)
                    {
                      skip=0;
                      for(is=0; is<NSurfaces; is++)
                        {
                          if (iEdge[is]==group[j][0])
                        {
                          for(i=0; i<NSurfaces; i++)
                            {
                              if (iEdge[i]==group[j][1])
                            {
                              if(iEdge[is]>iEdge[i])
                                {
                                  if (numberConnectionIsUsed[iEdge[is]-1][iEdge[i]-1]>1)
                                {skip=1; continue;}
                                  numberConnectionIsUsed[iEdge[is]-1][iEdge[i]-1]+=1;
                                }
                              else
                                {
                                  if (numberConnectionIsUsed[iEdge[i]-1][iEdge[is]-1]>1)
                                {skip=1; continue;}
                                  numberConnectionIsUsed[iEdge[i]-1][iEdge[is]-1]+=1;
                                }
                              if(iEdge[is]>group[j][2])
                                {
                                  if (numberConnectionIsUsed[iEdge[is]-1][group[j][2]-1]>1)
                                {skip=1; continue;}
                                  numberConnectionIsUsed[iEdge[is]-1][group[j][2]-1]+=1;
                                }                             
                              else
                                {
                                  if (numberConnectionIsUsed[group[j][2]-1][iEdge[is]-1]>1)
                                {skip=1; continue;}
                                  numberConnectionIsUsed[group[j][2]-1][iEdge[is]-1]+=1;
                                }
                              if(iEdge[i]>group[j][2])
                                {
                                  if (numberConnectionIsUsed[iEdge[i]-1][group[j][2]-1]>1)
                                {skip=1; continue;}
                                  numberConnectionIsUsed[iEdge[i]-1][group[j][2]-1]+=1;
                                }
                              else
                                {
                                  if (numberConnectionIsUsed[group[j][2]-1][iEdge[i]-1]>1)
                                {skip=1; continue;}
                                  numberConnectionIsUsed[group[j][2]-1][iEdge[i]-1]+=1;
                                }
                              //fprintf(stderr,"IN THE LOOP 1 and j=%d\n",j);
                              //fprintf(stderr,"edges= %d, %d, %d\n",iEdge[is], iEdge[i], group[j][2]);
                              for(k=0;k<4;k++)
                                {
                                  AD[0][k] = cuts[is][k]-cuts[i][k];
                                  BD[k] = VMID[k]-cuts[is][k];
                                }
                              skip=1;
                              continue;
                            }
                            }
                          for(i=0; i<NSurfaces; i++)
                            {
                              if (iEdge[i]==group[j][2])
                            {
                              for(k=0;k<4;k++)
                                {
                                  AD[1][k] = cuts[is][k]-cuts[i][k];
                                }
                              skip=1;
                              continue;
                            }
                            }
                        }
                          if ( skip==1 ) continue;
                        }
                      COUNTER+=1;
                      SU[COUNTER-1][0] = AD[0][1]*AD[1][2]*BD[3] - AD[0][1]*AD[1][3]*BD[2] + AD[0][3]*AD[1][1]*BD[2]
                        -  AD[0][3]*AD[1][2]*BD[1] + AD[0][2]*AD[1][3]*BD[1] - AD[0][2]*AD[1][1]*BD[3];
                      SU[COUNTER-1][1] = AD[0][2]*AD[1][0]*BD[3] - AD[0][0]*AD[1][2]*BD[3] + AD[0][0]*AD[1][3]*BD[2]
                        -  AD[0][3]*AD[1][0]*BD[2] + AD[0][3]*AD[1][2]*BD[0] - AD[0][2]*AD[1][3]*BD[0];
                      SU[COUNTER-1][2] = AD[0][0]*AD[1][1]*BD[3] - AD[0][0]*AD[1][3]*BD[1] + AD[0][3]*AD[1][0]*BD[1]
                        -  AD[0][3]*AD[1][1]*BD[0] + AD[0][1]*AD[1][3]*BD[0] - AD[0][1]*AD[1][0]*BD[3];
                      SU[COUNTER-1][3] = AD[0][2]*AD[1][1]*BD[0] - AD[0][1]*AD[1][2]*BD[0] + AD[0][0]*AD[1][2]*BD[1]
                        -  AD[0][2]*AD[1][0]*BD[1] + AD[0][1]*AD[1][0]*BD[2] - AD[0][0]*AD[1][1]*BD[2];
                      
                      for (k=0; k<4;k++)
                        {
                          SU[COUNTER-1][k]/=6.; // to get tetraedron's volume
                        }
                      
                    }
                      
                /*       for (j=0; j<IS; j++) */
/*                  { */
/*                    fprintf(stderr,"group[%d]=(%d,%d,%d)\n",j,group[j][0],group[j][1],group[j][2]); */
/*                  } */
                      // if it is not a multiple of 3 try something else...
                    }
                }
                }
            }         
          
          //--------------------------------------------------------------------------------------------------------------
          // at this point all tetraedra should have been found and added
          //--------------------------------------------------------------------------------------------------------------

          
          // make sure every vector points towards lower energy densities.
          for (j=0; j<COUNTER; j++)
            {
              if ( SU[j][0]*VD0 < 0)
            SU[j][0]*=-1;
              if ( SU[j][1]*VD1 < 0)
            SU[j][1]*=-1;
              if ( SU[j][2]*VD2 < 0)
            SU[j][2]*=-1;
              if ( SU[j][3]*VD3 < 0)
            SU[j][3]*=-1;
              for (k=0; k<4; k++)
            {
              FULLSU[k]+=SU[j][k];
            }
            }
          
          //fprintf(stderr,"number of tetraedra = %d\n",COUNTER);
          
          countSingleEdges=0;
          for(j=1; j<=32; j++)
            for (k=1; k<j; k++)
              {
            if ( numberConnectionIsUsed[j-1][k-1]>-1 )
              {
                //fprintf(stderr,"connection between %d and %d is used %d times.\n", j, k, numberConnectionIsUsed[j-1][k-1]);
                if (numberConnectionIsUsed[j-1][k-1]<2)
                  {
                countSingleEdges+=1;
                //fprintf(stderr,"-------------------------------------------\n");
                  }
                if (numberConnectionIsUsed[j-1][k-1]>2)
                  {
                //fprintf(stderr,"connection between %d and %d is used %d times.\n", j, k, numberConnectionIsUsed[j-1][k-1]);
                //fprintf(stderr,"Three connections is messed up. That can't happen.\n");
                break;
                  }
              }
              }
//        fprintf(stderr,"number of single Edges = %d\n",countSingleEdges);
          if (countSingleEdges%3!=0)
            {
              cerr << "NUMBER OF SINGLE EDGES IS NOT A MULTIPLE OF 3, number=" << countSingleEdges << endl;
              //fprintf(stderr,"NSurfaces=%d\n",NSurfaces);
              //fprintf(stderr,"Tetraedra=%d\n",COUNTER);
              //sleep(1);
              weirdCases+=1;
             /*  countSingleEdges=0; */
/*            for(j=1; j<=32; j++) */
/*          for (k=1; k<j; k++) */
/*            { */
/*              if ( numberConnectionIsUsed[j-1][k-1]>0 ) */
/*                { */
/*              fprintf(stderr,"connection between %d and %d is used %d times.\n", j, k, numberConnectionIsUsed[j-1][k-1]); */
/*              if (numberConnectionIsUsed[j-1][k-1]<2) */
/*                { */
/*                  countSingleEdges+=1; */
/*                  fprintf(stderr,"-------------------------------------------\n"); */
/*                } */
/*              if (numberConnectionIsUsed[j-1][k-1]>2) */
/*                { */
/*                  fprintf(stderr,"connection between %d and %d is used %d times.\n", j, k, numberConnectionIsUsed[j-1][k-1]); */
/*                  fprintf(stderr,"Three connections is messed up. That can't happen.\n"); */
/*                  exit(1); */
/*                } */
/*                } */
/*            } */
              
              //if (NSurfaces<12) exit(1);
              cerr << "percent error=" << 100*(warnings*1.0)/(cells*1.0) << endl;
            }
          

          cells+=1;
          SUM += sqrt(FULLSU[0]*FULLSU[0]+FULLSU[1]*FULLSU[1]+FULLSU[2]*FULLSU[2]+FULLSU[3]*FULLSU[3]);
          

          if ((NSurfaces==6 && COUNTER!=8) || (NSurfaces==8 && COUNTER!=12) 
              || (NSurfaces==10 && (COUNTER!=16 && COUNTER!=15)) || (NSurfaces==12 && (COUNTER!=19 && COUNTER!=20 && COUNTER!=21)))
            {
              cerr << "*************************************************** NSurfaces=" << NSurfaces << " but number of tetrahedra=" << COUNTER << endl;
              cerr << "tauf=" << tauf << " xf=" << xf << " yf=" << yf << " etaf=" << etaf << endl;
              //if (abs(yf)<=0.1 && abs(etaf)<=0.1) 
              //              sleep(1);
              // fprintf(t_file,"%f %f %f %f\n",xf,tauf,FULLSU[1],FULLSU[0]);
              //if(NSurfaces<10) exit(1);
              warnings+=1;
              // sleep(1);
              if (countSingleEdges%3!=0) 
            {
              cerr << "single edges=" << countSingleEdges << endl;
              //sleep(1);
              //break;
            }
              cerr << "percent error=" << 100*(warnings*1.0)/(cells*1.0) << endl;
              //continue; // don't add the flawed cells at all
            }
      
          //fprintf(stderr,"Volume=%f\n", SUM);
          //fprintf(stderr,"Cells done=%d\n", cells);
          //fprintf(stderr,"Warnings=%d\n", warnings);
          
          xfrac = (xf-x)/(DX);
          yfrac = (yf-y)/(DY);
          etafrac = (etaf-eta)/(DETA);
          taufrac = (tauf-(tau-DTAU))/(DTAU);

          // 4d interpolation:
          //utau:
          utauX1 = ((1.-xfrac)*arena[ix][iy][ieta].u[0][0] + xfrac*arena[ix+fac][iy][ieta].u[0][0]);
          utauX2 = ((1.-xfrac)*arena[ix][iy+fac][ieta].u[0][0] + xfrac*arena[ix+fac][iy+fac][ieta].u[0][0]);
          
          if (ieta<neta-fac)
            {
              utauX3 = ((1.-xfrac)*arena[ix][iy][ieta+fac].u[0][0] + xfrac*arena[ix+fac][iy][ieta+fac].u[0][0]);
              utauX4 = ((1.-xfrac)*arena[ix][iy+fac][ieta+fac].u[0][0] + xfrac*arena[ix+fac][iy+fac][ieta+fac].u[0][0]);
            }
          else
            {
              utauX3 = ((1.-xfrac)*Rneighbor_utau[ix][iy] + xfrac*Rneighbor_utau[ix+fac][iy]);
              utauX4 = ((1.-xfrac)*Rneighbor_utau[ix][iy+fac] + xfrac*Rneighbor_utau[ix+fac][iy+fac]);
              //cout << "utau=" << utauX3 << " " << utauX4 << endl;
            }

          utauY1 = ((1.-yfrac)*utauX1+yfrac*utauX2);
          utauY2 = ((1.-yfrac)*utauX3+yfrac*utauX4);

          utau1 = ((1.-etafrac)*utauY1+etafrac*utauY2);
        
          utauX1 = ((1.-xfrac)*arena[ix][iy][ieta].u_prev[0] + xfrac*arena[ix+fac][iy][ieta].u_prev[0]);
          utauX2 = ((1.-xfrac)*arena[ix][iy+fac][ieta].u_prev[0] + xfrac*arena[ix+fac][iy+fac][ieta].u_prev[0]);
          
          if (ieta<neta-fac)
            {
              utauX3 = ((1.-xfrac)*arena[ix][iy][ieta+fac].u_prev[0] 
                + xfrac*arena[ix+fac][iy][ieta+fac].u_prev[0]);
              utauX4 = ((1.-xfrac)*arena[ix][iy+fac][ieta+fac].u_prev[0] 
                + xfrac*arena[ix+fac][iy+fac][ieta+fac].u_prev[0]);
            }
          else
            {
              utauX3 = ((1.-xfrac)*Rneighbor_utau_prev[ix][iy] 
                + xfrac*Rneighbor_utau_prev[ix+fac][iy]);
              utauX4 = ((1.-xfrac)*Rneighbor_utau_prev[ix][iy+fac]
                + xfrac*Rneighbor_utau_prev[ix+fac][iy+fac]);
              //cout << "utau_prev=" << utauX3 << " " << utauX4 << endl;
            }
          
          utauY1 = ((1.-yfrac)*utauX1+yfrac*utauX2);
          utauY2 = ((1.-yfrac)*utauX3+yfrac*utauX4);

          utau2 = ((1.-etafrac)*utauY1+etafrac*utauY2);
          
          utau = (1.-taufrac)*utau2+taufrac*utau1;

          //ux:
          uxX1 = ((1.-xfrac)*arena[ix][iy][ieta].u[0][1] + xfrac*arena[ix+fac][iy][ieta].u[0][1]);
          uxX2 = ((1.-xfrac)*arena[ix][iy+fac][ieta].u[0][1] + xfrac*arena[ix+fac][iy+fac][ieta].u[0][1]);
        
          if (ieta<neta-fac)
            {
              uxX3 = ((1.-xfrac)*arena[ix][iy][ieta+fac].u[0][1] + xfrac*arena[ix+fac][iy][ieta+fac].u[0][1]);
              uxX4 = ((1.-xfrac)*arena[ix][iy+fac][ieta+fac].u[0][1] + xfrac*arena[ix+fac][iy+fac][ieta+fac].u[0][1]);
            }
          else
            {
              uxX3 = ((1.-xfrac)*Rneighbor_ux[ix][iy] + xfrac*Rneighbor_ux[ix+fac][iy]);
              uxX4 = ((1.-xfrac)*Rneighbor_ux[ix][iy+fac] + xfrac*Rneighbor_ux[ix+fac][iy+fac]);
              //cout << "ux=" << uxX3 << " " << uxX4 << endl;
            }
          
          uxY1 = ((1.-yfrac)*uxX1+yfrac*uxX2);
          uxY2 = ((1.-yfrac)*uxX3+yfrac*uxX4);

          ux1 = ((1.-etafrac)*uxY1+etafrac*uxY2);
              
          uxX1 = ((1.-xfrac)*arena[ix][iy][ieta].u_prev[1] + xfrac*arena[ix+fac][iy][ieta].u_prev[1]);
          uxX2 = ((1.-xfrac)*arena[ix][iy+fac][ieta].u_prev[1] + xfrac*arena[ix+fac][iy+fac][ieta].u_prev[1]);
          

          if (ieta<neta-fac)
            {
              uxX3 = ((1.-xfrac)*arena[ix][iy][ieta+fac].u_prev[1] + xfrac*arena[ix+fac][iy][ieta+fac].u_prev[1]);
              uxX4 = ((1.-xfrac)*arena[ix][iy+fac][ieta+fac].u_prev[1] + xfrac*arena[ix+fac][iy+fac][ieta+fac].u_prev[1]);
            }
          else
            {
              uxX3 = ((1.-xfrac)*Rneighbor_ux_prev[ix][iy] 
                  + xfrac*Rneighbor_ux_prev[ix+fac][iy]);
              uxX4 = ((1.-xfrac)*Rneighbor_ux_prev[ix][iy+fac] 
                  + xfrac*Rneighbor_ux_prev[ix+fac][iy+fac]);
              //cout << "ux_prev=" << uxX3 << " " << uxX4 << endl;
            }

          uxY1 = ((1.-yfrac)*uxX1+yfrac*uxX2);
          uxY2 = ((1.-yfrac)*uxX3+yfrac*uxX4);

          ux2 = ((1.-etafrac)*uxY1+etafrac*uxY2);

          ux = (1.-taufrac)*ux2+taufrac*ux1;
        
          //uy:
          uyX1 = ((1.-xfrac)*arena[ix][iy][ieta].u[0][2] + xfrac*arena[ix+fac][iy][ieta].u[0][2]);
          uyX2 = ((1.-xfrac)*arena[ix][iy+fac][ieta].u[0][2] + xfrac*arena[ix+fac][iy+fac][ieta].u[0][2]);

          if (ieta<neta-fac)
            {
              uyX3 = ((1.-xfrac)*arena[ix][iy][ieta+fac].u[0][2] + xfrac*arena[ix+fac][iy][ieta+fac].u[0][2]);
              uyX4 = ((1.-xfrac)*arena[ix][iy+fac][ieta+fac].u[0][2] + xfrac*arena[ix+fac][iy+fac][ieta+fac].u[0][2]);
            }
          else
            {
              uyX3 = ((1.-xfrac)*Rneighbor_uy[ix][iy] + xfrac*Rneighbor_uy[ix+fac][iy]);
              uyX4 = ((1.-xfrac)*Rneighbor_uy[ix][iy+fac] + xfrac*Rneighbor_uy[ix+fac][iy+fac]);
              //cout << "uy=" << uyX3 << " " << uyX4 << endl;
            }

          uyY1 = ((1.-yfrac)*uyX1+yfrac*uyX2);
          uyY2 = ((1.-yfrac)*uyX3+yfrac*uyX4);

          uy1 = ((1.-etafrac)*uyY1+etafrac*uyY2);
              
          uyX1 = ((1.-xfrac)*arena[ix][iy][ieta].u_prev[2] + xfrac*arena[ix+fac][iy][ieta].u_prev[2]);
          uyX2 = ((1.-xfrac)*arena[ix][iy+fac][ieta].u_prev[2] + xfrac*arena[ix+fac][iy+fac][ieta].u_prev[2]);

          if (ieta<neta-fac)
            {
              uyX3 = ((1.-xfrac)*arena[ix][iy][ieta+fac].u_prev[2] + xfrac*arena[ix+fac][iy][ieta+fac].u_prev[2]);
              uyX4 = ((1.-xfrac)*arena[ix][iy+fac][ieta+fac].u_prev[2] + xfrac*arena[ix+fac][iy+fac][ieta+fac].u_prev[2]);
            }
          else
            {
              uyX3 = ((1.-xfrac)*Rneighbor_uy_prev[ix][iy] + xfrac*Rneighbor_uy_prev[ix+fac][iy]);
              uyX4 = ((1.-xfrac)*Rneighbor_uy_prev[ix][iy+fac] + xfrac*Rneighbor_uy_prev[ix+fac][iy+fac]);
              //cout << "uy_prev=" << uyX3 << " " << uyX4 << endl;
            }

          uyY1 = ((1.-yfrac)*uyX1+yfrac*uyX2);
          uyY2 = ((1.-yfrac)*uyX3+yfrac*uyX4);

          uy2 = ((1.-etafrac)*uyY1+etafrac*uyY2);

          uy = (1.-taufrac)*uy2+taufrac*uy1;
          

          //ueta:
          uetaX1 = ((1.-xfrac)*arena[ix][iy][ieta].u[0][3] + xfrac*arena[ix+fac][iy][ieta].u[0][3]);
          uetaX2 = ((1.-xfrac)*arena[ix][iy+fac][ieta].u[0][3] + xfrac*arena[ix+fac][iy+fac][ieta].u[0][3]);

          if (ieta<neta-fac)
            {
              uetaX3 = ((1.-xfrac)*arena[ix][iy][ieta+fac].u[0][3] + xfrac*arena[ix+fac][iy][ieta+fac].u[0][3]);
              uetaX4 = ((1.-xfrac)*arena[ix][iy+fac][ieta+fac].u[0][3] + xfrac*arena[ix+fac][iy+fac][ieta+fac].u[0][3]);
            }
          else
            {
              uetaX3 = ((1.-xfrac)*Rneighbor_ueta[ix][iy] + xfrac*Rneighbor_ueta[ix+fac][iy]);
              uetaX4 = ((1.-xfrac)*Rneighbor_ueta[ix][iy+fac] + xfrac*Rneighbor_ueta[ix+fac][iy+fac]);
              //cout << "ueta=" << uetaX3 << " " << uetaX4 << endl;
            }
          
          uetaY1 = ((1.-yfrac)*uetaX1+yfrac*uetaX2);
          uetaY2 = ((1.-yfrac)*uetaX3+yfrac*uetaX4);

          ueta1 = ((1.-etafrac)*uetaY1+etafrac*uetaY2);
              
          uetaX1 = ((1.-xfrac)*arena[ix][iy][ieta].u_prev[3] + xfrac*arena[ix+fac][iy][ieta].u_prev[3]);
          uetaX2 = ((1.-xfrac)*arena[ix][iy+fac][ieta].u_prev[3] + xfrac*arena[ix+fac][iy+fac][ieta].u_prev[3]);

          if (ieta<neta-fac)
            {
              uetaX3 = ((1.-xfrac)*arena[ix][iy][ieta+fac].u_prev[3] + xfrac*arena[ix+fac][iy][ieta+fac].u_prev[3]);
              uetaX4 = ((1.-xfrac)*arena[ix][iy+fac][ieta+fac].u_prev[3] + xfrac*arena[ix+fac][iy+fac][ieta+fac].u_prev[3]);
            }
          else
            {
              uetaX3 = ((1.-xfrac)*Rneighbor_ueta_prev[ix][iy] 
                + xfrac*Rneighbor_ueta_prev[ix+fac][iy]);
              uetaX4 = ((1.-xfrac)*Rneighbor_ueta_prev[ix][iy+fac] 
                + xfrac*Rneighbor_ueta_prev[ix+fac][iy+fac]);
              //cout << "ueta_prev=" << uetaX3 << " " << uetaX4 << endl;
            }
          
          uetaY1 = ((1.-yfrac)*uetaX1+yfrac*uetaX2);
          uetaY2 = ((1.-yfrac)*uetaX3+yfrac*uetaX4);

          ueta2 = ((1.-etafrac)*uetaY1+etafrac*uetaY2);

          ueta = (1.-taufrac)*ueta2+taufrac*ueta1;
          
          
          //rhob: 3D interpolation for now (no tau interpolation)
          rhobX1 = ((1.-xfrac)*arena[ix][iy][ieta].rhob + xfrac*arena[ix+fac][iy][ieta].rhob);
          rhobX2 = ((1.-xfrac)*arena[ix][iy+fac][ieta].rhob + xfrac*arena[ix+fac][iy+fac][ieta].rhob);

          if (ieta<neta-fac)
            {
              rhobX3 = ((1.-xfrac)*arena[ix][iy][ieta+fac].rhob + xfrac*arena[ix+fac][iy][ieta+fac].rhob);
              rhobX4 = ((1.-xfrac)*arena[ix][iy+fac][ieta+fac].rhob + xfrac*arena[ix+fac][iy+fac][ieta+fac].rhob);
            }
          else
            {
              rhobX3 = ((1.-xfrac)*Rneighbor_rhob[ix][iy] + xfrac*Rneighbor_rhob[ix+fac][iy]);
              rhobX4 = ((1.-xfrac)*Rneighbor_rhob[ix][iy+fac] + xfrac*Rneighbor_rhob[ix+fac][iy+fac]);
            }
          
          rhobY1 = ((1.-yfrac)*rhobX1+yfrac*rhobX2);
          rhobY2 = ((1.-yfrac)*rhobX3+yfrac*rhobX4);

          rhob = ((1.-etafrac)*rhobY1+etafrac*rhobY2);
        
          //Wtautau:
          WX1 = ((1.-xfrac)*arena[ix][iy][ieta].Wmunu[0][0][0] + xfrac*arena[ix+fac][iy][ieta].Wmunu[0][0][0]);
          WX2 = ((1.-xfrac)*arena[ix][iy+fac][ieta].Wmunu[0][0][0] + xfrac*arena[ix+fac][iy+fac][ieta].Wmunu[0][0][0]);

          if (ieta<neta-fac)
            {
              WX3 = ((1.-xfrac)*arena[ix][iy][ieta+fac].Wmunu[0][0][0] + xfrac*arena[ix+fac][iy][ieta+fac].Wmunu[0][0][0]);
              WX4 = ((1.-xfrac)*arena[ix][iy+fac][ieta+fac].Wmunu[0][0][0] + xfrac*arena[ix+fac][iy+fac][ieta+fac].Wmunu[0][0][0]);
            }
          else
            {
              WX3 = ((1.-xfrac)*Rneighbor_Wtautau[ix][iy] + xfrac*Rneighbor_Wtautau[ix+fac][iy]);
              WX4 = ((1.-xfrac)*Rneighbor_Wtautau[ix][iy+fac] + xfrac*Rneighbor_Wtautau[ix+fac][iy+fac]);
            }
          
          WY1 = ((1.-yfrac)*WX1+yfrac*WX2);
          WY2 = ((1.-yfrac)*WX3+yfrac*WX4);

          Wtautau1 = ((1.-etafrac)*WY1+etafrac*WY2);
          
          WX1 = ((1.-xfrac)*arena[ix][iy][ieta].W_prev[0][0] + xfrac*arena[ix+fac][iy][ieta].W_prev[0][0]);
          WX2 = ((1.-xfrac)*arena[ix][iy+fac][ieta].W_prev[0][0] + xfrac*arena[ix+fac][iy+fac][ieta].W_prev[0][0]);

          if (ieta<neta-fac)
            {
              WX3 = ((1.-xfrac)*arena[ix][iy][ieta+fac].W_prev[0][0] + xfrac*arena[ix+fac][iy][ieta+fac].W_prev[0][0]);
              WX4 = ((1.-xfrac)*arena[ix][iy+fac][ieta+fac].W_prev[0][0] + xfrac*arena[ix+fac][iy+fac][ieta+fac].W_prev[0][0]);
            }
          else
            {
              WX3 = ((1.-xfrac)*Rneighbor_Wtautau_prev[ix][iy] + xfrac*Rneighbor_Wtautau_prev[ix+fac][iy]);
              WX4 = ((1.-xfrac)*Rneighbor_Wtautau_prev[ix][iy+fac] + xfrac*Rneighbor_Wtautau_prev[ix+fac][iy+fac]);
            }
          
          WY1 = ((1.-yfrac)*WX1+yfrac*WX2);
          WY2 = ((1.-yfrac)*WX3+yfrac*WX4);

          Wtautau2 = ((1.-etafrac)*WY1+etafrac*WY2);

          Wtautau = (1.-taufrac)*Wtautau2+taufrac*Wtautau1;

          //Wtaux
          WX1 = ((1.-xfrac)*arena[ix][iy][ieta].Wmunu[0][0][1] + xfrac*arena[ix+fac][iy][ieta].Wmunu[0][0][1]);
          WX2 = ((1.-xfrac)*arena[ix][iy+fac][ieta].Wmunu[0][0][1] + xfrac*arena[ix+fac][iy+fac][ieta].Wmunu[0][0][1]);

          if (ieta<neta-fac)
            {
              WX3 = ((1.-xfrac)*arena[ix][iy][ieta+fac].Wmunu[0][0][1] + xfrac*arena[ix+fac][iy][ieta+fac].Wmunu[0][0][1]);
              WX4 = ((1.-xfrac)*arena[ix][iy+fac][ieta+fac].Wmunu[0][0][1] + xfrac*arena[ix+fac][iy+fac][ieta+fac].Wmunu[0][0][1]);
            }
          else
            {
              WX3 = ((1.-xfrac)*Rneighbor_Wtaux[ix][iy] + xfrac*Rneighbor_Wtaux[ix+fac][iy]);
              WX4 = ((1.-xfrac)*Rneighbor_Wtaux[ix][iy+fac] + xfrac*Rneighbor_Wtaux[ix+fac][iy+fac]);
            }
          
          WY1 = ((1.-yfrac)*WX1+yfrac*WX2);
          WY2 = ((1.-yfrac)*WX3+yfrac*WX4);

          Wtaux1 = ((1.-etafrac)*WY1+etafrac*WY2);
        
          WX1 = ((1.-xfrac)*arena[ix][iy][ieta].W_prev[0][1] + xfrac*arena[ix+fac][iy][ieta].W_prev[0][1]);
          WX2 = ((1.-xfrac)*arena[ix][iy+fac][ieta].W_prev[0][1] + xfrac*arena[ix+fac][iy+fac][ieta].W_prev[0][1]);

          if (ieta<neta-fac)
            {
              WX3 = ((1.-xfrac)*arena[ix][iy][ieta+fac].W_prev[0][1] + xfrac*arena[ix+fac][iy][ieta+fac].W_prev[0][1]);
              WX4 = ((1.-xfrac)*arena[ix][iy+fac][ieta+fac].W_prev[0][1] + xfrac*arena[ix+fac][iy+fac][ieta+fac].W_prev[0][1]);
            }
          else
            {
              WX3 = ((1.-xfrac)*Rneighbor_Wtaux_prev[ix][iy] + xfrac*Rneighbor_Wtaux_prev[ix+fac][iy]);
              WX4 = ((1.-xfrac)*Rneighbor_Wtaux_prev[ix][iy+fac] + xfrac*Rneighbor_Wtaux_prev[ix+fac][iy+fac]);
            }
          
          WY1 = ((1.-yfrac)*WX1+yfrac*WX2);
          WY2 = ((1.-yfrac)*WX3+yfrac*WX4);

          Wtaux2 = ((1.-etafrac)*WY1+etafrac*WY2);
        
          Wtaux = (1.-taufrac)*Wtaux2+taufrac*Wtaux1;

          //Wtauy: 
          WX1 = ((1.-xfrac)*arena[ix][iy][ieta].Wmunu[0][0][2] + xfrac*arena[ix+fac][iy][ieta].Wmunu[0][0][2]);
          WX2 = ((1.-xfrac)*arena[ix][iy+fac][ieta].Wmunu[0][0][2] + xfrac*arena[ix+fac][iy+fac][ieta].Wmunu[0][0][2]);

          if (ieta<neta-fac)
            {
              WX3 = ((1.-xfrac)*arena[ix][iy][ieta+fac].Wmunu[0][0][2] + xfrac*arena[ix+fac][iy][ieta+fac].Wmunu[0][0][2]);
              WX4 = ((1.-xfrac)*arena[ix][iy+fac][ieta+fac].Wmunu[0][0][2] + xfrac*arena[ix+fac][iy+fac][ieta+fac].Wmunu[0][0][2]);
            }
          else
            {
              WX3 = ((1.-xfrac)*Rneighbor_Wtauy[ix][iy] + xfrac*Rneighbor_Wtauy[ix+fac][iy]);
              WX4 = ((1.-xfrac)*Rneighbor_Wtauy[ix][iy+fac] + xfrac*Rneighbor_Wtauy[ix+fac][iy+fac]);
            }
          
          WY1 = ((1.-yfrac)*WX1+yfrac*WX2);
          WY2 = ((1.-yfrac)*WX3+yfrac*WX4);

          Wtauy1 = ((1.-etafrac)*WY1+etafrac*WY2);
        
          WX1 = ((1.-xfrac)*arena[ix][iy][ieta].W_prev[0][2] + xfrac*arena[ix+fac][iy][ieta].W_prev[0][2]);
          WX2 = ((1.-xfrac)*arena[ix][iy+fac][ieta].W_prev[0][2] + xfrac*arena[ix+fac][iy+fac][ieta].W_prev[0][2]);

          if (ieta<neta-fac)
            {
              WX3 = ((1.-xfrac)*arena[ix][iy][ieta+fac].W_prev[0][2] + xfrac*arena[ix+fac][iy][ieta+fac].W_prev[0][2]);
              WX4 = ((1.-xfrac)*arena[ix][iy+fac][ieta+fac].W_prev[0][2] + xfrac*arena[ix+fac][iy+fac][ieta+fac].W_prev[0][2]);
            }
          else
            {
              WX3 = ((1.-xfrac)*Rneighbor_Wtauy_prev[ix][iy] + xfrac*Rneighbor_Wtauy_prev[ix+fac][iy]);
              WX4 = ((1.-xfrac)*Rneighbor_Wtauy_prev[ix][iy+fac] + xfrac*Rneighbor_Wtauy_prev[ix+fac][iy+fac]);
            }
          
          WY1 = ((1.-yfrac)*WX1+yfrac*WX2);
          WY2 = ((1.-yfrac)*WX3+yfrac*WX4);

          Wtauy2 = ((1.-etafrac)*WY1+etafrac*WY2);

          Wtauy = (1.-taufrac)*Wtauy2+taufrac*Wtauy1;
        
          //Wtaueta: 
          WX1 = ((1.-xfrac)*arena[ix][iy][ieta].Wmunu[0][0][3] + xfrac*arena[ix+fac][iy][ieta].Wmunu[0][0][3]);
          WX2 = ((1.-xfrac)*arena[ix][iy+fac][ieta].Wmunu[0][0][3] + xfrac*arena[ix+fac][iy+fac][ieta].Wmunu[0][0][3]);

          if (ieta<neta-fac)
            {
              WX3 = ((1.-xfrac)*arena[ix][iy][ieta+fac].Wmunu[0][0][3] + xfrac*arena[ix+fac][iy][ieta+fac].Wmunu[0][0][3]);
              WX4 = ((1.-xfrac)*arena[ix][iy+fac][ieta+fac].Wmunu[0][0][3] + xfrac*arena[ix+fac][iy+fac][ieta+fac].Wmunu[0][0][3]);
            }
          else
            {
              WX3 = ((1.-xfrac)*Rneighbor_Wtaueta[ix][iy] + xfrac*Rneighbor_Wtaueta[ix+fac][iy]);
              WX4 = ((1.-xfrac)*Rneighbor_Wtaueta[ix][iy+fac] + xfrac*Rneighbor_Wtaueta[ix+fac][iy+fac]);
            }
          
          WY1 = ((1.-yfrac)*WX1+yfrac*WX2);
          WY2 = ((1.-yfrac)*WX3+yfrac*WX4);

          Wtaueta1 = ((1.-etafrac)*WY1+etafrac*WY2);

          WX1 = ((1.-xfrac)*arena[ix][iy][ieta].W_prev[0][3] + xfrac*arena[ix+fac][iy][ieta].W_prev[0][3]);
          WX2 = ((1.-xfrac)*arena[ix][iy+fac][ieta].W_prev[0][3] + xfrac*arena[ix+fac][iy+fac][ieta].W_prev[0][3]);

          if (ieta<neta-fac)
            {
              WX3 = ((1.-xfrac)*arena[ix][iy][ieta+fac].W_prev[0][3] + xfrac*arena[ix+fac][iy][ieta+fac].W_prev[0][3]);
              WX4 = ((1.-xfrac)*arena[ix][iy+fac][ieta+fac].W_prev[0][3] + xfrac*arena[ix+fac][iy+fac][ieta+fac].W_prev[0][3]);
            }
          else
            {
              WX3 = ((1.-xfrac)*Rneighbor_Wtaueta_prev[ix][iy] + xfrac*Rneighbor_Wtaueta_prev[ix+fac][iy]);
              WX4 = ((1.-xfrac)*Rneighbor_Wtaueta_prev[ix][iy+fac] + xfrac*Rneighbor_Wtaueta_prev[ix+fac][iy+fac]);
            }
          
          WY1 = ((1.-yfrac)*WX1+yfrac*WX2);
          WY2 = ((1.-yfrac)*WX3+yfrac*WX4);

          Wtaueta2 = ((1.-etafrac)*WY1+etafrac*WY2);

          Wtaueta = (1.-taufrac)*Wtaueta2+taufrac*Wtaueta1;

          //Wxx: 
          WX1 = ((1.-xfrac)*arena[ix][iy][ieta].Wmunu[0][1][1] + xfrac*arena[ix+fac][iy][ieta].Wmunu[0][1][1]);
          WX2 = ((1.-xfrac)*arena[ix][iy+fac][ieta].Wmunu[0][1][1] + xfrac*arena[ix+fac][iy+fac][ieta].Wmunu[0][1][1]);

          if (ieta<neta-fac)
            {
              WX3 = ((1.-xfrac)*arena[ix][iy][ieta+fac].Wmunu[0][1][1] + xfrac*arena[ix+fac][iy][ieta+fac].Wmunu[0][1][1]);
              WX4 = ((1.-xfrac)*arena[ix][iy+fac][ieta+fac].Wmunu[0][1][1] + xfrac*arena[ix+fac][iy+fac][ieta+fac].Wmunu[0][1][1]);
            }
          else
            {
              WX3 = ((1.-xfrac)*Rneighbor_Wxx[ix][iy] + xfrac*Rneighbor_Wxx[ix+fac][iy]);
              WX4 = ((1.-xfrac)*Rneighbor_Wxx[ix][iy+fac] + xfrac*Rneighbor_Wxx[ix+fac][iy+fac]);
            }
          
          WY1 = ((1.-yfrac)*WX1+yfrac*WX2);
          WY2 = ((1.-yfrac)*WX3+yfrac*WX4);

          Wxx1 = ((1.-etafrac)*WY1+etafrac*WY2);
        
          WX1 = ((1.-xfrac)*arena[ix][iy][ieta].W_prev[1][1] + xfrac*arena[ix+fac][iy][ieta].W_prev[1][1]);
          WX2 = ((1.-xfrac)*arena[ix][iy+fac][ieta].W_prev[1][1] + xfrac*arena[ix+fac][iy+fac][ieta].W_prev[1][1]);

          if (ieta<neta-fac)
            {
              WX3 = ((1.-xfrac)*arena[ix][iy][ieta+fac].W_prev[1][1] + xfrac*arena[ix+fac][iy][ieta+fac].W_prev[1][1]);
              WX4 = ((1.-xfrac)*arena[ix][iy+fac][ieta+fac].W_prev[1][1] + xfrac*arena[ix+fac][iy+fac][ieta+fac].W_prev[1][1]);
            }
          else
            {
              WX3 = ((1.-xfrac)*Rneighbor_Wxx_prev[ix][iy] + xfrac*Rneighbor_Wxx_prev[ix+fac][iy]);
              WX4 = ((1.-xfrac)*Rneighbor_Wxx_prev[ix][iy+fac] + xfrac*Rneighbor_Wxx_prev[ix+fac][iy+fac]);
            }
          
          WY1 = ((1.-yfrac)*WX1+yfrac*WX2);
          WY2 = ((1.-yfrac)*WX3+yfrac*WX4);

          Wxx2 = ((1.-etafrac)*WY1+etafrac*WY2);
          
          Wxx = (1.-taufrac)*Wxx2+taufrac*Wxx1;
          
          //Wxy: 
          WX1 = ((1.-xfrac)*arena[ix][iy][ieta].Wmunu[0][1][2] + xfrac*arena[ix+fac][iy][ieta].Wmunu[0][1][2]);
          WX2 = ((1.-xfrac)*arena[ix][iy+fac][ieta].Wmunu[0][1][2] + xfrac*arena[ix+fac][iy+fac][ieta].Wmunu[0][1][2]);

          if (ieta<neta-fac)
            {
              WX3 = ((1.-xfrac)*arena[ix][iy][ieta+fac].Wmunu[0][1][2] + xfrac*arena[ix+fac][iy][ieta+fac].Wmunu[0][1][2]);
              WX4 = ((1.-xfrac)*arena[ix][iy+fac][ieta+fac].Wmunu[0][1][2] + xfrac*arena[ix+fac][iy+fac][ieta+fac].Wmunu[0][1][2]);
            }
          else
            {
              WX3 = ((1.-xfrac)*Rneighbor_Wxy[ix][iy] + xfrac*Rneighbor_Wxy[ix+fac][iy]);
              WX4 = ((1.-xfrac)*Rneighbor_Wxy[ix][iy+fac] + xfrac*Rneighbor_Wxy[ix+fac][iy+fac]);
            }
          
          WY1 = ((1.-yfrac)*WX1+yfrac*WX2);
          WY2 = ((1.-yfrac)*WX3+yfrac*WX4);

          Wxy1 = ((1.-etafrac)*WY1+etafrac*WY2);
          
          WX1 = ((1.-xfrac)*arena[ix][iy][ieta].W_prev[1][2] + xfrac*arena[ix+fac][iy][ieta].W_prev[1][2]);
          WX2 = ((1.-xfrac)*arena[ix][iy+fac][ieta].W_prev[1][2] + xfrac*arena[ix+fac][iy+fac][ieta].W_prev[1][2]);

          if (ieta<neta-fac)
            {
              WX3 = ((1.-xfrac)*arena[ix][iy][ieta+fac].W_prev[1][2] + xfrac*arena[ix+fac][iy][ieta+fac].W_prev[1][2]);
              WX4 = ((1.-xfrac)*arena[ix][iy+fac][ieta+fac].W_prev[1][2] + xfrac*arena[ix+fac][iy+fac][ieta+fac].W_prev[1][2]);
            }
          else
            {
              WX3 = ((1.-xfrac)*Rneighbor_Wxy_prev[ix][iy] + xfrac*Rneighbor_Wxy_prev[ix+fac][iy]);
              WX4 = ((1.-xfrac)*Rneighbor_Wxy_prev[ix][iy+fac] + xfrac*Rneighbor_Wxy_prev[ix+fac][iy+fac]);
            }
          
          WY1 = ((1.-yfrac)*WX1+yfrac*WX2);
          WY2 = ((1.-yfrac)*WX3+yfrac*WX4);

          Wxy2 = ((1.-etafrac)*WY1+etafrac*WY2);
          
          Wxy = (1.-taufrac)*Wxy2+taufrac*Wxy1;
          
          //Wxeta:
          WX1 = ((1.-xfrac)*arena[ix][iy][ieta].Wmunu[0][1][3] + xfrac*arena[ix+fac][iy][ieta].Wmunu[0][1][3]);
          WX2 = ((1.-xfrac)*arena[ix][iy+fac][ieta].Wmunu[0][1][3] + xfrac*arena[ix+fac][iy+fac][ieta].Wmunu[0][1][3]);

          if (ieta<neta-fac)
            {
              WX3 = ((1.-xfrac)*arena[ix][iy][ieta+fac].Wmunu[0][1][3] + xfrac*arena[ix+fac][iy][ieta+fac].Wmunu[0][1][3]);
              WX4 = ((1.-xfrac)*arena[ix][iy+fac][ieta+fac].Wmunu[0][1][3] + xfrac*arena[ix+fac][iy+fac][ieta+fac].Wmunu[0][1][3]);
            }
          else
            {
              WX3 = ((1.-xfrac)*Rneighbor_Wxeta[ix][iy] + xfrac*Rneighbor_Wxeta[ix+fac][iy]);
              WX4 = ((1.-xfrac)*Rneighbor_Wxeta[ix][iy+fac] + xfrac*Rneighbor_Wxeta[ix+fac][iy+fac]);
            }
          
          WY1 = ((1.-yfrac)*WX1+yfrac*WX2);
          WY2 = ((1.-yfrac)*WX3+yfrac*WX4);

          Wxeta1 = ((1.-etafrac)*WY1+etafrac*WY2);
          
          WX1 = ((1.-xfrac)*arena[ix][iy][ieta].W_prev[1][3] + xfrac*arena[ix+fac][iy][ieta].W_prev[1][3]);
          WX2 = ((1.-xfrac)*arena[ix][iy+fac][ieta].W_prev[1][3] + xfrac*arena[ix+fac][iy+fac][ieta].W_prev[1][3]);

          if (ieta<neta-fac)
            {
              WX3 = ((1.-xfrac)*arena[ix][iy][ieta+fac].W_prev[1][3] + xfrac*arena[ix+fac][iy][ieta+fac].W_prev[1][3]);
              WX4 = ((1.-xfrac)*arena[ix][iy+fac][ieta+fac].W_prev[1][3] + xfrac*arena[ix+fac][iy+fac][ieta+fac].W_prev[1][3]);
            }
          else
            {
              WX3 = ((1.-xfrac)*Rneighbor_Wxeta_prev[ix][iy] + xfrac*Rneighbor_Wxeta_prev[ix+fac][iy]);
              WX4 = ((1.-xfrac)*Rneighbor_Wxeta_prev[ix][iy+fac] + xfrac*Rneighbor_Wxeta_prev[ix+fac][iy+fac]);
            }
          
          WY1 = ((1.-yfrac)*WX1+yfrac*WX2);
          WY2 = ((1.-yfrac)*WX3+yfrac*WX4);

          Wxeta2 = ((1.-etafrac)*WY1+etafrac*WY2);

          Wxeta = (1.-taufrac)*Wxeta2+taufrac*Wxeta1;
        
          //Wyy:
          WX1 = ((1.-xfrac)*arena[ix][iy][ieta].Wmunu[0][2][2] + xfrac*arena[ix+fac][iy][ieta].Wmunu[0][2][2]);
          WX2 = ((1.-xfrac)*arena[ix][iy+fac][ieta].Wmunu[0][2][2] + xfrac*arena[ix+fac][iy+fac][ieta].Wmunu[0][2][2]);

          if (ieta<neta-fac)
            {
              WX3 = ((1.-xfrac)*arena[ix][iy][ieta+fac].Wmunu[0][2][2] + xfrac*arena[ix+fac][iy][ieta+fac].Wmunu[0][2][2]);
              WX4 = ((1.-xfrac)*arena[ix][iy+fac][ieta+fac].Wmunu[0][2][2] + xfrac*arena[ix+fac][iy+fac][ieta+fac].Wmunu[0][2][2]);
            }
          else
            {
              WX3 = ((1.-xfrac)*Rneighbor_Wyy[ix][iy] + xfrac*Rneighbor_Wyy[ix+fac][iy]);
              WX4 = ((1.-xfrac)*Rneighbor_Wyy[ix][iy+fac] + xfrac*Rneighbor_Wyy[ix+fac][iy+fac]);
            }
          
          WY1 = ((1.-yfrac)*WX1+yfrac*WX2);
          WY2 = ((1.-yfrac)*WX3+yfrac*WX4);

          Wyy1 = ((1.-etafrac)*WY1+etafrac*WY2);
        
          WX1 = ((1.-xfrac)*arena[ix][iy][ieta].W_prev[2][2] + xfrac*arena[ix+fac][iy][ieta].W_prev[2][2]);
          WX2 = ((1.-xfrac)*arena[ix][iy+fac][ieta].W_prev[2][2] + xfrac*arena[ix+fac][iy+fac][ieta].W_prev[2][2]);

          if (ieta<neta-fac)
            {
              WX3 = ((1.-xfrac)*arena[ix][iy][ieta+fac].W_prev[2][2] + xfrac*arena[ix+fac][iy][ieta+fac].W_prev[2][2]);
              WX4 = ((1.-xfrac)*arena[ix][iy+fac][ieta+fac].W_prev[2][2] + xfrac*arena[ix+fac][iy+fac][ieta+fac].W_prev[2][2]);
            }
          else
            {
              WX3 = ((1.-xfrac)*Rneighbor_Wyy_prev[ix][iy] + xfrac*Rneighbor_Wyy_prev[ix+fac][iy]);
              WX4 = ((1.-xfrac)*Rneighbor_Wyy_prev[ix][iy+fac] + xfrac*Rneighbor_Wyy_prev[ix+fac][iy+fac]);
            }
          
          WY1 = ((1.-yfrac)*WX1+yfrac*WX2);
          WY2 = ((1.-yfrac)*WX3+yfrac*WX4);

          Wyy2 = ((1.-etafrac)*WY1+etafrac*WY2);
          
          Wyy = (1.-taufrac)*Wyy2+taufrac*Wyy1;

          //Wyeta:
          WX1 = ((1.-xfrac)*arena[ix][iy][ieta].Wmunu[0][2][3] + xfrac*arena[ix+fac][iy][ieta].Wmunu[0][2][3]);
          WX2 = ((1.-xfrac)*arena[ix][iy+fac][ieta].Wmunu[0][2][3] + xfrac*arena[ix+fac][iy+fac][ieta].Wmunu[0][2][3]);

          if (ieta<neta-fac)
            {
              WX3 = ((1.-xfrac)*arena[ix][iy][ieta+fac].Wmunu[0][2][3] + xfrac*arena[ix+fac][iy][ieta+fac].Wmunu[0][2][3]);
              WX4 = ((1.-xfrac)*arena[ix][iy+fac][ieta+fac].Wmunu[0][2][3] + xfrac*arena[ix+fac][iy+fac][ieta+fac].Wmunu[0][2][3]);
            }
          else
            {
              WX3 = ((1.-xfrac)*Rneighbor_Wyeta[ix][iy] + xfrac*Rneighbor_Wyeta[ix+fac][iy]);
              WX4 = ((1.-xfrac)*Rneighbor_Wyeta[ix][iy+fac] + xfrac*Rneighbor_Wyeta[ix+fac][iy+fac]);
            }
          
          WY1 = ((1.-yfrac)*WX1+yfrac*WX2);
          WY2 = ((1.-yfrac)*WX3+yfrac*WX4);

          Wyeta1 = ((1.-etafrac)*WY1+etafrac*WY2);

          WX1 = ((1.-xfrac)*arena[ix][iy][ieta].W_prev[2][3] + xfrac*arena[ix+fac][iy][ieta].W_prev[2][3]);
          WX2 = ((1.-xfrac)*arena[ix][iy+fac][ieta].W_prev[2][3] + xfrac*arena[ix+fac][iy+fac][ieta].W_prev[2][3]);

          if (ieta<neta-fac)
            {
              WX3 = ((1.-xfrac)*arena[ix][iy][ieta+fac].W_prev[2][3] + xfrac*arena[ix+fac][iy][ieta+fac].W_prev[2][3]);
              WX4 = ((1.-xfrac)*arena[ix][iy+fac][ieta+fac].W_prev[2][3] + xfrac*arena[ix+fac][iy+fac][ieta+fac].W_prev[2][3]);
            }
          else
            {
              WX3 = ((1.-xfrac)*Rneighbor_Wyeta_prev[ix][iy] + xfrac*Rneighbor_Wyeta_prev[ix+fac][iy]);
              WX4 = ((1.-xfrac)*Rneighbor_Wyeta_prev[ix][iy+fac] + xfrac*Rneighbor_Wyeta_prev[ix+fac][iy+fac]);
            }
          
          WY1 = ((1.-yfrac)*WX1+yfrac*WX2);
          WY2 = ((1.-yfrac)*WX3+yfrac*WX4);

          Wyeta2 = ((1.-etafrac)*WY1+etafrac*WY2);

          Wyeta = (1.-taufrac)*Wyeta2+taufrac*Wyeta1;

          //Wetaeta:
          WX1 = ((1.-xfrac)*arena[ix][iy][ieta].Wmunu[0][3][3] + xfrac*arena[ix+fac][iy][ieta].Wmunu[0][3][3]);
          WX2 = ((1.-xfrac)*arena[ix][iy+fac][ieta].Wmunu[0][3][3] + xfrac*arena[ix+fac][iy+fac][ieta].Wmunu[0][3][3]);

          if (ieta<neta-fac)
            {
              WX3 = ((1.-xfrac)*arena[ix][iy][ieta+fac].Wmunu[0][3][3] + xfrac*arena[ix+fac][iy][ieta+fac].Wmunu[0][3][3]);
              WX4 = ((1.-xfrac)*arena[ix][iy+fac][ieta+fac].Wmunu[0][3][3] + xfrac*arena[ix+fac][iy+fac][ieta+fac].Wmunu[0][3][3]);
            }
          else
            {
              WX3 = ((1.-xfrac)*Rneighbor_Wetaeta[ix][iy] + xfrac*Rneighbor_Wetaeta[ix+fac][iy]);
              WX4 = ((1.-xfrac)*Rneighbor_Wetaeta[ix][iy+fac] + xfrac*Rneighbor_Wetaeta[ix+fac][iy+fac]);
            }
          
          WY1 = ((1.-yfrac)*WX1+yfrac*WX2);
          WY2 = ((1.-yfrac)*WX3+yfrac*WX4);

          Wetaeta1 = ((1.-etafrac)*WY1+etafrac*WY2);
          
          WX1 = ((1.-xfrac)*arena[ix][iy][ieta].W_prev[3][3] + xfrac*arena[ix+fac][iy][ieta].W_prev[3][3]);
          WX2 = ((1.-xfrac)*arena[ix][iy+fac][ieta].W_prev[3][3] + xfrac*arena[ix+fac][iy+fac][ieta].W_prev[3][3]);

          if (ieta<neta-fac)
            {
              WX3 = ((1.-xfrac)*arena[ix][iy][ieta+fac].W_prev[3][3] + xfrac*arena[ix+fac][iy][ieta+fac].W_prev[3][3]);
              WX4 = ((1.-xfrac)*arena[ix][iy+fac][ieta+fac].W_prev[3][3] + xfrac*arena[ix+fac][iy+fac][ieta+fac].W_prev[3][3]);
            }
          else
            {
              WX3 = ((1.-xfrac)*Rneighbor_Wetaeta_prev[ix][iy] + xfrac*Rneighbor_Wetaeta_prev[ix+fac][iy]);
              WX4 = ((1.-xfrac)*Rneighbor_Wetaeta_prev[ix][iy+fac] + xfrac*Rneighbor_Wetaeta_prev[ix+fac][iy+fac]);
            }
          
          WY1 = ((1.-yfrac)*WX1+yfrac*WX2);
          WY2 = ((1.-yfrac)*WX3+yfrac*WX4);

          Wetaeta2 = ((1.-etafrac)*WY1+etafrac*WY2);
        
          Wetaeta = (1.-taufrac)*Wetaeta2+taufrac*Wetaeta1;
          
          TFO = eos->get_temperature(epsFO, rhob);
          muB = eos->get_mu(epsFO, rhob);
          if (TFO<0) {
            cout << "TFO=" << TFO << "<0. ERROR. exiting." << endl;
            exit(1);
          }
          
          P=eos->get_pressure(epsFO, rhob);
          eps_plus_p_over_T_FO=(epsFO+P)/TFO;

          if (fabs(FULLSU[0])>DX*DY*DETA+0.01)
            {
              cerr << "problem: volume in tau direction " << fabs(FULLSU[0]) << "  > DX*DY*DETA = "  << DX*DY*DETA << endl;
              //FULLSU[0] = DX*DY*DETA*(FULLSU[0])/fabs(FULLSU[0]);
            }
          if (fabs(FULLSU[1])>DTAU*DY*DETA+0.01)
            {
              cerr << "problem: volume in x direction " << fabs(FULLSU[1]) << "  > DTAU*DY*DETA = "  << DTAU*DY*DETA << endl;
              //FULLSU[1] = DTAU*DY*DETA*(FULLSU[1])/fabs(FULLSU[1]);
            }
          if (fabs(FULLSU[2])>DX*DTAU*DETA+0.01)
            {
              cerr << "problem: volume in y direction " << fabs(FULLSU[2]) << "  > DX*DTAU*DETA = "  << DX*DTAU*DETA << endl;
              //FULLSU[2] = DX*DTAU*DETA*(FULLSU[2])/fabs(FULLSU[2]);
            }
          if (fabs(FULLSU[3])>DX*DY*DTAU+0.01)
            {
              cerr << "problem: volume in eta direction " << fabs(FULLSU[3]) << "  > DX*DY*DTAU = "  << DX*DY*DTAU << endl;
              //FULLSU[3] = DX*DY*DTAU*(FULLSU[3])/fabs(FULLSU[3]);
            }
        
          s_file << setprecision(10) << tauf << " " << xf << " " << yf << " " << etaf << " " 
             << FULLSU[0] << " " <<FULLSU[1] << " " <<FULLSU[2] << " " <<FULLSU[3] 
             << " " <<  utau << " " << ux << " " << uy << " " << ueta << " " 
             << epsFO << " " << TFO << " " << muB << " " << eps_plus_p_over_T_FO << " " 
             << Wtautau << " " << Wtaux << " " << Wtauy << " " << Wtaueta << " " 
             << Wxx << " " << Wxy << " " << Wxeta << " " << Wyy << " " << Wyeta << " " << Wetaeta << endl;
          
            //        fprintf(s_file,"%e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e \n",
            //    tauf, xf, yf, etaf, FULLSU[0],FULLSU[1],FULLSU[2],FULLSU[3],
            //    utau, ux, uy, ueta, epsFO, TFO, muB, sFO, Wtautau, Wtaux, Wtauy, Wtaueta, Wxx, Wxy, Wxeta, Wyy, Wyeta, Wetaeta);
          
          if(fabs(x)<10. && (fabs(y)<10.))
            //if(fabs(x)<0.05 && (fabs(y)<0.05))
            {
              //      fprintf(t_file,"%f %f %f %f %f %f %f %f %f %f %f\n",tauf, xf, yf, etaf, FULLSU[0],FULLSU[1],FULLSU[2],FULLSU[3], Wxx, Wxy, Wxeta);
              t_file << tauf  << " " << xf << " " << yf << " " << etaf << " " 
                 << FULLSU[0] << " " <<FULLSU[1] << " " <<FULLSU[2] << " " <<FULLSU[3] << " " 
                 << Wxx << " " << Wxy << " " << Wxeta << endl;
            }
          if(fabs(eta)<0.01)
            {
              t2_file << tauf << " " << xf << " " << yf << " " << etaf 
                  << " " << FULLSU[0] << " " <<FULLSU[1] << " " <<FULLSU[2] << " " <<FULLSU[3] << " " 
                  << Wtautau << " " << Wtaux << " " << Wtauy << " " << Wtaueta << " " 
                  << Wxx << " " << Wxy << " " << Wxeta << " " << Wyy << " " 
                  << Wyeta << " " << Wetaeta << " " <<epsFO << " " << P << endl;

              //              fprintf(t2_file,"%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n",tauf, xf, yf, etaf, FULLSU[0],FULLSU[1],FULLSU[2],FULLSU[3],Wtautau, Wtaux, Wtauy, Wtaueta, Wxx, Wxy, Wxeta, Wyy, Wyeta, Wetaeta,epsFO,P);
            }
          if(fabs(eta)<0.05 && (fabs(x)<0.1))
            {
              t3_file << tauf << " " << xf << " " << yf << " " << etaf << " " 
                  << FULLSU[0] << " " <<FULLSU[1] << " " <<FULLSU[2] << " " << FULLSU[3] << endl;
              //              fprintf(t3_file << " " <<"%f %f %f %f %f %f %f %f\n", tauf, xf, yf, etaf, FULLSU[0],FULLSU[1],FULLSU[2],FULLSU[3]);
            }
        }
        }
    }
    }
  
  int intersectionsArray[1];
  int allIntersectionsArray[1];
  intersectionsArray[0] = intersections;
  if (rank!=0) //send to rank 0
    {
      MPI::COMM_WORLD.Send(intersectionsArray,1,MPI::INT,0,1);
    }
  if (rank==0) // receive from all ranks >0
    {
      //fprintf(stderr,"percent error=%f\n", 100*(warnings*1.0)/(cells*1.0));
          
      int allIntersections = intersections;
      for (from = 1; from<size; from ++)
    {
      MPI::COMM_WORLD.Recv(intersectionsArray,1,MPI::INT,from,1);
      allIntersections+=intersectionsArray[0];
    }
      for (from = 1; from<size; from ++)
    {
      allIntersectionsArray[0]=allIntersections;
      MPI::COMM_WORLD.Send(allIntersectionsArray,1,MPI::INT,from,2);
    }
      if (allIntersections==0)
    {
      cout << "All cells frozen out. Exiting." << endl;
      // write OSCAR header if wanted:
      FILE *oout_file;
      const char* oout_name = "OSCARheader.dat";
      oout_file = fopen(oout_name, "w");
      if(DATA->outputEvolutionData)
        {
          if(DATA->viscosity_flag==1)
        fprintf(oout_file,"OSCAR2008H viscous history\n");
          else
        fprintf(oout_file,"OSCAR2008H ideal history\n");

          fprintf(oout_file,"INIT: MUSIC %s+%s at b=%f fm, Glauber\n",DATA->Target.c_str(), DATA->Projectile.c_str(),DATA->b);
          fprintf(oout_file,"INIT: \n");
          
          if (DATA->whichEOS==0)
        {
          fprintf(oout_file,"EOS: ideal gas EOS \n");
        }
          else if (DATA->whichEOS==1)
        {
            fprintf(oout_file,"EOS: EOS-Q from AZHYDRO \n");
        }
          else if (DATA->whichEOS==2)
        {
            fprintf(oout_file,"EOS: lattice EOS from Huovinen/Petreczky \n");
        }
          else if (DATA->whichEOS==3)
        {
            fprintf(oout_file,"EOS: lattice EOS from Huovinen/Petreczky with partial chemical equilibrium (PCE) chem. f.o. at 150 MeV\n");
        }
          else if (DATA->whichEOS==4)
        {
            fprintf(oout_file,"EOS: lattice EOS from Huovinen/Petreczky with partial chemical equilibrium (PCE) chem. f.o. at 155 MeV\n");
        }
          else if (DATA->whichEOS==5)
        {
            fprintf(oout_file,"EOS: lattice EOS from Huovinen/Petreczky with partial chemical equilibrium (PCE) chem. f.o. at 160 MeV\n");
        }
          else if (DATA->whichEOS==6)
        {
            fprintf(oout_file,"EOS: lattice EOS from Huovinen/Petreczky with partial chemical equilibrium (PCE) chem. f.o. at 165 MeV\n");
        }

          if(DATA->turn_on_rhob==0)
        fprintf(oout_file,"CHARGES: none\n");
          else
        fprintf(oout_file,"CHARGES: baryon\n");

          fprintf(oout_file,"HYPER: full evolution\n");
 
          fprintf(oout_file,"GEOM: 3d\n");
       
          fprintf(oout_file,"GRID: Euler\n");
 
          fprintf(oout_file,"%d %d %d %d %d %d %d\n", static_cast<int>(((tau-DATA->tau0)/static_cast<double>(DATA->delta_tau)+1)/10.)+1, 
              DATA->nx, DATA->ny, DATA->neta, 0, 0, 0);
        
          fprintf(oout_file,"%f %f %f %f %f %f %f %f\n", DATA->tau0, tau, -DATA->x_size/2., DATA->x_size/2., -DATA->y_size/2., DATA->y_size/2.,
              -DATA->eta_size/2., DATA->eta_size/2.);

          if(DATA->viscosity_flag==1)
        {
          if(DATA->turn_on_shear && DATA->turn_on_bulk)
            {
              fprintf(oout_file,"VISCOSITY: shear and bulk viscosity\n");
              fprintf(oout_file,"VISCOSITY: eta/s = %f, zeta/s = %f\n", DATA->shear_to_s, DATA->bulk_to_s);
            }
          else if(DATA->turn_on_shear==1 && DATA->turn_on_bulk==0)
            {
              fprintf(oout_file,"VISCOSITY: shear viscosity only\n");
              fprintf(oout_file,"VISCOSITY: eta/s = %f\n", DATA->shear_to_s);
            }
          else if(DATA->turn_on_shear==0 && DATA->turn_on_bulk==1)
            {
              fprintf(oout_file,"VISCOSITY: bulk viscosity only\n");
              fprintf(oout_file,"VISCOSITY: zeta/s = %f\n", DATA->bulk_to_s);
            }
          else if(DATA->turn_on_shear==0 && DATA->turn_on_bulk==0)
            {
              fprintf(oout_file,"VISCOSITY: none\n");
              fprintf(oout_file,"VISCOSITY: \n");
            }
        }
          else
        {
          fprintf(oout_file,"VISCOSITY: none\n");
          fprintf(oout_file,"VISCOSITY: \n");
        }
          
          fprintf(oout_file,"COMM:\n");

          fprintf(oout_file,"END_OF_HEADER\n");
        }
          
      fclose(oout_file);
      
      int check=system ("cat OSCARheader.dat OSCAR.dat >OSCARoutput.dat");
      if(check==0)
        {
          system ("rm OSCAR.dat");
          system ("rm OSCARheader.dat");
        }
      
      MPI::Finalize();
      exit(1);
    }
    }
  if (rank!=0)
    {
      MPI::COMM_WORLD.Recv(allIntersectionsArray,1,MPI::INT,0,2);
      if (allIntersectionsArray[0]==0)
    {
      cout << "All cells frozen out. Exiting." << endl;
      MPI::Finalize();
      exit(1);
    }
    }

  delete[] packageWtautau;
  delete[] packageWtaux;
  delete[] packageWtauy;
  delete[] packageWtaueta;
  delete[] packageWxx;
  delete[] packageWxy;
  delete[] packageWxeta;
  delete[] packageWyy;
  delete[] packageWyeta; 
  delete[] packageWetaeta;

  delete[] packageWtautau_prev;
  delete[] packageWtaux_prev;
  delete[] packageWtauy_prev;
  delete[] packageWtaueta_prev;
  delete[] packageWxx_prev;
  delete[] packageWxy_prev;
  delete[] packageWxeta_prev;
  delete[] packageWyy_prev;
  delete[] packageWyeta_prev; 
  delete[] packageWetaeta_prev;

  delete[] package;
  delete[] package_prev;
  delete[] packageutau;
  delete[] packageux;
  delete[] packageuy;
  delete[] packageueta;
  delete[] packagerhob;

  delete[] packageutau_prev;
  delete[] packageux_prev;
  delete[] packageuy_prev;
  delete[] packageueta_prev;
  delete[] packagerhob_prev;

  util->mtx_free(Rneighbor_eps,nx+1,ny+1);
  util->mtx_free(Rneighbor_eps_prev,nx+1,ny+1);
  util->mtx_free(Rneighbor_utau,nx+1,ny+1);
  util->mtx_free(Rneighbor_ux,nx+1,ny+1);
  util->mtx_free(Rneighbor_uy,nx+1,ny+1);
  util->mtx_free(Rneighbor_ueta,nx+1,ny+1);
  util->mtx_free(Rneighbor_rhob,nx+1,ny+1);
 
  util->mtx_free(Rneighbor_utau_prev,nx+1,ny+1);
  util->mtx_free(Rneighbor_ux_prev,nx+1,ny+1);
  util->mtx_free(Rneighbor_uy_prev,nx+1,ny+1);
  util->mtx_free(Rneighbor_ueta_prev,nx+1,ny+1);
  util->mtx_free(Rneighbor_rhob_prev,nx+1,ny+1);
 
  util->mtx_free(Rneighbor_Wtautau,nx+1,ny+1);
  util->mtx_free(Rneighbor_Wtaux,nx+1,ny+1);
  util->mtx_free(Rneighbor_Wtauy,nx+1,ny+1);
  util->mtx_free(Rneighbor_Wtaueta,nx+1,ny+1);
  util->mtx_free(Rneighbor_Wxx,nx+1,ny+1);
  util->mtx_free(Rneighbor_Wxy,nx+1,ny+1);
  util->mtx_free(Rneighbor_Wxeta,nx+1,ny+1);
  util->mtx_free(Rneighbor_Wyy,nx+1,ny+1);
  util->mtx_free(Rneighbor_Wyeta,nx+1,ny+1);
  util->mtx_free(Rneighbor_Wetaeta,nx+1,ny+1);
  
  util->mtx_free(Rneighbor_Wtautau_prev,nx+1,ny+1);
  util->mtx_free(Rneighbor_Wtaux_prev,nx+1,ny+1);
  util->mtx_free(Rneighbor_Wtauy_prev,nx+1,ny+1);
  util->mtx_free(Rneighbor_Wtaueta_prev,nx+1,ny+1);
  util->mtx_free(Rneighbor_Wxx_prev,nx+1,ny+1);
  util->mtx_free(Rneighbor_Wxy_prev,nx+1,ny+1);
  util->mtx_free(Rneighbor_Wxeta_prev,nx+1,ny+1);
  util->mtx_free(Rneighbor_Wyy_prev,nx+1,ny+1);
  util->mtx_free(Rneighbor_Wyeta_prev,nx+1,ny+1);
  util->mtx_free(Rneighbor_Wetaeta_prev,nx+1,ny+1);
 
  t_file.close();
  t2_file.close();
  t3_file.close();
  s_file.close();

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

