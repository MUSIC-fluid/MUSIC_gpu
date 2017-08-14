// Copyright 2011 @ Bjoern Schenke, Sangyong Jeon, and Charles Gale
#include <omp.h>
#include "./util.h"
#include "./grid.h"
#include "./init.h"
#include "./eos.h"

using namespace std;

Init::Init(EOS *eosIn, InitData *DATA_in) {
    eos = eosIn;
    util = new Util;
    DATA_ptr = DATA_in;
}

// destructor
Init::~Init() {
    delete util;
}

void Init::initialize_hydro_fields(Field *hydro_fields, InitData *DATA) {
    int n_cell = GRID_SIZE_ETA*(GRID_SIZE_X + 1)*(GRID_SIZE_Y + 1);
    hydro_fields->e_rk0 = new double[n_cell];
    hydro_fields->e_rk1 = new double[n_cell];
    hydro_fields->e_prev = new double[n_cell];
    hydro_fields->rhob_rk0 = new double[n_cell];
    hydro_fields->rhob_rk1 = new double[n_cell];
    hydro_fields->rhob_prev = new double[n_cell];
    hydro_fields->u_rk0 = new double* [4];
    hydro_fields->u_rk1 = new double* [4];
    hydro_fields->u_prev = new double* [4];
    hydro_fields->dUsup = new double* [20];
    hydro_fields->Wmunu_rk0 = new double* [14];
    hydro_fields->Wmunu_rk1 = new double* [14];
    hydro_fields->Wmunu_prev = new double* [14];
    for (int i = 0; i < 4; i++) {
        hydro_fields->u_rk0[i] = new double[n_cell];
        hydro_fields->u_rk1[i] = new double[n_cell];
        hydro_fields->u_prev[i] = new double[n_cell];
    }
    for (int i = 0; i < 20; i++) {
        hydro_fields->dUsup[i] = new double[n_cell];
    }
    for (int i = 0; i < 14; i++) {
        hydro_fields->Wmunu_rk0[i] = new double[n_cell];
        hydro_fields->Wmunu_rk1[i] = new double[n_cell];
        hydro_fields->Wmunu_prev[i] = new double[n_cell];
    }
    hydro_fields->pi_b_rk0 = new double[n_cell];
    hydro_fields->pi_b_rk1 = new double[n_cell];
    hydro_fields->pi_b_prev = new double[n_cell];
}

void Init::InitArena(InitData *DATA, Field *hydro_fields) {
    //Grid *helperGrid;
    //helperGrid = new Grid;
    cout << "initArena" << endl;
    if (DATA->Initial_profile <= 1) {
        cout << "Using Initial_profile=" << DATA->Initial_profile << endl;
        //DATA->nx = DATA->nx - 1;
        //DATA->ny = DATA->ny - 1;
        cout << "nx=" << DATA->nx+1 << ", ny=" << DATA->ny+1 << endl;
        cout << "dx=" << DATA->delta_x << ", dy=" << DATA->delta_y << endl;
    } else if (DATA->Initial_profile == 8) {
        cout << DATA->initName <<endl;
        ifstream profile(DATA->initName.c_str());
        string dummy;
        int nx, ny, neta;
        double deta, dx, dy, dummy2;
        // read the first line with general info
        profile >> dummy >> dummy >> dummy2
                >> dummy >> neta >> dummy >> nx >> dummy >> ny
                >> dummy >> deta >> dummy >> dx >> dummy >> dy;
        profile.close();
        cout << "Using Initial_profile=" << DATA->Initial_profile
             << ". Overwriting lattice dimensions:" << endl;

        DATA->nx = nx - 1;
        DATA->ny = ny - 1;
        DATA->delta_x = dx;
        DATA->delta_y = dy;

        cout << "neta=" << neta << ", nx=" << nx << ", ny=" << ny << endl;
        cout << "deta=" << DATA->delta_eta << ", dx=" << DATA->delta_x
             << ", dy=" << DATA->delta_y << endl;
    }

    // initialize arena
    //*arena = helperGrid->grid_c_malloc(DATA->neta, DATA->nx + 1, DATA->ny + 1);
    initialize_hydro_fields(hydro_fields, DATA);
    cout << "Grid allocated." << endl;

    InitTJb(DATA, hydro_fields);

    //if (DATA->output_initial_density_profiles == 1) {
    //    output_initial_density_profiles(DATA, *arena);
    //}

    //LinkNeighbors(DATA, arena);
    //delete helperGrid;
}/* InitArena */


void Init::LinkNeighbors(InitData *DATA, Grid ****arena) {
    int nx = DATA->nx;
    int ny = DATA->ny;
    int neta = DATA->neta;

    /* allocate memory */
    for (int ieta = 0; ieta < neta; ieta++) {
        for (int ix = 0; ix <= nx; ix++) {
            for (int iy = 0; iy <= ny; iy++) {
                (*arena)[ieta][ix][iy].nbr_p_1 = new Grid *[4];
                (*arena)[ieta][ix][iy].nbr_m_1 = new Grid *[4];
                (*arena)[ieta][ix][iy].nbr_p_2 = new Grid *[4];
                (*arena)[ieta][ix][iy].nbr_m_2 = new Grid *[4];
            }
        }
    }

    int ieta;
    #pragma omp parallel private(ieta)
    {
        #pragma omp for
        for (ieta = 0; ieta < neta; ieta++) {
            //printf("Thread %d executes loop iteraction %d\n",
            //       omp_get_thread_num(), ieta);
            LinkNeighbors_XY(DATA, ieta, (*arena));
        }
    }
}  /* LinkNeighbors */

void Init::LinkNeighbors_XY(InitData *DATA, int ieta, Grid ***arena) {
    int nx = DATA->nx;
    int ny = DATA->ny;
    int neta = DATA->neta;
    for (int ix = 0; ix <= nx; ix++) {
        for (int iy = 0; iy <= ny; iy++) {
            if (ix != nx)
                arena[ieta][ix][iy].nbr_p_1[1] = &arena[ieta][ix+1][iy];
            else
                arena[ieta][ix][iy].nbr_p_1[1] = &arena[ieta][nx][iy];
            if (ix < nx - 1)
                arena[ieta][ix][iy].nbr_p_2[1] = &arena[ieta][ix+2][iy];
            else
                arena[ieta][ix][iy].nbr_p_2[1] = &arena[ieta][nx][iy];
            if (ix != 0)
                arena[ieta][ix][iy].nbr_m_1[1] = &arena[ieta][ix-1][iy];
            else
                arena[ieta][ix][iy].nbr_m_1[1] = &arena[ieta][0][iy];
            if (ix > 1)
                arena[ieta][ix][iy].nbr_m_2[1] = &arena[ieta][ix-2][iy];
            else
                arena[ieta][ix][iy].nbr_m_2[1] = &arena[ieta][0][iy];
            if (iy != ny)
                arena[ieta][ix][iy].nbr_p_1[2] = &arena[ieta][ix][iy+1];
            else
                arena[ieta][ix][iy].nbr_p_1[2] = &arena[ieta][ix][ny];
            if (iy < ny - 1)
                arena[ieta][ix][iy].nbr_p_2[2] = &arena[ieta][ix][iy+2];
            else
                arena[ieta][ix][iy].nbr_p_2[2] = &arena[ieta][ix][ny];
            if (iy != 0)
                arena[ieta][ix][iy].nbr_m_1[2] = &arena[ieta][ix][iy-1];
            else
                arena[ieta][ix][iy].nbr_m_1[2] = &arena[ieta][ix][0];
            if (iy > 1)
                arena[ieta][ix][iy].nbr_m_2[2] = &arena[ieta][ix][iy-2];
            else
                arena[ieta][ix][iy].nbr_m_2[2] = &arena[ieta][ix][0];

            if (ieta != neta-1)
                arena[ieta][ix][iy].nbr_p_1[3] = &arena[ieta+1][ix][iy];
            else
                arena[ieta][ix][iy].nbr_p_1[3] = &arena[neta-1][ix][iy];
            if (ieta < neta-2)
                arena[ieta][ix][iy].nbr_p_2[3] = &arena[ieta+2][ix][iy];
            else
                arena[ieta][ix][iy].nbr_p_2[3] = &arena[neta-1][ix][iy];
            if (ieta != 0)
                arena[ieta][ix][iy].nbr_m_1[3] = &arena[ieta-1][ix][iy];
            else
                arena[ieta][ix][iy].nbr_m_1[3] = &arena[0][ix][iy];
            if (ieta > 1)
                arena[ieta][ix][iy].nbr_m_2[3] = &arena[ieta-2][ix][iy];
            else
                arena[ieta][ix][iy].nbr_m_2[3] = &arena[0][ix][iy];
        }
    }
}

int Init::InitTJb(InitData *DATA, Field *hydro_fields) {
    int rk_order = DATA->rk_order;
    cout << "rk_order=" << rk_order << endl;
    if (DATA->Initial_profile == 0) {
        // Gubser flow test
        cout << " Perform Gubser flow test ... " << endl;
        cout << " ----- information on initial distribution -----" << endl;
        
        int ieta;
        #pragma omp parallel private(ieta)
        {
            #pragma omp for
            for (ieta = 0; ieta < DATA->neta; ieta++) {
                printf("Thread %d executes loop iteraction ieta = %d\n",
                       omp_get_thread_num(), ieta);
                initial_Gubser_XY(DATA, ieta, hydro_fields);
            }/* ieta */
            #pragma omp barrier
        }
    } else if (DATA->Initial_profile == 1) {
        // Gubser flow test
        cout << " Perform Bjorken flow test ... " << endl;
        cout << " ----- information on initial distribution -----" << endl;
        
        int ieta;
        #pragma omp parallel private(ieta)
        {
            #pragma omp for
            for (ieta = 0; ieta < DATA->neta; ieta++) {
                printf("Thread %d executes loop iteraction ieta = %d\n",
                       omp_get_thread_num(), ieta);
                initial_Bjorken_XY(DATA, ieta, hydro_fields);
            }/* ieta */
            #pragma omp barrier
        }
    } else if (DATA->Initial_profile == 8) {
        // read in the profile from file
        // - IPGlasma initial conditions with initial flow
        cout << " ----- information on initial distribution -----" << endl;
        cout << "file name used: " << DATA->initName << endl;
  
        int ieta;
        #pragma omp parallel private(ieta)
        {
            #pragma omp for
            for (ieta = 0; ieta < DATA->neta; ieta++) {
                printf("Thread %d executes loop iteraction ieta = %d\n",
                       omp_get_thread_num(), ieta);
                initial_IPGlasma_XY(DATA, ieta, hydro_fields);
            } /* ieta */
            #pragma omp barrier
        }
    }
    cout << "initial distribution done." << endl;
    return 1;
}  /* InitTJb*/

void Init::initial_Gubser_XY(InitData *DATA, int ieta, Field *hydro_fields) {
    string input_filename;
    string input_filename_prev;
    if (DATA->turn_on_shear == 1) {
        input_filename = "tests/Gubser_flow/Initial_Profile.dat";
    } else {
        input_filename = "tests/Gubser_flow/y=0_tau=1.00_ideal.dat";
        input_filename_prev = "tests/Gubser_flow/y=0_tau=0.98_ideal.dat";
    }
    //if (omp_get_thread_num() == 0) {
    //    cout << "file name used: " << input_filename << endl;
    //}
    
    ifstream profile(input_filename.c_str());
    if (!profile.good()) {
        cout << "Init::InitTJb: "
             << "Can not open the initial file: " << input_filename
             << endl;
        exit(1);
    }
    ifstream profile_prev;
    if (DATA->turn_on_shear == 0) {
        profile_prev.open(input_filename_prev.c_str());
        if (!profile_prev.good()) {
            cout << "Init::InitTJb: "
                 << "Can not open the initial file: " << input_filename_prev
                 << endl;
            exit(1);
        }
    }

    int nx = DATA->nx + 1;
    int ny = DATA->ny + 1;
    double** temp_profile_ed = new double* [nx];
    double** temp_profile_ux = new double* [nx];
    double** temp_profile_uy = new double* [nx];
    double **temp_profile_ed_prev = NULL;
    double **temp_profile_rhob = NULL;
    double **temp_profile_rhob_prev = NULL;
    double **temp_profile_ux_prev = NULL;
    double **temp_profile_uy_prev = NULL;
    double **temp_profile_pixx = NULL;
    double **temp_profile_piyy = NULL;
    double **temp_profile_pixy = NULL;
    double **temp_profile_pi00 = NULL;
    double **temp_profile_pi0x = NULL;
    double **temp_profile_pi0y = NULL;
    double **temp_profile_pi33 = NULL;
    if (DATA->turn_on_shear == 1) {
        temp_profile_pixx = new double* [nx];
        temp_profile_piyy = new double* [nx];
        temp_profile_pixy = new double* [nx];
        temp_profile_pi00 = new double* [nx];
        temp_profile_pi0x = new double* [nx];
        temp_profile_pi0y = new double* [nx];
        temp_profile_pi33 = new double* [nx];
    } else {
        temp_profile_ed_prev = new double* [nx];
        temp_profile_rhob = new double* [nx];
        temp_profile_rhob_prev = new double* [nx];
        temp_profile_ux_prev = new double* [nx];
        temp_profile_uy_prev = new double* [nx];
    }
    for (int i = 0; i < nx; i++) {
        temp_profile_ed[i] = new double[ny];
        temp_profile_ux[i] = new double[ny];
        temp_profile_uy[i] = new double[ny];
        if (DATA->turn_on_shear == 1) {
            temp_profile_pixx[i] = new double[ny];
            temp_profile_pixy[i] = new double[ny];
            temp_profile_piyy[i] = new double[ny];
            temp_profile_pi00[i] = new double[ny];
            temp_profile_pi0x[i] = new double[ny];
            temp_profile_pi0y[i] = new double[ny];
            temp_profile_pi33[i] = new double[ny];
        } else {
            temp_profile_ed_prev[i] = new double[ny];
            temp_profile_rhob[i] = new double[ny];
            temp_profile_rhob_prev[i] = new double[ny];
            temp_profile_ux_prev[i] = new double[ny];
            temp_profile_uy_prev[i] = new double[ny];
        }
    }

    double dummy;
    double u[4];
    for (int ix = 0; ix < nx; ix++) {
        for (int iy = 0; iy < ny; iy++) {
            if (DATA->turn_on_shear == 1) {
                profile >> dummy >> dummy >> temp_profile_ed[ix][iy]
                        >> temp_profile_ux[ix][iy] >> temp_profile_uy[ix][iy];
                profile >> temp_profile_pixx[ix][iy]
                        >> temp_profile_piyy[ix][iy]
                        >> temp_profile_pixy[ix][iy]
                        >> temp_profile_pi00[ix][iy]
                        >> temp_profile_pi0x[ix][iy]
                        >> temp_profile_pi0y[ix][iy]
                        >> temp_profile_pi33[ix][iy];
            } else {
                profile >> dummy >> dummy >> temp_profile_ed[ix][iy]
                        >> temp_profile_rhob[ix][iy]
                        >> temp_profile_ux[ix][iy] >> temp_profile_uy[ix][iy];
                profile_prev >> dummy >> dummy >> temp_profile_ed_prev[ix][iy]
                             >> temp_profile_rhob_prev[ix][iy]
                             >> temp_profile_ux_prev[ix][iy]
                             >> temp_profile_uy_prev[ix][iy];
            }
        }
    }
    profile.close();
    if (DATA->turn_on_shear == 0) {
        profile_prev.close();
    }

    for (int ix = 0; ix < nx; ix++) {
        for (int iy = 0; iy< ny; iy++) {
            int idx = iy + ny*ix + ny*nx*ieta;
            double rhob = 0.0;
            if (DATA->turn_on_shear == 0) {
                if (DATA->turn_on_rhob == 1) {
                    rhob = temp_profile_rhob[ix][iy];
                }
            }

            double epsilon = temp_profile_ed[ix][iy];
            
            // set all values in the grid element:
            hydro_fields->e_rk0[idx] = epsilon;
            hydro_fields->e_rk1[idx] = epsilon;
            hydro_fields->e_prev[idx] = epsilon;
            hydro_fields->rhob_rk0[idx] = rhob;
            hydro_fields->rhob_rk1[idx] = rhob;
            hydro_fields->rhob_prev[idx] = rhob;
            
            /* for HIC */
            double utau_local = sqrt(1.
                          + temp_profile_ux[ix][iy]*temp_profile_ux[ix][iy]
                          + temp_profile_uy[ix][iy]*temp_profile_uy[ix][iy]);
            u[0] = utau_local;
            u[1] = temp_profile_ux[ix][iy];
            u[2] = temp_profile_uy[ix][iy];
            u[3] = 0.0;
            hydro_fields->u_rk0[0][idx] = u[0];
            hydro_fields->u_rk0[1][idx] = u[1];
            hydro_fields->u_rk0[2][idx] = u[2];
            hydro_fields->u_rk0[3][idx] = u[3];
            hydro_fields->u_prev[0][idx] = u[0];
            hydro_fields->u_prev[1][idx] = u[1];
            hydro_fields->u_prev[2][idx] = u[2];
            hydro_fields->u_prev[3][idx] = u[3];

            if (DATA->turn_on_shear == 0) {
                hydro_fields->e_prev[idx] = temp_profile_ed_prev[ix][iy];
                double utau_prev = sqrt(1.
                    + temp_profile_ux_prev[ix][iy]*temp_profile_ux_prev[ix][iy]
                    + temp_profile_uy_prev[ix][iy]*temp_profile_uy_prev[ix][iy]
                );
                hydro_fields->u_prev[0][idx] = utau_prev;
                hydro_fields->u_prev[1][idx] = temp_profile_ux_prev[ix][iy];
                hydro_fields->u_prev[2][idx] = temp_profile_uy_prev[ix][iy];
                hydro_fields->u_prev[3][idx] = 0.0;
            }
            hydro_fields->pi_b_prev[idx] = 0.0;
            hydro_fields->pi_b_rk0[idx] = 0.0;

            if (DATA->turn_on_shear == 1) {
                hydro_fields->Wmunu_rk0[0][idx] = temp_profile_pi00[ix][iy];
                hydro_fields->Wmunu_rk0[1][idx] = temp_profile_pi0x[ix][iy];
                hydro_fields->Wmunu_rk0[2][idx] = temp_profile_pi0y[ix][iy];
                hydro_fields->Wmunu_rk0[3][idx] = 0.0;
                hydro_fields->Wmunu_rk0[4][idx] = temp_profile_pixx[ix][iy];
                hydro_fields->Wmunu_rk0[5][idx] = temp_profile_pixy[ix][iy];
                hydro_fields->Wmunu_rk0[6][idx] = 0.0;
                hydro_fields->Wmunu_rk0[7][idx] = temp_profile_piyy[ix][iy];
                hydro_fields->Wmunu_rk0[8][idx] = 0.0;
                hydro_fields->Wmunu_rk0[9][idx] = temp_profile_pi33[ix][iy];
                for (int mu = 10; mu < 14; mu++) {
                        hydro_fields->Wmunu_rk0[mu][idx] = 0.0;
                }
            } else {
                for (int mu = 0; mu < 14; mu++) {
                        hydro_fields->Wmunu_rk0[mu][idx] = 0.0;
                }
            }
            for (int rkstep = 0; rkstep < 1; rkstep++) {
                for (int ii = 0; ii < 14; ii++) {
                    hydro_fields->Wmunu_prev[ii][idx] =
                                        hydro_fields->Wmunu_rk0[ii][idx];
                }
            }
        }
    }
    // clean up
    for (int i = 0; i < nx; i++) {
        delete[] temp_profile_ed[i];
        delete[] temp_profile_ux[i];
        delete[] temp_profile_uy[i];
        if (DATA->turn_on_shear == 1) {
            delete[] temp_profile_pixx[i];
            delete[] temp_profile_piyy[i];
            delete[] temp_profile_pixy[i];
            delete[] temp_profile_pi00[i];
            delete[] temp_profile_pi0x[i];
            delete[] temp_profile_pi0y[i];
            delete[] temp_profile_pi33[i];
        } else {
            delete[] temp_profile_ed_prev[i];
            delete[] temp_profile_rhob[i];
            delete[] temp_profile_rhob_prev[i];
            delete[] temp_profile_ux_prev[i];
            delete[] temp_profile_uy_prev[i];
        }
    }
    delete[] temp_profile_ed;
    delete[] temp_profile_ux;
    delete[] temp_profile_uy;
    if (DATA->turn_on_shear == 1) {
        delete[] temp_profile_pixx;
        delete[] temp_profile_piyy;
        delete[] temp_profile_pixy;
        delete[] temp_profile_pi00;
        delete[] temp_profile_pi0x;
        delete[] temp_profile_pi0y;
        delete[] temp_profile_pi33;
    } else {
        delete[] temp_profile_ed_prev;
        delete[] temp_profile_rhob;
        delete[] temp_profile_rhob_prev;
        delete[] temp_profile_ux_prev;
        delete[] temp_profile_uy_prev;
    }
}


void Init::initial_Bjorken_XY(InitData *DATA, int ieta, Field *hydro_fields) {
    string input_filename;
    string input_filename_prev;
    
    int nx = GRID_SIZE_X + 1;
    int ny = GRID_SIZE_Y + 1;
    double** temp_profile_ed = new double* [nx];
    double** temp_profile_ux = new double* [nx];
    double** temp_profile_uy = new double* [nx];
    double **temp_profile_ed_prev = NULL;
    double **temp_profile_rhob = NULL;
    double **temp_profile_rhob_prev = NULL;
    double **temp_profile_ux_prev = NULL;
    double **temp_profile_uy_prev = NULL;
    double **temp_profile_pixx = NULL;
    double **temp_profile_piyy = NULL;
    double **temp_profile_pixy = NULL;
    double **temp_profile_pi00 = NULL;
    double **temp_profile_pi0x = NULL;
    double **temp_profile_pi0y = NULL;
    double **temp_profile_pi33 = NULL;
        temp_profile_pixx = new double* [nx];
        temp_profile_piyy = new double* [nx];
        temp_profile_pixy = new double* [nx];
        temp_profile_pi00 = new double* [nx];
        temp_profile_pi0x = new double* [nx];
        temp_profile_pi0y = new double* [nx];
        temp_profile_pi33 = new double* [nx];
        temp_profile_ed_prev = new double* [nx];
        temp_profile_rhob = new double* [nx];
        temp_profile_rhob_prev = new double* [nx];
        temp_profile_ux_prev = new double* [nx];
        temp_profile_uy_prev = new double* [nx];
    for (int i = 0; i < nx; i++) {
        temp_profile_ed[i] = new double[ny];
        temp_profile_ux[i] = new double[ny];
        temp_profile_uy[i] = new double[ny];
            temp_profile_pixx[i] = new double[ny];
            temp_profile_pixy[i] = new double[ny];
            temp_profile_piyy[i] = new double[ny];
            temp_profile_pi00[i] = new double[ny];
            temp_profile_pi0x[i] = new double[ny];
            temp_profile_pi0y[i] = new double[ny];
            temp_profile_pi33[i] = new double[ny];
            temp_profile_ed_prev[i] = new double[ny];
            temp_profile_rhob[i] = new double[ny];
            temp_profile_rhob_prev[i] = new double[ny];
            temp_profile_ux_prev[i] = new double[ny];
            temp_profile_uy_prev[i] = new double[ny];
    }

    double dummy;
    double u[4];
    for (int ix = 0; ix < nx; ix++) {
        for (int iy = 0; iy < ny; iy++) {
                temp_profile_ed[ix][iy]=5.;
                temp_profile_rhob[ix][iy]=0.;
                temp_profile_ux[ix][iy]=0.;
                temp_profile_uy[ix][iy]=0.;
                temp_profile_ed_prev[ix][iy]=5.;
                temp_profile_rhob_prev[ix][iy]=0.;
                temp_profile_ux_prev[ix][iy]=0.;
                temp_profile_uy_prev[ix][iy]=0.;
                temp_profile_pixx[ix][iy]=0.0;
                temp_profile_piyy[ix][iy]=0.0;
                temp_profile_pixy[ix][iy]=0.0;
                temp_profile_pi00[ix][iy]=0.0;
                temp_profile_pi0x[ix][iy]=0.0;
                temp_profile_pi0y[ix][iy]=0.0;
                temp_profile_pi33[ix][iy]=0.0;
        }
    }

    for (int ix = 0; ix < nx; ix++) {
        for (int iy = 0; iy< ny; iy++) {
            int idx = iy + ny*ix + ny*nx*ieta;
            double rhob = 0.0;
            if (DATA->turn_on_shear == 0) {
                if (DATA->turn_on_rhob == 1) {
                    rhob = temp_profile_rhob[ix][iy];
                }
            }

            double epsilon = temp_profile_ed[ix][iy];
            
            // set all values in the grid element:
            hydro_fields->e_rk0[idx] = epsilon;
            hydro_fields->e_rk1[idx] = epsilon;
            hydro_fields->e_prev[idx] = epsilon;
            hydro_fields->rhob_rk0[idx] = rhob;
            hydro_fields->rhob_rk1[idx] = rhob;
            hydro_fields->rhob_prev[idx] = rhob;
            
            /* for HIC */
            double utau_local = sqrt(1.
                          + temp_profile_ux[ix][iy]*temp_profile_ux[ix][iy]
                          + temp_profile_uy[ix][iy]*temp_profile_uy[ix][iy]);
            u[0] = utau_local;
            u[1] = temp_profile_ux[ix][iy];
            u[2] = temp_profile_uy[ix][iy];
            u[3] = 0.0;
            hydro_fields->u_rk0[0][idx] = u[0];
            hydro_fields->u_rk0[1][idx] = u[1];
            hydro_fields->u_rk0[2][idx] = u[2];
            hydro_fields->u_rk0[3][idx] = u[3];
            hydro_fields->u_prev[0][idx] = u[0];
            hydro_fields->u_prev[1][idx] = u[1];
            hydro_fields->u_prev[2][idx] = u[2];
            hydro_fields->u_prev[3][idx] = u[3];

                double utau_prev = sqrt(1.
                    + temp_profile_ux_prev[ix][iy]*temp_profile_ux_prev[ix][iy]
                    + temp_profile_uy_prev[ix][iy]*temp_profile_uy_prev[ix][iy]
                );
                hydro_fields->u_prev[0][idx] = utau_prev;
                hydro_fields->u_prev[1][idx] = temp_profile_ux_prev[ix][iy];
                hydro_fields->u_prev[2][idx] = temp_profile_uy_prev[ix][iy];
                hydro_fields->u_prev[3][idx] = 0.0;
            hydro_fields->pi_b_prev[idx] = 0.0;
            hydro_fields->pi_b_rk0[idx] = 0.0;

                hydro_fields->Wmunu_rk0[0][idx] = temp_profile_pi00[ix][iy];
                hydro_fields->Wmunu_rk0[1][idx] = temp_profile_pi0x[ix][iy];
                hydro_fields->Wmunu_rk0[2][idx] = temp_profile_pi0y[ix][iy];
                hydro_fields->Wmunu_rk0[3][idx] = 0.0;
                hydro_fields->Wmunu_rk0[4][idx] = temp_profile_pixx[ix][iy];
                hydro_fields->Wmunu_rk0[5][idx] = temp_profile_pixy[ix][iy];
                hydro_fields->Wmunu_rk0[6][idx] = 0.0;
                hydro_fields->Wmunu_rk0[7][idx] = temp_profile_piyy[ix][iy];
                hydro_fields->Wmunu_rk0[8][idx] = 0.0;
                hydro_fields->Wmunu_rk0[9][idx] = temp_profile_pi33[ix][iy];
                for (int mu = 10; mu < 14; mu++) {
                        hydro_fields->Wmunu_rk0[mu][idx] = 0.0;
                }
                for (int mu = 0; mu < 14; mu++) {
                        hydro_fields->Wmunu_rk0[mu][idx] = 0.0;
                }
            for (int rkstep = 0; rkstep < 1; rkstep++) {
                for (int ii = 0; ii < 14; ii++) {
                    hydro_fields->Wmunu_prev[ii][idx] =
                                        hydro_fields->Wmunu_rk0[ii][idx];
                }
            }
        }
    }
    // clean up
    for (int i = 0; i < nx; i++) {
        delete[] temp_profile_ed[i];
        delete[] temp_profile_ux[i];
        delete[] temp_profile_uy[i];
            delete[] temp_profile_pixx[i];
            delete[] temp_profile_piyy[i];
            delete[] temp_profile_pixy[i];
            delete[] temp_profile_pi00[i];
            delete[] temp_profile_pi0x[i];
            delete[] temp_profile_pi0y[i];
            delete[] temp_profile_pi33[i];
            delete[] temp_profile_ed_prev[i];
            delete[] temp_profile_rhob[i];
            delete[] temp_profile_rhob_prev[i];
            delete[] temp_profile_ux_prev[i];
            delete[] temp_profile_uy_prev[i];
    }
    delete[] temp_profile_ed;
    delete[] temp_profile_ux;
    delete[] temp_profile_uy;
        delete[] temp_profile_pixx;
        delete[] temp_profile_piyy;
        delete[] temp_profile_pixy;
        delete[] temp_profile_pi00;
        delete[] temp_profile_pi0x;
        delete[] temp_profile_pi0y;
        delete[] temp_profile_pi33;
        delete[] temp_profile_ed_prev;
        delete[] temp_profile_rhob;
        delete[] temp_profile_rhob_prev;
        delete[] temp_profile_ux_prev;
        delete[] temp_profile_uy_prev;
}

void Init::initial_IPGlasma_XY(InitData *DATA, int ieta, Field *hydro_fields) {
    ifstream profile(DATA->initName.c_str());

    string dummy;
    int nx, ny, neta;
    double dx, dy, deta;
    // read the information line
    profile >> dummy >> dummy >> dummy >> dummy >> neta
            >> dummy >> nx >> dummy >> ny
            >> dummy >> deta >> dummy >> dx >> dummy >> dy;

    if (omp_get_thread_num() == 0) {
        cout << "neta=" << DATA->neta << ", nx=" << nx << ", ny=" << ny
             << ", deta=" << DATA->delta_eta << ", dx=" << dx << ", dy=" << dy
             << endl;
    }

    double density, dummy1, dummy2, dummy3;
    double ux, uy, utau;

    double** temp_profile_ed = new double* [nx];
    double** temp_profile_utau = new double* [nx];
    double** temp_profile_ux = new double* [nx];
    double** temp_profile_uy = new double* [nx];
    for (int i = 0; i < nx; i++) {
        temp_profile_ed[i] = new double[ny];
        temp_profile_utau[i] = new double[ny];
        temp_profile_ux[i] = new double[ny];
        temp_profile_uy[i] = new double[ny];
    }

    int grid_nx = DATA->nx + 1;
    int grid_ny = DATA->ny + 1;
    // read the one slice
    for (int ix = 0; ix < grid_nx; ix++) {
        for (int iy = 0; iy < grid_ny; iy++) {
            profile >> dummy1 >> dummy2 >> dummy3
                    >> density >> utau >> ux >> uy
                    >> dummy  >> dummy  >> dummy  >> dummy;
            temp_profile_ed[ix][iy] = density;
            temp_profile_utau[ix][iy] = utau;
            temp_profile_ux[ix][iy] = ux;
            temp_profile_uy[ix][iy] = uy;
            if (ix == 0 && iy == 0) {
                DATA->x_size = -dummy2*2;
                DATA->y_size = -dummy3*2;
                if (omp_get_thread_num() == 0) {
                    cout << "eta_size=" << DATA->eta_size
                         << ", x_size=" << DATA->x_size
                         << ", y_size=" << DATA->y_size << endl;
                }
            }
        }
    }
    profile.close();

    double eta = (DATA->delta_eta)*(ieta) - (DATA->eta_size)/2.0;
    double eta_envelop_ed = eta_profile_normalisation(DATA, eta);
    int entropy_flag = DATA->initializeEntropy;
    double u[4];
    for (int ix = 0; ix < grid_nx; ix++) {
        for (int iy = 0; iy< grid_ny; iy++) {
            int idx = iy + ix*grid_ny + ieta*grid_ny*grid_nx;
            double rhob = 0.0;
            double epsilon = 0.0;
            if (entropy_flag == 0) {
                epsilon = (temp_profile_ed[ix][iy]*eta_envelop_ed
                           *DATA->sFactor/hbarc);  // 1/fm^4
            } else {
                double local_sd = (temp_profile_ed[ix][iy]*DATA->sFactor
                                   *eta_envelop_ed);
                epsilon = eos->get_s2e(local_sd, rhob);
            }
            if (epsilon < 0.00000000001)
                epsilon = 0.00000000001;

            // set all values in the grid element:
            hydro_fields->e_rk0[idx] = epsilon;
            hydro_fields->e_rk1[idx] = epsilon;
            hydro_fields->e_prev[idx] = epsilon;
            hydro_fields->rhob_rk0[idx] = rhob;
            hydro_fields->rhob_rk1[idx] = rhob;
            hydro_fields->rhob_prev[idx] = rhob;

            /* for HIC */
            u[1] = temp_profile_ux[ix][iy];
            u[2] = temp_profile_uy[ix][iy];
            u[0] = sqrt(1. + u[1]*u[1] + u[2]*u[2]);
            u[3] = 0.0;
            hydro_fields->u_rk0[0][idx] = u[0];
            hydro_fields->u_rk0[1][idx] = u[1];
            hydro_fields->u_rk0[2][idx] = u[2];
            hydro_fields->u_rk0[3][idx] = u[3];
            hydro_fields->u_prev[0][idx] = u[0];
            hydro_fields->u_prev[1][idx] = u[1];
            hydro_fields->u_prev[2][idx] = u[2];
            hydro_fields->u_prev[3][idx] = u[3];

            hydro_fields->pi_b_prev[idx] = 0.0;
            hydro_fields->pi_b_rk0[idx] = 0.0;
            hydro_fields->pi_b_rk1[idx] = 0.0;

            for (int ii = 0; ii < 14; ii++) {
                hydro_fields->Wmunu_prev[ii][idx] = 0.0;
                hydro_fields->Wmunu_rk0[ii][idx] = 0.0;
                hydro_fields->Wmunu_rk1[ii][idx] = 0.0;
            }
        }
    }
    // clean up
    for (int i = 0; i < nx; i++) {
        delete[] temp_profile_ed[i];
        delete[] temp_profile_utau[i];
        delete[] temp_profile_ux[i];
        delete[] temp_profile_uy[i];
    }
    delete[] temp_profile_ed;
    delete[] temp_profile_utau;
    delete[] temp_profile_ux;
    delete[] temp_profile_uy;
}


double Init::eta_profile_normalisation(InitData *DATA, double eta) {
    // this function return the eta envelope profile for energy density
    double res;
    // Hirano's plateau + Gaussian fall-off
    if (DATA->initial_eta_profile == 1) {
        double exparg1, exparg;
        exparg1 = (fabs(eta) - DATA->eta_flat/2.0)/DATA->eta_fall_off;
        exparg = exparg1*exparg1/2.0;
        res = exp(-exparg*theta(exparg1));
    } else if (DATA->initial_eta_profile == 2) {
        // Woods-Saxon
        // The radius is set to be half of DATA->eta_flat
        // The diffusiveness is set to DATA->eta_fall_off
        double ws_R = DATA->eta_flat/2.0;
        double ws_a = DATA->eta_fall_off;
        res = (1.0 + exp(-ws_R/ws_a))/(1.0 + exp((abs(eta) - ws_R)/ws_a));
    } else {
        fprintf(stderr, "initial_eta_profile out of range.\n");
        exit(0);
    }
    return res;
}

double Init::eta_profile_left_factor(InitData *Data, double eta) {
    // this function return the eta envelope for projectile
    double res = eta_profile_normalisation(Data, eta);
    if (fabs(eta) < Data->beam_rapidity) {
        res = (1. - eta/Data->beam_rapidity)*res;
    } else {
        res = 0.0;
    }
    return(res);
}

double Init::eta_profile_right_factor(InitData *Data, double eta) {
    // this function return the eta envelope for target
    double res = eta_profile_normalisation(Data, eta);
    if (fabs(eta) < Data->beam_rapidity) {
        res = (1. + eta/Data->beam_rapidity)*res;
    } else {
        res = 0.0;
    }
    return(res);
}

double Init::eta_rhob_profile_normalisation(InitData *DATA, double eta) {
    // this function return the eta envelope profile for net baryon density
    double res;
    int profile_flag = DATA->initial_eta_rhob_profile;
    double eta_0 = DATA->eta_rhob_0;
    double tau0 = DATA->tau0;
    if (profile_flag == 1) {
        double eta_width = DATA->eta_rhob_width;
        double norm = 1./(2.*sqrt(2*M_PI)*eta_width*tau0);
        double exparg1 = (eta - eta_0)/eta_width;
        double exparg2 = (eta + eta_0)/eta_width;
        res = norm*(exp(-exparg1*exparg1/2.0) + exp(-exparg2*exparg2/2.0));
    } else if (profile_flag == 2) {
        double eta_abs = fabs(eta);
        double delta_eta_1 = DATA->eta_rhob_width_1;
        double delta_eta_2 = DATA->eta_rhob_width_2;
        double A = DATA->eta_rhob_plateau_height;
        double exparg1 = (eta_abs - eta_0)/delta_eta_1;
        double exparg2 = (eta_abs - eta_0)/delta_eta_2;
        double theta;
        double norm = 1./(tau0*(sqrt(2.*M_PI)*delta_eta_1
                          + (1. - A)*sqrt(2.*M_PI)*delta_eta_2 + 2.*A*eta_0));
        if (eta_abs > eta_0)
            theta = 1.0;
        else
            theta = 0.0;
        res = norm*(theta*exp(-exparg1*exparg1/2.)
                    + (1. - theta)*(A + (1. - A)*exp(-exparg2*exparg2/2.)));
    } else {
        fprintf(stderr, "initial_eta_rhob_profile = %d out of range.\n",
                profile_flag);
        exit(0);
    }
    return res;
}

double Init::eta_rhob_left_factor(InitData *DATA, double eta) {
    double eta_0 = -fabs(DATA->eta_rhob_0);
    double tau0 = DATA->tau0;
    double delta_eta_1 = DATA->eta_rhob_width_1;
    double delta_eta_2 = DATA->eta_rhob_width_2;
    double norm = 2./(sqrt(M_PI)*tau0*(delta_eta_1 + delta_eta_2));
    double exp_arg = 0.0;
    if (eta < eta_0) {
        exp_arg = (eta - eta_0)/delta_eta_1;
    } else {
        exp_arg = (eta - eta_0)/delta_eta_2;
    }
    double res = norm*exp(-exp_arg*exp_arg);
    return(res);
}

double Init::eta_rhob_right_factor(InitData *DATA, double eta) {
    double eta_0 = fabs(DATA->eta_rhob_0);
    double tau0 = DATA->tau0;
    double delta_eta_1 = DATA->eta_rhob_width_1;
    double delta_eta_2 = DATA->eta_rhob_width_2;
    double norm = 2./(sqrt(M_PI)*tau0*(delta_eta_1 + delta_eta_2));
    double exp_arg = 0.0;
    if (eta < eta_0) {
        exp_arg = (eta - eta_0)/delta_eta_2;
    } else {
        exp_arg = (eta - eta_0)/delta_eta_1;
    }
    double res = norm*exp(-exp_arg*exp_arg);
    return(res);
}

void Init::output_initial_density_profiles(InitData *DATA, Grid ***arena) {
    // this function outputs the 3d initial energy density profile
    // and net baryon density profile (if turn_on_rhob == 1)
    // for checking purpose
    cout << "output initial density profiles into a file... " << flush;
    ofstream of("check_initial_density_profiles.dat");
    of << "# x(fm)  y(fm)  eta  ed(GeV/fm^3)";
    if (DATA->turn_on_rhob == 1)
        of << "  rhob(1/fm^3)";
    of << endl;
    for (int ieta = 0; ieta < DATA->neta; ieta++) {
        double eta_local = (DATA->delta_eta)*ieta - (DATA->eta_size)/2.0;
        for(int ix = 0; ix < (DATA->nx+1); ix++) {
            double x_local = -DATA->x_size/2. + ix*DATA->delta_x;
            for(int iy = 0; iy < (DATA->ny+1); iy++) {
                double y_local = -DATA->y_size/2. + iy*DATA->delta_y;
                of << scientific << setw(18) << setprecision(8)
                   << x_local << "   " << y_local << "   "
                   << eta_local << "   " << arena[ieta][ix][iy].epsilon*hbarc;
                if (DATA->turn_on_rhob == 1) {
                    of << "   " << arena[ieta][ix][iy].rhob;
                }
                of << endl;
            }
        }
    }
    cout << "done!" << endl;
}
