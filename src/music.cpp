// Original copyright 2011 @ Bjoern Schenke, Sangyong Jeon, and Charles Gale
// Massively cleaned up and improved by Chun Shen 2015-2016
#include <stdio.h>
#include <sys/stat.h>

#include <cstring>
#include "./music.h"
#include "./dissipative.h"

using namespace std;

MUSIC::MUSIC(InitData *DATA_in, string input_file) {
    DATA = DATA_in;
    reader.read_in_parameters(DATA, input_file);
    mode = DATA->mode;
    eos = new EOS(DATA);
    util = new Util();
    hydro_fields = new Field;
    flag_hydro_run = 0;
    flag_hydro_initialized = 0;
}


MUSIC::~MUSIC() {
    if (flag_hydro_initialized == 1) {
        delete init;
    }
    if (flag_hydro_run == 1) {
        delete evolve;
    }
    delete eos;
    delete util;
    delete hydro_fields;
}

int MUSIC::initialize_hydro() {
    // clean all the surface files
    system("rm surface.dat surface?.dat surface??.dat 2> /dev/null");

    init = new Init(eos, DATA);
    init->InitArena(DATA, hydro_fields);
    flag_hydro_initialized = 1;
    return(0);
}


int MUSIC::run_hydro() {
    // this is a shell function to run hydro
    evolve = new Evolve(eos, DATA);
    evolve->EvolveIt(DATA, hydro_fields);
    flag_hydro_run = 1;
    return(0);
}
