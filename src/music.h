// Copyright @ Bjoern Schenke, Sangyong Jeon, Charles Gale, and Chun Shen
#ifndef SRC_MUSIC_H_
#define SRC_MUSIC_H_

#include "./util.h"
#include "./grid.h"
#include "./field.h"
#include "./data.h"
#include "./init.h"
#include "./eos.h"
#include "./evolve.h"
#include "./read_in_parameters.h"

//! This is a wrapper class for the MUSIC hydro
class MUSIC {
 private:
    //! records running mode
    int mode;

    //! flag to tell whether hydro is initialized
    int flag_hydro_initialized;
    
    //! flag to tell whether hydro is run
    int flag_hydro_run;

    InitData *DATA;

    Util *util;
    ReadInParameters reader;

    EOS *eos;

    Grid ***arena;

    Field *hydro_fields;

    Init *init;
    Evolve *evolve;

 public:
    MUSIC(InitData *DATA_in, string input_file);
    ~MUSIC();

    //! this function returns the running mode
    int get_running_mode() {return(mode);}

    //! This function initialize hydro
    int initialize_hydro();

    //! this is a shell function to run hydro
    int run_hydro();

};

#endif  // SRC_MUSIC_H_
