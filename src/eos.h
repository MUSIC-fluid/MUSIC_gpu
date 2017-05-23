// Copyright 2011 @ Bjoern Schenke, Sangyong Jeon, and Charles Gale
#ifndef SRC_EOS_H_
#define SRC_EOS_H_

#include <iostream>

#include "./util.h"
#include "./data.h"

class EOS {
 private:
    InitData *parameters_ptr;

    int whichEOS;
    double eps_max;

    Util *util;

 public:
    EOS(InitData *para_in);  // constructor
    ~EOS();  // destructor
    void initialize_eos();
    void init_eos0();                // for whichEOS=0

    // returns maximum local energy density of the EoS table
    // in the unit of [1/fm^4]
    double get_eps_max() {return(eps_max);}

    double get_cs2(double e, double rhob);
    double get_rhob_from_mub(double e, double mub);
    double p_rho_func(double e, double rhob);
    double p_e_func(double e, double rhob);
    double T_from_eps_ideal_gas(double eps);
    double get_entropy(double epsilon, double rhob);
    double get_temperature(double epsilon, double rhob);
    double get_mu(double epsilon, double rhob);
    double get_muS(double epsilon, double rhob);
    double get_pressure(double epsilon, double rhob);
    double ssolve(double e, double rhob, double s);
    double Tsolve(double e, double rhob, double T);
    double findRoot(double (EOS::*function)(double, double, double),
                    double rhob, double s, double e1, double e2, double eacc);
    double s2e_ideal_gas(double s);
    double get_s2e(double s, double rhob);
};

#endif  // SRC_EOS_H_
