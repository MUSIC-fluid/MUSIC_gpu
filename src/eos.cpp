// Copyright 2011 @ Bjoern Schenke, Sangyong Jeon, and Charles Gale
#include <fstream>
#include <iomanip>
#include <string>
#include <cmath>

#include "./eos.h"
#include "./data.h"
using namespace std;

#define cs2 (1.0/3.0)

EOS::EOS(InitData *para_in) {
    parameters_ptr = para_in;
    initialize_eos();
    whichEOS = parameters_ptr->whichEOS;
    eps_max = 1e5;  // [1/fm^4]
}

// destructor
EOS::~EOS() {
}

void EOS::initialize_eos() {
    if (parameters_ptr->Initial_profile == 0) {
        //cout << "Using the ideal gas EOS" << endl;
        init_eos0();
    } //else {
      //  exit(1);
    //}

}

void EOS::init_eos0() {
    whichEOS = 0;
}


double EOS::get_cs2(double e, double rhob) {
    double f;
    if (whichEOS == 0) {
        f = cs2;
    } //else {
      //  fprintf(stderr,"EOS::get_cs2: whichEOS = %d is out of range!\n",
      //          whichEOS);
      //  exit(0);
    //}
    return f;
}

    
//! This function returns the local pressure in [1/fm^4]
//! the input local energy density [1/fm^4], rhob [1/fm^3]
double EOS::get_pressure(double e, double rhob) {
    double f = 0.0;
    if (whichEOS == 0) {
        f = cs2*e;
    } //else {
        //fprintf(stderr, "EOS::get_pressure: whichEOS = %d is out of range!\n",
        //        whichEOS);
        //exit(1);
    //}
    return(f);
}/* get_pressure */


double EOS::p_rho_func(double e, double rhob) {
    // return dP/drho_b (in 1/fm)
    double f;
    if (whichEOS == 0) {
        f = 0.0;
    } //else {
      //  fprintf(stderr, "EOS::p_rho_func: whichEOS = %d is out of range!\n",
      //          whichEOS);
      //  exit(1);
    //}
    return f;
}/* p_rho_func */

double EOS::p_e_func(double e, double rhob) {
    // return dP/de
    double f = 0.0;
    if (whichEOS == 0) {
        f = cs2;
    } //else {
        //fprintf(stderr, "EOS::p_e_func: whichEOS = %d is out of range!\n",
        //        whichEOS);
        //exit(1);
    //}
    return f;
}/* p_e_func */


double EOS::T_from_eps_ideal_gas(double eps) {
    // Define number of colours and of flavours
    double Nc = 3;
    double Nf = 2.5;
    return pow(90.0/M_PI/M_PI*(eps/3.0)/(2*(Nc*Nc-1)+7./2*Nc*Nf), .25);
}

double EOS::s2e_ideal_gas(double s) {
    // Define number of colours and of flavours
    double Nc = 3;
    double Nf = 2.5;

    //e=T*T*T*T*(M_PI*M_PI*3.0*(2*(Nc*Nc-1)+7./2*Nc*Nf)/90.0);
    //s = 4 e / (3 T)
    //s =4/3 T*T*T*(M_PI*M_PI*3.0*(2*(Nc*Nc-1)+7./2*Nc*Nf)/90.0);
    //T = pow(3. * s / 4. / (M_PI*M_PI*3.0*(2*(Nc*Nc-1)+7./2*Nc*Nf)/90.0), 1./3.);
    return 3. / 4. * s * pow(3. * s / 4. / (M_PI*M_PI*3.0*(2*(Nc*Nc-1)+7./2*Nc*Nf)/90.0), 1./3.); //in 1/fm^4

}

//! This function returns entropy density in [1/fm^3]
//! The input local energy density e [1/fm^4], rhob[1/fm^3]
double EOS::get_entropy(double epsilon, double rhob) {
    double f;
    double P, T, mu;
    P = get_pressure(epsilon, rhob);
    T = get_temperature(epsilon, rhob);
    mu = get_mu(epsilon, rhob);
    if (T > 0)
        f = (epsilon + P - mu*rhob)/(T + 1e-15);
    else
        f = 0.;
    if (f < 0.) {
        f = 1e-16;
        // cout << "[Warning]EOS::get_entropy: s < 0!" << endl;
        // cout << "s = " << f << ", e = " << epsilon << ", P = " << P
        //      << ", mu = " << mu << ", rhob = " << rhob << ", T = " << T
        //      << endl;
    }
    return f;
}/* get_entropy */



//! This function returns the local temperature in [1/fm]
//! input local energy density eps [1/fm^4] and rhob [1/fm^3]
double EOS::get_temperature(double eps, double rhob) {
    double T = 0.0;
    if (whichEOS == 0) {
        T = T_from_eps_ideal_gas(eps);
    } //else {
      //  fprintf(stderr,"EOS::get_temperature: "
      //          "whichEOS = %d is out of range!\n", whichEOS);
      //  exit(0);
    //}
    return max(T, 1e-15);
}


//! This function returns the local baryon chemical potential  mu_B in [1/fm]
//! input local energy density eps [1/fm^4] and rhob [1/fm^3]
double EOS::get_mu(double eps, double rhob) {
    double mu = 0.0;
    if (whichEOS == 0) {
        mu = 0.0;
    } //else {
      //  fprintf(stderr, "EOS::get_mu: whichEOS = %d is out of range!\n",
      //          whichEOS);
      //  exit(0);
    //}
    return mu;
}


double EOS::get_muS(double eps, double rhob) {
    // return mu_B in [1/fm]
    double mu;
    if (whichEOS == 0) {
        mu = 0.0;
    } //else {
      //  fprintf(stderr, "EOS::get_mu: whichEOS = %d is out of range!\n",
      //          whichEOS);
      //  exit(0);
    //}
    return mu;
}

double EOS::get_s2e(double s, double rhob) {
    // s - entropy density in 1/fm^3
    double e = 0.0;  // epsilon - energy density
    if (whichEOS == 0) {
        e = s2e_ideal_gas(s);
    } //else {
      //  fprintf(stderr, "get_s2e:: whichEOS = %d out of range.\n", whichEOS);
      //  exit(0);
    //}
    return e;  // in 1/fm^4
}


