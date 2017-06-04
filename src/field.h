// Copyright 2017 Chun Shen

#ifndef SRC_FIELD_H_
#define SRC_FIELD_H_

struct Field {
     double *e_rk0;
     double *e_rk1;
     double *e_prev;
     double *rhob_rk0;
     double *rhob_rk1;
     double *rhob_prev;
     double **u_rk0;
     double **u_rk1;
     double **u_prev;
     double **dUsup;
     double **Wmunu_rk0;
     double **Wmunu_rk1;
     double **Wmunu_prev;
     double *pi_b_rk0;
     double *pi_b_rk1;
     double *pi_b_prev;
};

#endif  // SRC_FIELD_H_
  
