#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sstream>
#include <fstream>
#include <cstring>
#include <sys/stat.h>

using namespace std;

#ifndef PI
#define PI (3.14159265358979324)
#endif

#ifndef hbarc
#define hbarc (0.1973)
#endif

#ifndef default_tol
#define default_tol (1.0e-8)
#endif

#define absol(a) ((a) > 0 ? (a) : (-(a)))
#define maxi(a, b) ((a) > (b) ? (a) : (b))
#define mini(a, b) ((a) < (b) ? (a) : (b))
#define sgn(x) ((x) < 0.0 ? (-1.0) : (1.0))
#define theta(x) ((x) < 0.0 ? (0.0) : (1.0))
#define SQR(a) ((a)*(a))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define DMAX(a,b) ((a) > (b) ? (a) : (b))
#define DMIN(a,b) ((a) < (b) ? (a) : (b))
#define SIGN(a,b) ((b) >= 0.0 ? absol(a): -absol(a)) 
/* Sangyong Nov 18 2014 */
/* added gmn = Minkowski metric to be used in sums */
#define gmn(a) ((a) == 0 ? (-1.0) : (1.0))

#define BT_BUF_SIZE 500

class Util{
 public:
double ***cube_malloc(int , int , int ); 
double **mtx_malloc(int , int );
double *vector_malloc(int );
char *char_malloc(int );
void char_free(char *);

void cube_free(double ***, int, int, int);
void mtx_free(double **, int, int);
void vector_free(double *);

double Power(double, int);

int IsFile(string );

double DFind(string file_name, const char *st);
string StringFind(string file_name, const char *st);
string StringFind4(string file_name, string str_in);

double lin_int(double x1,double x2,double f1,double f2,double x);

};

#endif
