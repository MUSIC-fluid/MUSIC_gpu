#MUSIC               {#mainpage}
===================

MUSIC requires GSL libraries to be installed from
[here](https://www.gnu.org/software/gsl/)
as well as a C++ compiler.  
It only runs properly on POSIX operating systems 
(tested on LINUX and Mac OS X).

## Compile the code

* The MUSIC code can be compiled using standard cmake. 
* Alternatively, you can compile the MUSIC code using a makefile. In this way,
  please make sure the information about the MPI compiler and the directory of
  the GSL library is correct in the src/GNUmakefile. Then one can compile
  the code by typing `make`.

## Run MUSIC with an input file

An input file is required that contains the line "EndOfData", 
preceded by a list of parameter names and values, one per line,
with parameter names and values separated by a space or tab.
If omitted, each parameter will be assigned a default value.  

To run MUSIC, compiled under the name "mpihydro", with an input file "input_Gubser", use
./mpihydro input_Gubser
