#include "util.h"
#include <iostream>
#include <string>
#include <execinfo.h>

using namespace std;

double ***Util::cube_malloc(int n1, int n2, int n3)
{
  //  cout << "test1" << endl;
    int i,j,k;
//     int inc;
    double ***d1_ptr; 
//     double *tmp_ptr;
    n1+=1;
    n2+=1;
    n3+=1;


    //    tmp_ptr = (double *) malloc(sizeof(double)*n1*n2*n3);
    //    tmp_ptr = new double[n1*n2*n3];

    // cout << "test2" << endl;

    /* pointer to the n1*n2*n3 memory */

    //    d1_ptr = (double ***) malloc (sizeof(double **)*n1);
    d1_ptr = new double **[n1];
    // cout << "test3" << endl;

    for(i=0; i<n1; i++) 
     {
       //      d1_ptr[i] = (double **) malloc (sizeof(double *)*n2);
       d1_ptr[i] = new double *[n2];
     } 
    //  cout << "test4" << endl;

    for(i=0; i<n1; i++)
    {
     for(j=0; j<n2; j++) 
      {
       // inc = n2*n3*i + n3*j;
       // d1_ptr[i][j] = &(tmp_ptr[inc]);
       d1_ptr[i][j] = new double [n3];
      }
    }
    //  cout << "test5" << endl;

    for(i=0; i<n1; i++)
    {
     for(j=0; j<n2; j++) 
      {
	for(k=0; k<n3; k++) 
	  {
	    d1_ptr[i][j][k] = 0.0;
	  }
      }
    }
    // cout << "test6" << endl;

    return d1_ptr;
}/* cube_malloc */


void Util::cube_free(double ***cube, int n1, int n2, int n3)
{
  int i,j;
  n1+=1;
  n2+=1;
  n3+=1;
  

  for(i=0; i<n1; i++)
    {
      for(j=0; j<n2; j++) 
	{
	  // inc = n2*n3*i + n3*j;
	  // d1_ptr[i][j] = &(tmp_ptr[inc]);
	  delete [] cube[i][j];
	}
    }
  for(j=0; j<n1; j++) 
    {
      delete [] cube[j];
    }
  
  delete [] cube;

}/* cube_free */


double **Util::mtx_malloc(int n1, int n2)
{
    int i, j;
    double **d1_ptr; 
//     double *tmp_ptr;

    //    tmp_ptr = (double *)malloc(sizeof(double)*n1*n2);
    // tmp_ptr = new double[n1*n2];

    //    d1_ptr = (double **) malloc (sizeof(double *)*n1);
    d1_ptr = new double *[n1];
    
    for(i=0; i<n1; i++) 
     {
       //      d1_ptr[i] = &(tmp_ptr[i*n2]);
       d1_ptr[i] = new double [n2];
     }
    
    for(i=0; i<n1; i++) 
     {
      for(j=0; j<n2; j++) 
	{
	  d1_ptr[i][j] = 0.0;
	}
     }

return d1_ptr;
}


void Util::mtx_free(double **m, int n1, int n2)
{
  int j;
  
 for(j=0; j<n1; j++) 
   {
     delete [] m[j];
   }

 delete [] m;

}


double *Util::vector_malloc(int n1)
{
 double *d1_ptr;
 int i;

    /* pointer to the n1 array */
 //    d1_ptr = (double *) malloc (sizeof(double )*n1);
 d1_ptr = new double[n1];
 for(i=0; i<n1; i++) d1_ptr[i] = 0.0; 
    
 return d1_ptr;
}


void Util::vector_free(double *vec)
{
  delete [] vec;
}

char *Util::char_malloc(int n1)
{
    char *char_ptr;

    /* pointer to the n1 array */
    char_ptr = (char *) malloc (sizeof(char)*n1);
    //char_ptr = new char[n1];

    strcpy(char_ptr, "");
return char_ptr;
}


void Util::char_free(char *vec)
{
  free(vec);
  //delete vec;
}

double Util::Power(double x, int n)
{
 int i;
 double y;

 if(n == 0) return 1.0;

 y = 1.0;
 for(i=1; i<=n; i++) y *= x;
 
 return y;
}

int Util::IsFile(string file_name)
{
//  static int isf;
//  static int ind = 0;
//  char st[80];
 FILE *temp;

 if( (temp = fopen(file_name.c_str(),"r")) == NULL) return 0;
 else 
  {
   fclose(temp);
   return 1;
  }
}/* IsFile */


string Util::StringFind(string file_name, const char *st)
{
  string inputname = file_name;
  string tmpfilename;
  stringstream sinput;
  stringstream strinput;
  string str;
  strinput << st;
  strinput >> str;

  string s;
  string xstr;
 
  tmpfilename = "input";

  int ind;
  static int flag = 0;
  
  if(flag == 0)
    {
      if(!IsFile(file_name))
	{
	  cerr << "The file named " << file_name << " is absent." << endl;
	  if(file_name == "") 
	    {
	      fprintf(stderr, "No input file name specified.\n");
	      fprintf(stderr, "Creating a default file named input...\n");
	    }
	  else 
	    {
	      cout << "Creating " << tmpfilename << "..." << endl;
	    }
	  ofstream tmp_file(tmpfilename.c_str());
	  tmp_file << "EndOfData" << endl;
	  tmp_file.close();
	}/* if isfile */
//       flag == 1;
    }/* if flag == 0 */
  
    ifstream input(inputname.c_str());
    
    input >> s;

    ind = 0;
    while(s.compare("EndOfData") != 0)
      {
	input >> xstr;
	if(s.compare(str) == 0)
	  {
	    ind++;
	    input.close();
	    return xstr;
	  }/* if right, return */
	input >> s;
      }/* while */

    input.close();
    
    if(ind == 0)
      {
	cerr << str << " not found in " << inputname << endl; 
	cout << "Create an input file." << endl;
	exit(1);
	return xstr;
      }  
  cout << "Error in Util::StringFind\n";
 return "empty";
}/* StringFind */

// support comments in the parameters file
// comments need to start with #
string Util::StringFind4(string file_name, string str_in)
{
  string inputname = file_name;
  string str = str_in;

  string tmpfilename;
  tmpfilename = "input.default";
  
  // check whether the input parameter file is exist or not
  if(!IsFile(file_name))
  {
    if(file_name == "") 
    {
      fprintf(stderr, "No input file name specified.\n");
      fprintf(stderr, "Creating a default file named input.default\n");
    }
    else 
    {
      cerr << "The file named " << file_name << " is absent." << endl;
      cout << "Creating " << file_name << "..." << endl;
      tmpfilename = file_name;
    }
    ofstream tmp_file(tmpfilename.c_str());
    tmp_file << "EndOfData" << endl;
    tmp_file.close();
    exit(1);
  }/* if isfile */
  
  // pass checking, now read in the parameter file
  string temp_string;
  ifstream input(inputname.c_str());
  getline(input, temp_string);  // read in the first entry

  int ind = 0;
  string para_name;
  string para_val;
  while(temp_string.compare("EndOfData") != 0)  // check whether it is the end of the file
  {
    string para_string;
    stringstream temp_ss(temp_string);
    getline(temp_ss, para_string, '#');  // remove the comments
    if(para_string.compare("") != 0 && para_string.find_first_not_of(' ') != std::string::npos)  // check the read in string is not empty
    {
      stringstream para_stream(para_string);
      para_stream >> para_name >> para_val;
      if(para_name.compare(str) == 0)  // find the desired parameter
      {
        ind++;
        input.close();
        return para_val;
      }/* if right, return */
    }
    getline(input, temp_string);  // read in the next entry
  }/* while */

  input.close(); // finish read in and close the file
    
  if(ind == 0)  // the desired parameter is not in the parameter file, then return "empty"
     return "empty";

  // should not cross here !!!
  cout << "Error in Util::StringFind4 !!!\n";
  return "empty";
}/* StringFind4 */


double Util::DFind(string file_name, const char *st)
{
  string s;
  double x;
  stringstream stm;
  
  s = StringFind(file_name, st);
  stm << s;
  stm >> x;
  return x;
}/* DFind */

double Util::lin_int(double x1,double x2,double f1,double f2,double x)
{
  double aa, bb;
  
  if (x2 == x1) 
    aa = 0.0;
  else
    aa =(f2-f1)/(x2-x1);
  bb = f1 - aa * x1;
  
  return aa*x + bb;
}

