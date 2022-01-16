//
//  Semi_Comp_Simu.cpp
//  Med_Semi
//
//  Created by Fei Gao on 8/8/19.
//  Copyright © 2019 Fei Gao. All rights reserved.
//
//input --n or --nsub sample size
//      --nrep #replicates for this file
//      --out output name for file
//      --seed seed
// Optional:
//      --hn hn for profile-likelihood

#include <iostream>
#include <fstream>
#include <unistd.h>
#include <string>
#include <stdlib.h>
#include "Semi_Comp.h"

string int2string(int x)
{   char temp[64];
    string str;
    sprintf(temp, "%d", x);
    string xs(temp);
    return(xs);
}
bool ParseCommandLineArgs(const int argc, char* argv[],int* nsub, double* hn, int* nrep, int* nfile, int* seed) {
    // Initialize with bad parameter, so we can check at end value was updated.
    *seed = -1;
    *nsub = -1;
    *nrep = -1;
    string inputstring;
    // Loop through command line arguments, starting at 1 (first arg is the command
    // to run the program).
    for (int i = 1; i < argc; ++i) {
        string arg = string(argv[i]);
        if (arg == "--seed") {
            if (i == argc - 1) {
                cout << "\nERROR Reading Command: Expected argument after '--seed'.\n"
               << "Aborting.\n";
                return false;
            }
            ++i; inputstring=string(argv[i]); *seed = atoi(inputstring.c_str());
        } else if (arg == "--nsub" || arg == "--n") {
            if (i == argc - 1) {
                cout << "\nERROR Reading Command: Expected argument after '--n'.\n"
               << "Aborting.\n";
                return false;
            }
            ++i; inputstring=string(argv[i]); *nsub = atoi(inputstring.c_str());
        } else if (arg == "--nrep") {
            if (i == argc - 1) {
                cout << "\nERROR Reading Command: Expected argument after '--nrep'.\n"
               << "Aborting.\n";
                return false;
            }
            ++i; inputstring=string(argv[i]); *nrep = atoi(inputstring.c_str());
        } else if (arg == "--hn") {
            if (i == argc - 1) {
                cout << "\nERROR Reading Command: Expected argument after '--hn'.\n"
               << "Aborting.\n";
                return false;
            }
            ++i; inputstring=string(argv[i]); *hn = atoi(inputstring.c_str());
        } else if (arg == "--nfile") {
            if (i == argc - 1) {
                cout << "\nERROR Reading Command: Expected argument after '--nfile'.\n";
                return false;
            }
            ++i; inputstring=string(argv[i]); *nfile = atoi(inputstring.c_str());
        }  else {
            cout << "\nERROR Reading Command: unrecognized argument: " << arg << endl;
            return false;
        }
    }
    
    return *nsub != -1 && *nrep != -1 && *seed != -1;
}


int main(int argc, char* argv[])
{   // Parse simulation parameters from command line.
    int seed, nsub, nrep, nfile; double hn=1.0;
    if (!ParseCommandLineArgs(argc, argv, &nsub, &hn, &nrep, &nfile, &seed)) { return -1;}
    string nf = to_string(nfile);
    string ns = to_string(nsub);
    string title = "n_" + ns;
    string out_file = title + "theta_" + nf +  ".dat";
    string out_fileM = title + "LambdaM_" + nf + ".dat";
    string out_fileR = title + "LambdaR_" + nf + ".dat";
    string out_fileT = title + "LambdaT_" + nf + ".dat";
    string out_file_NDE1 = title + "Med_NDE1_" + nf + ".dat";
    string out_file_NIE1 = title + "Med_NIE1_" + nf + ".dat";
    string out_file_TE2 = title + "Med_TE2_" + nf + ".dat";
    string out_file_TE3 = title + "Med_TE3_" + nf + ".dat";
    
    
    int nt = 1000; VectorXd t (nt);
    for (int l=0;l<nt;++l) t(l) = (double) l / (double) nt * 15;
    int nt_med = 5; VectorXd t_med (nt_med); t_med << 2.0, 4.0, 6.0, 8.0, 10.0;
    int px = 2; VectorXd x(px); x << 0.5,0.5;
    int ptheta = (6+2)*(px+1);
    
    default_random_engine generator(seed);
    VectorXd beta(6); beta << 0.5, 0.5, -0.2, 0.4, 0, 0.2;
    MatrixXd gamma(px,6); gamma << 0.5, -0.2, 0.4, 0.5, -0.5, -0.2, 0.5, -0.2, 0.5, 0.5, -0.2, 0;
    MatrixXd alpha((px+1),2); alpha << 0, 0.2, 0.3, -0.5, 0.1, 0.3;
    
    ofstream myfile (out_file);
    ofstream myfileT (out_fileT); ofstream myfileR (out_fileR); ofstream myfileM (out_fileM);
    ofstream myfile_NDE1 (out_file_NDE1); ofstream myfile_NIE1 (out_file_NIE1);
    ofstream myfile_TE2 (out_file_TE2); ofstream myfile_TE3 (out_file_TE3);
    if (myfile.is_open()){
        for (int k=0;k<nrep;++k){
            Semi_Comp data = simudata(nsub, beta, gamma, alpha, generator);
            data.solve_semi_comp();
            for (int k=0;k<6;++k) myfile << data.theta_[k].transpose() << " ";
            for (int k=0;k<2;++k) myfile << data.alpha_[k].transpose() << " ";
            myfile << data.iter << " ";
            VectorXd SE = data.semi_comp_boot_med(100,generator,x,t_med);
            myfile << SE.head(ptheta).transpose() << " " << data.boot_trynum << endl;
            myfileM << data.t_[0](data.m_(0)-1) << " " << get_form_value(t, data.t_[0], cumsum(data.lambda_[0])).transpose() << endl;
            myfileR << data.t_[1](data.m_(1)-1) << " " << get_form_value(t, data.t_[1], cumsum(data.lambda_[1])).transpose() << endl;
            myfileT << data.t_[2](data.m_(2)-1) << " " << get_form_value(t, data.t_[2], cumsum(data.lambda_[2])).transpose() << endl;
            
            vector<VectorXd> Med = data.est_med_t(x,t_med);
            myfile_NDE1 << data.t_[0](data.m_(0)-1) << " ";
            myfile_NDE1 << Med[0].transpose() << " " << SE.segment(ptheta,nt_med).transpose() << endl;
            myfile_NIE1 << data.t_[0](data.m_(0)-1) << " ";
            myfile_NIE1 << Med[1].transpose() << " " << SE.segment(ptheta+nt_med,nt_med).transpose() << endl;
            myfile_TE2 << max(data.t_[0](data.m_(0)-1),data.t_[2](data.m_(2)-1)) << " ";
            myfile_TE2 << Med[2].transpose() << " " << SE.segment(ptheta+2*nt_med,nt_med).transpose() << endl;
            myfile_TE3 << data.t_[2](data.m_(2)-1) << " ";
            myfile_TE3 << Med[3].transpose() << " " << SE.segment(ptheta+3*nt_med,nt_med).transpose() << endl;
        }
        myfile.close(); myfileT.close(); myfileR.close(); myfileM.close();
        myfile_NDE1.close();  myfile_NIE1.close(); myfile_TE2.close(); myfile_TE3.close();
        cout << "File written!";
    }
    else cout << "Unable to open file";
    return(0);
}
