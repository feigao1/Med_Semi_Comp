//
//  Eigen_Base.h
//  Med_Semi
//
//  Created by Fei Gao on 8/5/19.
//  Copyright © 2019 Fei Gao. All rights reserved.
//

#ifndef Eigen_Base_h
#define Eigen_Base_h

#include "Eigen/Dense"
using namespace Eigen;
using namespace std;


VectorXd VecSort(VectorXd x); // Sort VectorXd increasingly
VectorXd VecSortUniq(VectorXd x); // Sort VectorXd increasingly and with only unique values
VectorXd get_distinct_time(VectorXd L, VectorXd R, VectorXi Rinf); //Get the distinct increasing jump points for interval-censored data
VectorXd get_distinct_time(VectorXd Y, VectorXi Delta); //Get the distinct increasing jump points for right-censored data
VectorXd cumdiff(VectorXd x){ long n=x.size(); for (int i=(int)n-1;i>0;--i) x(i) -= x(i-1); return(x);}
VectorXd cumsum(VectorXd x){ long n=x.size(); for (int i=1;i<n;++i) x(i) += x(i-1); return(x);}
VectorXi gettline(VectorXd t, VectorXd Y); //get line for vector Y in vector t
VectorXi gettline(VectorXd t, VectorXd Y, VectorXi start); //get line for vector Y in vector t, start checking from location given in start

VectorXd VecSort(VectorXd x){
    long p=x.size(); double x1;
    for (int i=1;i<p;++i){
        for(int j=0;j<p-i;++j){
            if (x(j)>=x(j+1)){x1=x(j);x(j)=x(j+1);x(j+1)=x1;}
        }
    }
    return(x);
}
VectorXd VecSortUniq(VectorXd x){
    long p=x.size(); double x1;
    for (int i=1;i<p;++i){
        for(int j=0;j<p-i;++j) {
            if (x(j)>=x(j+1)) {
                x1=x(j); x(j)=x(j+1); x(j+1)=x1;
            }
        }
    }
    VectorXd z(1); z(0)=x(0); int linez=0;
    for(int i=1;i<p;++i){
        if (x(i)>z(linez)){
            VectorXd z1=z; linez=linez+1; z.resize(linez+1);
            for (int i=0;i<linez;++i) z(i)=z1(i);
            z(linez)=x(i);
        }
    }
    return(z);
}
VectorXd get_distinct_time(VectorXd L, VectorXd R, VectorXi Rinf){
    long n1=L.size(); long n2=R.size();
    VectorXd x(n1+n2+1); x.head(n1)=L; int line=(int) n1;
    for (int i=0;i<n2;++i) {
        if (Rinf(i)==0) {
            x(line)=R(i); line++;
        }
    }
    x(line) = 0; line++;//add zero to the jump set
    VectorXd x1=x.head(line); return(VecSortUniq(x1));
}
VectorXd get_distinct_time(VectorXd Y, VectorXi Delta){
    long n=Y.size(); VectorXd x(n+1);int line=0;
    for (int i=0;i<n;++i) {
        if (Delta(i)==1) {
            x(line)=Y(i); line++;
        }
    }
    x(line) = 0; line++;//add zero to the jump set
    VectorXd x1=x.head(line); return(VecSortUniq(x1));
}
VectorXi gettline(VectorXd t, VectorXd Y){
    long m=t.size(); long n = Y.size(); VectorXi Yline(n);
    for (int i=0;i<n;++i){
        int l=0;
        while (l<m){
            if (t(l)<=Y(i)) l++;
            else break;
        }
        Yline(i)=l-1;
    }
    return(Yline);
}
VectorXi gettline(VectorXd t, VectorXd Y, VectorXi start){
    long m=t.size(); long n = Y.size(); VectorXi Yline(n);
    for (int i=0;i<n;++i){
        int l=start(i);
        while (l<m){
            if (t(l)<=Y(i)) l++;
            else break;
        }
        Yline(i)=l-1;
    }
    return(Yline);
}

VectorXd get_form_value(VectorXd t_new, VectorXd t, VectorXd value){
    VectorXi line = gettline(t,t_new);
    VectorXd value_new(t_new.size());
    for (int j=0;j<t_new.size();++j) value_new(j) = value(line(j));
    return(value_new);
}
#endif /* Eigen_Base_h */
