//
//  Semi_Comp.h
//  Med_Semi
//
//  Created by Fei Gao on 8/5/19.
//  Copyright © 2019 Fei Gao. All rights reserved.
//

#ifndef Semi_Comp_h
#define Semi_Comp_h

#include "Eigen_Base.h"
#include <random>

class Semi_Comp{
private:
    vector<VectorXd> theta0_, alpha0_, lambda0_;
    
    MatrixXd z_, w_;
    VectorXi y1line_, y21line_, y2line_;
    MatrixXd PU_; VectorXd logLik_i;
    const double epsilon=1E-6;
    const int maxiter=10000;
    
    //Lambda -> lambda
    VectorXd Lambdatrans(VectorXd x){
        long n=x.size();
        for (int i=(int)n-1;i>0;--i) x(i) -= x(i-1);
        return(x);
    }
    //lambda -> Lambda
    VectorXd lambdatrans(VectorXd x){
        long n=x.size();
        for (int i=1;i<n;++i) x(i) += x(i-1);
        return(x);
    }
    //Calculate conditional expectations and likelihood
    void Estep(vector<VectorXd> theta, vector<VectorXd> alpha, vector<VectorXd> lambda) {
        logLik_i.setZero(); PU_.setZero();
        double Ai1, Ai2, Bi2, Bi3, Ci1, Ci2, Ci3;
        vector<VectorXd> Lambda; Lambda.resize(K_);
        double thetaXM1, thetaXM2, thetaXR1, thetaXR2, thetaXT2, thetaXT3, ea1, ea2;
        for (int k=0;k<K_;++k) Lambda[k] = lambdatrans(lambda[k]);
        for (int i=0;i<n_;++i){
            thetaXM1 = exp(z_.row(i) * theta[0]); thetaXR1 = exp(z_.row(i) * theta[1]);
            thetaXM2 = exp(w_.row(i) * theta[2]); thetaXR2 = exp(w_.row(i) * theta[3]);
            thetaXT2 = exp(w_.row(i) * theta[4]); thetaXT3 = exp(z_.row(i) * theta[5]);
            ea1 = exp(w_.row(i)*alpha[0]); ea2 = exp(w_.row(i)*alpha[1]);
            logLik_i(i) = -log(1+ea1+ea2);
            if (delta1_(i)==1){
                Ai1 = ea1 * lambda[0](y1line_(i)) * thetaXM1 * exp(-thetaXM1 * Lambda[0](y1line_(i)) - thetaXR1 * Lambda[1](y21line_(i)));
                if (delta2_(i)==1) Ai1 *= lambda[1](y21line_(i)) * thetaXR1;
                if (a_(i)==1) {PU_(i,0) = 1; logLik_i(i) += log(Ai1);} else{
                    Ai2 = ea2 * lambda[0](y1line_(i)) * thetaXM2 * exp(-thetaXM2 * Lambda[0](y1line_(i)) - thetaXR2 * Lambda[1](y21line_(i)));
                    if (delta2_(i)==1) Ai2 *= lambda[1](y21line_(i)) * thetaXR2;
                    PU_(i,0) = Ai1 / (Ai1 + Ai2); PU_(i,1) = Ai2 / (Ai1 + Ai2);
                    logLik_i(i) += PU_(i,0) * log(Ai1) + PU_(i,1) * log(Ai2);
                }
            } else {
                if (delta2_(i)==1){
                    Bi3 = lambda[2](y2line_(i)) * thetaXT3 * exp(-thetaXT3 * Lambda[2](y2line_(i)));
                    if (a_(i)==0) {PU_(i,2) = 1; logLik_i(i) += log(Bi3);} else{
                        Bi2 = ea2 * lambda[2](y2line_(i)) * thetaXT2 * exp(-thetaXT2 * Lambda[2](y2line_(i)));
                        PU_(i,1) = Bi2 / (Bi2 + Bi3); PU_(i,2) = Bi3 / (Bi2 + Bi3);
                        logLik_i(i) += PU_(i,1) * log(Bi2) + PU_(i,2) * log(Bi3);
                    }
                } else{
                    Ci1 = ea1 * exp(-thetaXM1 * Lambda[0](y1line_(i)));
                    if (a_(i)==0) Ci2 = ea2 * exp(-thetaXM2 * Lambda[0](y1line_(i)));
                    else Ci2 = ea2 * exp(-thetaXT2 * Lambda[2](y2line_(i)));
                    Ci3 = exp(-thetaXT3 * Lambda[2](y2line_(i)));
                    PU_(i,0) = Ci1 / (Ci1 + Ci2 + Ci3); PU_(i,1) = Ci2 / (Ci1 + Ci2 + Ci3); PU_(i,2) = Ci3 / (Ci1 + Ci2 + Ci3);
                    logLik_i(i) += PU_(i,0) * log(Ci1) + PU_(i,1) * log(Ci2) + PU_(i,2) * log(Ci3);
                }
            }
        }
    }
    
    void Mstepalpha(vector<VectorXd>& alpha){
        VectorXd Score((p_+1)*2);
        MatrixXd Hessian((p_+1)*2,(p_+1)*2);
        double ea1, ea2;
        Score.setZero(); Hessian.setZero();
        for (int i=0;i<n_;++i){
            ea1 = exp(w_.row(i)*alpha[0]); ea2 = exp(w_.row(i)*alpha[1]);
            Score.head(p_+1) += (PU_(i,0) - ea1 / (1 + ea1 + ea2)) * w_.row(i);
            Score.tail(p_+1) += (PU_(i,1) - ea2 / (1 + ea1 + ea2)) * w_.row(i);
            Hessian.block(0,0,(p_+1),(p_+1)) += ea1*(1+ea2)/pow(1+ea1+ea2,2) * w_.row(i).transpose() * w_.row(i);
            Hessian.block(0,(p_+1),(p_+1),(p_+1)) -= ea1*ea2/pow(1+ea1+ea2,2) * w_.row(i).transpose() * w_.row(i);
            Hessian.block((p_+1),(p_+1),(p_+1),(p_+1)) += ea2*(1+ea1)/pow(1+ea1+ea2,2) * w_.row(i).transpose() * w_.row(i);
        }
        Hessian.block((p_+1),0,(p_+1),(p_+1)) = Hessian.block(0,(p_+1),(p_+1),(p_+1)).transpose();
        VectorXd step = Hessian.inverse() * Score;
        alpha[0] += step.head((p_+1)); alpha[1] += step.tail((p_+1));
    }
    
    void Msteptheta(vector<VectorXd>& theta){
        VectorXd Score(2*(p_+1)); MatrixXd Hessian(2*(p_+1),2*(p_+1)); VectorXd step;
        double s0; VectorXd s1((p_+1)*2); MatrixXd  s2((p_+1)*2,(p_+1)*2);
        double thetaX1, thetaX2, thetaX3;
        double ind;
        // thetaM1, thetaM2
        Score.setZero(); Hessian.setZero();
        for (int i=0;i<n_;++i){
            if (delta1_(i)==1){
                Score.head(p_+1) += PU_(i,0) * z_.row(i);
                Score.tail(p_+1) += PU_(i,1) * (1-(double)a_(i)) * w_.row(i);
                s0 = 0; s1.setZero(); s2.setZero();
                for (int j=0;j<n_;++j){
                    if (y1line_(j)>=y1line_(i)){
                        ind = 1 - (double)delta2_(j) * (1-(double)delta1_(j));
                        thetaX1 = exp(z_.row(j) * theta[0]); thetaX2 = exp(w_.row(j) * theta[2]);
                        s0 += ind * (PU_(j,0) * thetaX1 + PU_(j,1) * (1-(double)a_(j)) * thetaX2);
                        s1.head(p_+1) += ind * PU_(j,0) * thetaX1 * z_.row(j);
                        s1.tail(p_+1) += ind * PU_(j,1) * (1-(double)a_(j)) * thetaX2 * w_.row(j);
                        s2.block(0,0,p_+1,p_+1) += ind * PU_(j,0) * thetaX1 * z_.row(j).transpose() * z_.row(j);
                        s2.block(p_+1,p_+1,p_+1,p_+1) += ind * PU_(j,1) * (1-(double)a_(j)) * thetaX2 * w_.row(j).transpose() * w_.row(j);
                    }
                }
                Score -= s1 / s0;
                Hessian += s2/s0 - s1*s1.transpose()/(pow(s0,2));
            }
        }
        step = Hessian.inverse() * Score;
        theta[0] = theta[0] + step.head(p_+1); theta[2] = theta[2] + step.tail(p_+1);
        
        // thetaR1, thetaR2
        Score.setZero(); Hessian.setZero();
        for (int i=0;i<n_;++i){
            if ((delta1_(i)==1)&(delta2_(i)==1)){
                Score.head(p_+1) += PU_(i,0) * z_.row(i);
                Score.tail(p_+1) += PU_(i,1) * (1-(double)a_(i)) * w_.row(i);
                s0 = 0; s1.setZero(); s2.setZero();
                for (int j=0;j<n_;++j){
                    if (y21line_(j)>=y21line_(i)){
                        ind = (double)delta1_(j);
                        thetaX1 = exp(z_.row(j) * theta[1]); thetaX2 = exp(w_.row(j) * theta[3]);
                        s0 += ind * (PU_(j,0) * thetaX1 + PU_(j,1) * (1-(double)a_(j)) * thetaX2);
                        s1.head(p_+1) += ind * PU_(j,0) * thetaX1 * z_.row(j);
                        s1.tail(p_+1) += ind * PU_(j,1) * (1-(double)a_(j)) * thetaX2 * w_.row(j);
                        s2.block(0,0,p_+1,p_+1) += ind * PU_(j,0) * thetaX1 * z_.row(j).transpose() * z_.row(j);
                        s2.block(p_+1,p_+1,p_+1,p_+1) += ind * PU_(j,1) * (1-(double)a_(j)) * thetaX2 * w_.row(j).transpose() * w_.row(j);
                    }
                }
                Score -= s1 / s0;
                Hessian += s2/s0 - s1*s1.transpose()/(pow(s0,2));
            }
        }
        step = Hessian.inverse() * Score;
        theta[1] = theta[1] + step.head(p_+1); theta[3] = theta[3] + step.tail(p_+1);
        
        // thetaT2, thetaT3
        Score.setZero(); Hessian.setZero();
        for (int i=0;i<n_;++i){
            if ((delta1_(i)==0)&(delta2_(i)==1)){
                Score.head(p_+1) += PU_(i,1) * (double)a_(i) * w_.row(i);
                Score.tail(p_+1) += PU_(i,2) * z_.row(i);
                s0 = 0; s1.setZero(); s2.setZero();
                for (int j=0;j<n_;++j){
                    if (y2line_(j)>=y2line_(i)){
                        ind = 1-(double)delta1_(j);
                        thetaX2 = exp(w_.row(j) * theta[4]); thetaX3 = exp(z_.row(j) * theta[5]);
                        s0 += ind * (PU_(j,1) * (double)a_(j) * thetaX2 + PU_(j,2) * thetaX3);
                        s1.head(p_+1) += ind * PU_(j,1) * (double)a_(j) * thetaX2 * w_.row(j);
                        s1.tail(p_+1) += ind * PU_(j,2) * thetaX3 * z_.row(j);
                        s2.block(0,0,p_+1,p_+1) += ind * PU_(j,1) * (double)a_(j) * thetaX2 * w_.row(j).transpose() * w_.row(j);
                        s2.block(p_+1,p_+1,p_+1,p_+1) += ind * PU_(j,2) * thetaX3 * z_.row(j).transpose() * z_.row(j);
                    }
                }
                Score -= s1 / s0;
                Hessian += s2/s0 - s1*s1.transpose()/(pow(s0,2));
            }
        }
        step = Hessian.inverse() * Score;
        theta[4] = theta[4] + step.head(p_+1); theta[5] = theta[5] + step.tail(p_+1);
    }
    vector<VectorXd> Msteplambda(vector<VectorXd> theta) {
        vector<VectorXd> lambda(K_); for (int k=0;k<K_;++k) lambda[k].resize(m_(k));
        double thetaX1, thetaX2, thetaX3;
        double ind, numer, s0;
        // lambda1
        s0 = 0;
        for (int l=(m_(0)-1);l>=0;--l){
            numer = 0;
            for (int i=0;i<n_;++i){
                if (y1line_(i)==l){
                    ind = 1 - (double)delta2_(i) * (1-(double)delta1_(i));
                    thetaX1 = exp(z_.row(i) * theta[0]); thetaX2 = exp(w_.row(i) * theta[2]);
                    s0 += ind * (PU_(i,0) * thetaX1 + PU_(i,1) * (1-(double)a_(i)) * thetaX2);
                    if (delta1_(i)==1) numer += 1;
                }
            }
            lambda[0](l)=numer/s0;
        }
        // lambda2
        s0 = 0;
        for (int l=(m_(1)-1);l>=0;--l){
            numer = 0;
            for (int i=0;i<n_;++i){
                if ((y21line_(i)==l)&(delta1_(i)==1)){
                    thetaX1 = exp(z_.row(i) * theta[1]); thetaX2 = exp(w_.row(i) * theta[3]);
                    s0 += PU_(i,0) * thetaX1 + PU_(i,1) * (1-(double)a_(i)) * thetaX2;
                    if (delta2_(i)==1) numer += 1;
                }
            }
            lambda[1](l)=numer/s0;
        }
        // lambda3
        s0 = 0;
        for (int l=(m_(2)-1);l>=0;--l){
            numer = 0;
            for (int i=0;i<n_;++i){
                if ((y2line_(i)==l)&(delta1_(i)==0)){
                    thetaX2 = exp(w_.row(i) * theta[4]); thetaX3 = exp(z_.row(i) * theta[5]);
                    s0 += PU_(i,1) * (double)a_(i) * thetaX2 + PU_(i,2) * thetaX3;
                    if (delta2_(i)==1) numer += 1;
                }
            }
            lambda[2](l)=numer/s0;
        }
        return(lambda);
    }
    
    double diff_theta(){
        double x=0;
        for (int k=0;k<Kb_;++k) x += (theta_[k] - theta0_[k]).norm();
        for (int k=0;k<2;++k) x += (alpha_[k]-alpha0_[k]).norm();
        return(x);
    }
    double diff_lambda(vector<VectorXd> lambda,vector<VectorXd> lambda0){
        double x=0;
        for (int k=0;k<K_;++k) x += (lambda[k] - lambda0[k]).norm();
        return(x);
    }
    
public:
    VectorXd y1_, y2_; VectorXi delta1_, delta2_; MatrixXd x_; VectorXi a_;
    long n_, p_; long K_ = 3; long Kb_ = 6;
    Vector3i m_; vector<VectorXd> t_; vector<VectorXd> lambda_;
    vector<VectorXd> theta_; // thetaM1, thetaR1, thetaM2, thetaR2, thetaT2, thetaT3
    vector<VectorXd> alpha_; //alpha1, alpha2
    
    VectorXd logLik_MLE_i; int iter; int boot_trynum;
    MatrixXd pI_score, pI_infor; VectorXd sd_score, sd_infor; VectorXd pscore;
    
    VectorXd NDE1, NIE1, TE2, TE3; VectorXd t_NDE1, t_NIE1, t_TE2, t_TE3;
    
    Semi_Comp() = default;
    Semi_Comp(VectorXd y1, VectorXd y2, VectorXi delta1, VectorXi delta2, MatrixXd x, VectorXi a){
        y1_ = y1; y2_ = y2; delta1_ = delta1; delta2_ = delta2; x_ = x; a_ = a;
        n_ = x.rows(); p_ = x.cols();
        theta_.resize(Kb_); theta0_.resize(Kb_);
        for (int k=0;k<Kb_;++k) { theta_[k].resize(p_+1); theta_[k].setZero();}
        alpha_.resize(2); alpha0_.resize(2);
        for (int k=0;k<2;++k) { alpha_[k].resize(p_+1); alpha_[k].setZero();}
        
        t_.resize(K_); lambda_.resize(K_); lambda0_.resize(K_);
        t_[0] = get_distinct_time(y1_,delta1_);
        VectorXi delta12(n_); for (int i=0;i<n_;++i) delta12(i) = delta1_(i) * delta2_(i);
        t_[1] = get_distinct_time(y2_-y1_,delta12);
        t_[2] = get_distinct_time(y2_,delta2_-delta12);
        for (int k=0;k<K_;++k){
            m_(k) = (int) t_[k].size();
            VectorXd Lambda = t_[k]/t_[k].maxCoeff();
            lambda_[k] = Lambdatrans(Lambda);
        }
        y1line_ = gettline(t_[0], y1_);
        y21line_ = gettline(t_[1], y2_-y1_);
        y2line_ = gettline(t_[2], y2_);
        
        z_.resize(n_,p_+1); for (int i=0;i<n_;++i) z_(i,0)= (double)a_(i); z_.block(0,1,n_,p_) = x_;
        w_.resize(n_,p_+1); w_.block(0,0,n_,1).setOnes(); w_.block(0,1,n_,p_) = x_;
        logLik_i.resize(n_); PU_.resize(n_,K_);
    }
    
    void solve_semi_comp(){
        iter=0; double diff = 0;
        while (iter<=maxiter){
            for (int k=0;k<Kb_;++k) theta0_[k]=theta_[k];
            for (int k=0;k<K_;++k) lambda0_[k]=lambda_[k];
            for (int k=0;k<2;++k) alpha0_[k]=alpha_[k];
            
            Estep(theta_,alpha_,lambda_);
            Msteptheta(theta_); Mstepalpha(alpha_);
            lambda_ = Msteplambda(theta_);
            diff = diff_theta() + diff_lambda(lambda_,lambda0_);
            iter++; if (diff<epsilon) break;
            cout<<"iter= "<<iter<<endl;
            for (int k=0;k<Kb_;++k) cout<<" theta"<<k<<" = "<<theta_[k].transpose()<<endl;
            for (int k=0;k<2;++k) cout<<" alpha"<<k<<" = "<<alpha_[k].transpose()<<endl;
            for (int k=0;k<K_;++k) cout<<" lambda"<<k<<" = "<<lambda_[k].head(5).transpose()<<endl;
            cout<<", diff="<<diff<<", logLik="<<logLik_i.sum()/(double)(n_)<<endl;
            
            if (std::isnan(theta_[0](0))==1){ iter=maxiter+1; }
        }
        logLik_MLE_i=logLik_i;
        if (iter<=maxiter){
            cout<<"The Algorithm converges in "<<iter<<" iterations."<<endl;
            for (int k=0;k<Kb_;++k) cout<<"The estimate for theta"<<k<<" is: "<<theta_[k].transpose()<<endl;
            for (int k=0;k<2;++k) cout<<"The estimate for alpha"<<k<<" is: "<<alpha_[k].transpose()<<endl;
        }
        else {cout<<"Maximum Iteration Times ("<<maxiter<<") Reached!"<<endl;}
    }
    
    VectorXd semi_comp_boot(long nrep, default_random_engine & generator){
        uniform_real_distribution<double> bootdist(0,(double)n_);
        VectorXd y1(n_); VectorXd y2(n_); VectorXi delta1(n_); VectorXi delta2(n_); MatrixXd x(n_,p_); VectorXi a(n_);
        long p = (Kb_+2)*(p_+1);
        MatrixXd varres(nrep,p);
        int rep=0; boot_trynum=0;
        while(rep<nrep){
            for (int i=0;i<n_;++i){
                double subid = bootdist(generator);
                y1(i) = y1_(subid);
                y2(i) = y2_(subid);
                delta1(i) = delta1_(subid);
                delta2(i) = delta2_(subid);
                a(i) = a_(subid);
                x.row(i) = x_.row(subid);
            }
            Semi_Comp data_boot(y1, y2, delta1, delta2, x, a);
            data_boot.solve_semi_comp(); boot_trynum++;
            if (data_boot.iter<=maxiter){
                int pp=0;
                for (int k=0;k<Kb_;++k) {for (int j=0;j<(p_+1);++j) {varres(rep,pp) = data_boot.theta_[k](j); ++pp;}}
                for (int k=0;k<2;++k) { for (int j=0;j<(p_+1);++j) {varres(rep,pp) = data_boot.alpha_[k](j); ++pp;}}
                ++rep;
            }
        }
        VectorXd SE(p);SE.setZero();
        for (int j=0;j<p;++j){
            double barX=(varres.col(j)).mean();
            for (int i=0;i<nrep;++i){SE(j)=SE(j)+pow(varres(i,j)-barX,2.0)/(double(nrep)-1.0);}
            SE(j)=pow(SE(j),0.5);
        }
        return(SE);
    }

    void est_med(VectorXd x){
        double thetaWM1, betaXM1, thetaWM2, thetaWR1, betaXR1,thetaWR2, thetaWT2, thetaWT3, betaXT3;
        VectorXd w(p_+1); w(0) = 1; w.tail(p_)=x;
        vector<VectorXd> Lambda(K_); for (int k=0;k<K_;++k) Lambda[k] = lambdatrans(lambda_[k]);
        thetaWM1 = exp(w.transpose() * theta_[0]); betaXM1 = exp(x.transpose() * theta_[0].tail(p_));
        thetaWR1 = exp(w.transpose() * theta_[1]); betaXR1 = exp(x.transpose() * theta_[1].tail(p_));
        thetaWM2 = exp(w.transpose() * theta_[2]);
        thetaWR2 = exp(w.transpose() * theta_[3]);
        thetaWT2 = exp(w.transpose() * theta_[4]);
        thetaWT3 = exp(w.transpose() * theta_[5]); betaXT3 = exp(x.transpose() * theta_[5].tail(p_));
        double inc; int t2line;
        // U=1
        t_NDE1 = t_[0]; NDE1.resize(m_(0)); NDE1.setZero();
        t_NIE1 = t_[0]; NIE1.resize(m_(0)); NIE1.setZero();
        for (int t=0;t<m_(0);++t){
            for (int j=0;j<=t;++j){
                VectorXd tt1j(1); tt1j(0) = t_[0](t)-t_[0](j);
                t2line = gettline(t_[1],tt1j)(0);
                inc = lambda_[0](j)*betaXM1 * exp(-Lambda[0](j) * betaXM1);
                inc *= exp(-Lambda[1](t2line) * thetaWR1) - exp(-Lambda[1](t2line) * betaXR1);
                NDE1(t) += inc;
                inc = exp(-Lambda[1](t2line) * thetaWR1) * lambda_[0](j);
                inc *= thetaWM1 * exp(-Lambda[0](j) * thetaWM1) - betaXM1 * exp(-Lambda[0](j) * betaXM1);
                NIE1(t) += inc;
            }
            inc = exp(-Lambda[0](t) * thetaWM1) - exp(-Lambda[0](t) * betaXM1);
            NIE1(t) += inc;
        }
        // U=2
        VectorXd t13(m_(0)+m_(2)); t13.head(m_(0)) = t_[0]; t13.tail(m_(2)) = t_[2];
        t13 = VecSortUniq (t13); long m13 = t13.size();
        VectorXi t1line = gettline(t_[0], t13); VectorXi t3line = gettline(t_[2],t13);
        t_TE2 = t13; TE2.resize(m13); TE2.setZero();
        for (int t=0;t<m13;++t){
            TE2(t) = exp(-Lambda[2](t3line(t)) * thetaWT2) - 1;
            for (int j=0;j<=t1line(t);++j){
                VectorXd tt1j(1); tt1j(0) = t13(t)-t_[0](j);
                t2line = gettline(t_[1],tt1j)(0);
                inc = lambda_[0](j)*thetaWM2 * exp(-Lambda[0](j)*thetaWM2) * (1 - exp(-Lambda[1](t2line)*thetaWR2));
                TE2(t) += inc;
            }
        }
        //U=3
        t_TE3 = t_[2]; TE3.resize(m_(2));
        for (int t=0;t<m_(2);++t) TE3(t) = exp(-Lambda[2](t) * thetaWT3) - exp(-Lambda[2](t) * betaXT3);
    }
    
    vector<VectorXd> est_med_t(VectorXd x, VectorXd t){
        double thetaWM1, betaXM1, thetaWM2, thetaWR1, betaXR1,thetaWR2, thetaWT2, thetaWT3, betaXT3;
        VectorXd w(p_+1); w(0) = 1; w.tail(p_)=x;
        vector<VectorXd> Lambda(K_); for (int k=0;k<K_;++k) Lambda[k] = lambdatrans(lambda_[k]);
        thetaWM1 = exp(w.transpose() * theta_[0]); betaXM1 = exp(x.transpose() * theta_[0].tail(p_));
        thetaWR1 = exp(w.transpose() * theta_[1]); betaXR1 = exp(x.transpose() * theta_[1].tail(p_));
        thetaWM2 = exp(w.transpose() * theta_[2]);
        thetaWR2 = exp(w.transpose() * theta_[3]);
        thetaWT2 = exp(w.transpose() * theta_[4]);
        thetaWT3 = exp(w.transpose() * theta_[5]); betaXT3 = exp(x.transpose() * theta_[5].tail(p_));
        double inc; int t2line;
        long nt = t.size(); vector<VectorXd> Med(4); for (int k=0;k<4;++k){ Med[k].resize(nt); Med[k].setZero();}
        
        VectorXi t1line = gettline(t_[0],t);
        VectorXi t3line = gettline(t_[2],t);
        for (int j=0;j<nt;++j){
            for (int jj=0;jj<=t1line(j);++jj){
                VectorXd tt1j(1); tt1j(0) = t(j)-t_[0](jj);
                t2line = gettline(t_[1],tt1j)(0);
                // U=1
                inc = lambda_[0](jj)*betaXM1 * exp(-Lambda[0](jj) * betaXM1);
                inc *= exp(-Lambda[1](t2line) * thetaWR1) - exp(-Lambda[1](t2line) * betaXR1);
                Med[0](j) += inc;
                inc = exp(-Lambda[1](t2line) * thetaWR1) * lambda_[0](jj);
                inc *= thetaWM1 * exp(-Lambda[0](jj) * thetaWM1) - betaXM1 * exp(-Lambda[0](jj) * betaXM1);
                Med[1](j) += inc;
                // U=2
                inc = lambda_[0](jj)*thetaWM2 * exp(-Lambda[0](jj)*thetaWM2) * (1 - exp(-Lambda[1](t2line)*thetaWR2));
                Med[2](j) += inc;
            }
            // U=1
            Med[1](j) += exp(-Lambda[0](t1line(j)) * thetaWM1) - exp(-Lambda[0](t1line(j)) * betaXM1);
            // U=2
            Med[2](j) += exp(-Lambda[2](t3line(j)) * thetaWT2) - 1;
            //U=3
            Med[3](j) = exp(-Lambda[2](t3line(j)) * thetaWT3) - exp(-Lambda[2](t3line(j)) * betaXT3);
        }
        return(Med);
    }
    
    VectorXd semi_comp_boot_med(long nrep, default_random_engine & generator, VectorXd xx, VectorXd t){
        uniform_real_distribution<double> bootdist(0,(double)n_);
        VectorXd y1(n_); VectorXd y2(n_); VectorXi delta1(n_); VectorXi delta2(n_); MatrixXd x(n_,p_); VectorXi a(n_);
        long nt = t.size(); long p = (Kb_+2)*(p_+1) + 4*nt;
        MatrixXd varres(nrep,p);
        int rep=0; boot_trynum=0;
        while(rep<nrep){
            for (int i=0;i<n_;++i){
                double subid = bootdist(generator);
                y1(i) = y1_(subid);
                y2(i) = y2_(subid);
                delta1(i) = delta1_(subid);
                delta2(i) = delta2_(subid);
                a(i) = a_(subid);
                x.row(i) = x_.row(subid);
            }
            Semi_Comp data_boot(y1, y2, delta1, delta2, x, a);
            data_boot.solve_semi_comp(); boot_trynum++;
            if (data_boot.iter<=maxiter){
                vector<VectorXd> med_boot = data_boot.est_med_t(xx,t);
                int pp=0;
                for (int k=0;k<Kb_;++k) {for (int j=0;j<(p_+1);++j) {varres(rep,pp) = data_boot.theta_[k](j); ++pp;}}
                for (int k=0;k<2;++k) { for (int j=0;j<(p_+1);++j) {varres(rep,pp) = data_boot.alpha_[k](j); ++pp;}}
                for (int k=0;k<4;++k) {for (int j=0;j<nt;++j) {varres(rep,pp) = med_boot[k](j); ++pp;}}
                ++rep;
            }
        }
        VectorXd SE(p);SE.setZero();
        for (int j=0;j<p;++j){
            double barX=(varres.col(j)).mean();
            for (int i=0;i<nrep;++i){SE(j)=SE(j)+pow(varres(i,j)-barX,2.0)/(double(nrep)-1.0);}
            SE(j)=pow(SE(j),0.5);
        }
        return(SE);
    }
};

//Simulation: Lambda1 = t, Lambda2 = 0.2t, Lambda3 = log(1+t)
Semi_Comp simudata(long nsub, VectorXd beta, MatrixXd gamma, MatrixXd alpha, default_random_engine & generator){
    //Distributions
    normal_distribution<double> X1dist(0.0,1.0);
    uniform_real_distribution<double> X2dist(0.0,1.0);
    bernoulli_distribution Adist(0.5);
    
    uniform_real_distribution<double> udist(0.0,1.0);
    uniform_real_distribution<double> Cdist(0.0,15.0);
    
    MatrixXd X(nsub,2); VectorXi A(nsub);
    MatrixXd Y(nsub,2); MatrixXi Delta(nsub,2);
    VectorXi U(nsub);
    
    double u1, u2, u3, expa1, expa2, expM, expR, expT, M , R, T, C;
    int nU1, nU2A0, nU2A1, nU3; nU1=0; nU2A0=0; nU2A1=0; nU3=0;
    int n010, n111, n101; n010=0; n111=0; n101=0;
    VectorXd W(3); W(0) = 1;
    for (int i=0;i<nsub;++i){
        X(i,0) = X1dist(generator); X(i,1) = X2dist(generator); A(i) = Adist(generator);
        W.tail(2) = X.row(i);
        expa1 = exp(W.transpose()*alpha.col(0)); expa2 = exp(W.transpose()*alpha.col(1));
        u1 = udist(generator); u2 = udist(generator); u3 = udist(generator); C = Cdist(generator);
        if (u1<expa1/(1+expa1+expa2)){ U(i) = 1; // U = 1
            expM = exp((double)A(i) * beta(0) + X.row(i)*gamma.col(0));
            expR = exp((double)A(i) * beta(1) + X.row(i)*gamma.col(1));
            M = - log(u2) / expM; R = -log(u3) / expR / 0.2;
            if (M<=C){ Delta(i,0) = 1; Y(i,0) = M;} else { Delta(i,0) = 0; Y(i,0) = C;}
            if (M+R<=C){ Delta(i,1) = 1; Y(i,1) = M+R;} else { Delta(i,1) = 0; Y(i,1) = C;}
            if ((Delta(i,0)==1)&(Delta(i,1)==1)&(A(i)==1)) n111++;
            if ((Delta(i,0)==1)&(Delta(i,1)==0)&(A(i)==1)) n101++;
            nU1 ++;
        } else if (u1<(expa1 + expa2)/(1+expa1+expa2)){ U(i) = 2;// U = 2
            if (A(i)==0){
                expM = exp(beta(2) + X.row(i)*gamma.col(2));
                expR = exp(beta(3) + X.row(i)*gamma.col(3));
                M = - log(u2) / expM; R = -log(u3) / expR / 0.2;
                if (M<=C){ Delta(i,0) = 1; Y(i,0) = M;} else { Delta(i,0) = 0; Y(i,0) = C;}
                if (M+R<=C){ Delta(i,1) = 1; Y(i,1) = M+R;} else { Delta(i,1) = 0; Y(i,1) = C;}
                nU2A0 ++;
            } else {
                expT = exp(beta(4) + X.row(i)*gamma.col(4));
                T = exp(-log(u2) / expT) - 1; Delta(i,0) = 0;
                if (T<=C){ Delta(i,1) = 1; Y(i,0) = T; Y(i,1) = T;} else { Delta(i,1) = 0; Y(i,0) = C; Y(i,1) = C;}
                nU2A1 ++;
            }
        } else{ U(i) = 3; // U = 3
            expT = exp((double)A(i) * beta(5) + X.row(i)*gamma.col(5));
            T = exp(-log(u2) / expT) - 1; Delta(i,0) = 0;
            if (T<=C){ Delta(i,1) = 1; Y(i,0) = T; Y(i,1) = T;} else { Delta(i,1) = 0; Y(i,0) = C; Y(i,1) = C;}
            if ((Delta(i,0)==0)&(Delta(i,1)==1)&(A(i)==0)) n010++;
            nU3 ++;
        }
    }
    cout<<"The censoring rate for event 1 is "<<1-(double)Delta.col(0).sum()/(double)nsub<<endl;
    cout<<"The censoring rate for event 2 is "<<1-(double)Delta.col(1).sum()/(double)nsub<<endl;
    cout<<"The proportion of subjects are "<<(double)nU1/(double)nsub<<" "<<(double)nU2A0/(double)nsub<<" "<<(double)nU2A1/(double)nsub<<" "<<(double)nU3/(double)nsub<<" "<<endl;
    cout<<"The proportion of identified subjects are "<<(double)n111/(double)nsub<<" "<<(double)n101/(double)nsub<<" "<<(double)n010/(double)nsub<<" "<<endl;
    Semi_Comp data(Y.col(0), Y.col(1), Delta.col(0), Delta.col(1), X, A);
    return(data);
}

#endif /* Semi_Comp_h */
