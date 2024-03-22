#include "randUtils.hpp"

double hrand::uniform_real(const double& a, const double& b) {
    boost::random::uniform_real_distribution<double> dist(a, b);
    return dist(gen);
}

int hrand::uniform_int(const int &a, const int &b){
    boost::random::uniform_int_distribution<int> dist(a,b);
    return dist(gen);
}

double hrand::normal_distributed(const double& mu, const double& sigma) {
    boost::random::normal_distribution<double> dist(mu, sigma);
    return dist(gen);
}


double hrand::inv_power_law_distributed(const double &a, const double &b, const double &alpha) {

    double u = uniform_real(0.0, 1.0);
    if(alpha == 0){
        //double x = pow(b, u);
        return 0;
    }
    double x = b/pow((u*(1-pow(b,alpha))+pow(b,alpha)),1/alpha);    
    return clip(x, a, b);    
}

Vector4d hrand::uniform_hypersphere(const int &dim){
    
    boost::random::uniform_on_sphere<double> dist(dim);
    std::vector<double> v = dist(gen);
    Vector4d p;
    p.setZero();
    for(long unsigned int i =0; i < v.size(); ++i)
        p(i) = v[i];
    return p;
}

