#ifndef randutils_hpp
#define randutils_hpp

#include "eigen_vec.hpp"
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_on_sphere.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <ctime>
#include <cmath>
#include <random>

#define PI acos(-1)

// Defining clip template for double variable
template<typename T>
T clip(T value, T min_val, T max_val) {
    if (value < min_val) {
        return min_val;
    } else if (value > max_val) {
        return max_val;
    } else {
        return value;
    }
}

class hrand {
  private:
    boost::mt19937 gen;

  public:
    hrand(const int& seed) {
        if (seed > 0)
            gen.seed(seed);
        else {
            // std::time_t now = std::time(0);
            // gen.seed(now);
            std::random_device rd;
            boost::mt19937 Gen(rd());
            boost::random::uniform_int_distribution<int> dist(1, 2147483647);
            int now = dist(Gen);
            gen.seed(now);
        }
    }

    int uniform_int(const int& a, const int& b);
    double uniform_real(const double& a, const double& b);
    double normal_distributed(const double& mu, const double& sigma);
    //double inv_power_law_distributed(const double& a, const double& b, const double& alpha);
    double inv_power_law_distributed(const double& a, const double& b, const double& alpha);
    Vector4d uniform_hypersphere(const int &dim);
};

namespace sam {} // namespace sam

#endif // randutils_hpp
