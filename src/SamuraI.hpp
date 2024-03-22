#ifndef samurai_hpp
#define samurai_hpp

#include "eigen_vec.hpp"
#include "randUtils.hpp"

#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/properties.hpp>
#include <boost/pending/property.hpp>
#include <boost/property_map/property_map.hpp>

#include <cassert>
#include <iostream>
#include <vector>

using namespace boost;

typedef Matrix<double, 4, 1> Vector4d;

struct samargs {
    int num_vertices;
    double r_min;
    double r_max;
    int dim;
    double alpha_a;
    double alpha_g;
    int seed;
};

struct Navigation{
    int diamater;
    double shortestpath;
};

typedef adjacency_list<vecS, vecS, undirectedS, property<vertex_degree_t, int>>
    graph_t;

typedef graph_traits<graph_t>::edge_descriptor edge_t;
typedef graph_traits<graph_t>::vertex_descriptor vertex_t;
typedef std::pair<vertex_t, int> vertex_prop_int;
typedef std::pair<vertex_t, double> vertex_prop_double;

typedef graph_traits<graph_t>::vertex_iterator vertex_it;
typedef graph_traits<graph_t>::edge_iterator edge_it;

class SamuraI {
  private:
    graph_t G; // the graph ...

    int m_num_vertices;
    int m_num_edges;

    double m_alpha_a; // attachment exponent
    double m_alpha_g; // grouth exponent
    double m_r_min;   // minimum distance for placing particles
    double m_r_max;   // maximum distance for placing particles
    int m_dim;        // bin/ system dimension (1-4)

    std::vector<Vector4d> pos;  // vector with positions
    Vector4d center_of_mass;    // vector with center of mass
    Vector4d sum_positions;     // auxiliar vector to center of mass
    //std::vector<float> r;       // vector with r gen for power law

    void CreateNetwork();
    int m_seed;

  public:
    SamuraI(const samargs& xargs)
        : m_num_vertices(xargs.num_vertices), m_alpha_a(xargs.alpha_a), m_alpha_g(xargs.alpha_g),
          m_r_min(xargs.r_min), m_r_max(xargs.r_max), m_dim(xargs.dim), m_seed(xargs.seed) {}

    inline void createGraph() {
        assert(m_num_vertices > 0);
        center_of_mass.setZero();
        sum_positions.setZero();
        CreateNetwork();
    }
    void writeDegrees(std::string fname);
    void writeConnections(std::string fname);
	Navigation computeGlobalNavigation(void);
    void writeGML(std::string fname);
    double computeAssortativityCoefficient(void);
    //void writeR(std::string fname);
    
    void clear();
};

#endif // !samurai_hpp
