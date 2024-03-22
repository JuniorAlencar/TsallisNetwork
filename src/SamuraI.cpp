#include "SamuraI.hpp"
#include "eigen_vec.hpp"
#include "randUtils.hpp"

#include <boost/graph/graph_concepts.hpp>
#include <boost/graph/named_function_params.hpp>
#include <boost/graph/properties.hpp>
#include <boost/range/iterator_range_core.hpp>
#include <boost/graph/breadth_first_search.hpp> //BFS
#include <boost/graph/visitors.hpp>
#include <boost/array.hpp>
#include<zlib.h>
#include <limits>

inline bool cmp(vertex_prop_double p1, vertex_prop_double p2) { return p1.second < p2.second; }

void SamuraI::CreateNetwork(){
    hrand rnd(m_seed);
    Vector4d p(0.0, 0.0, 0.0, 0.0);
    boost::add_edge(0, 1, G);
    boost::add_vertex(G);
    
    pos.push_back(p);
    
    for(int i=1; i < m_num_vertices; i++){
        double radius;
        Vector4d dp;
        // generate a inverse power law distance
        radius = rnd.inv_power_law_distributed(m_r_min, m_r_max, m_alpha_g);
        // generate dX point...
        dp = rnd.uniform_hypersphere(m_dim);
        
        p = center_of_mass + radius * dp;
        sum_positions += p;
        center_of_mass = sum_positions / (i + 1);
        boost::add_vertex(G);
        pos.push_back(p);
        
        if(i>1){        
        
        vertex_t New = i;
        std::vector<vertex_prop_double> prob;
        double exponent = -0.5 * m_alpha_a;
        double p_total = 0.0;
        
        for (int u=0;u<i;u++){
                vertex_t v = u;
                int k_v = boost::degree(v, G);
                Vector4d Ruv = pos[v] - p;
                //std::cout << "dist:" << Ruv << std::endl;
                double Ruv_SQ = Ruv.transpose() * Ruv;
                //std::cout << "Ruv_SQ:" << Ruv_SQ << std::endl;
                double Pv = k_v * pow(Ruv_SQ, exponent);
                //std::cout << Pv << std::endl;
                p_total += Pv;
                prob.push_back(std::make_pair(v, Pv));
                //std::cout << "Pv:" << Pv << "," << "v:" << v << std::endl;
                }
            
        // normalize, cumpute cumsum and pick
        double r = rnd.uniform_real(0.0, 1.0);
        double cumsum = 0;
        for (size_t i = 0; i < prob.size(); ++i) {
            prob[i].second /= p_total; // normalizar
            cumsum += prob[i].second;
            //std::cout << "cumu:" << cumsum << std::endl;
            if (cumsum > r){
                boost::add_edge(prob[i].first, New, G);
                break;
                }
            }
        }
    }
}

Navigation SamuraI::computeGlobalNavigation(){
	struct Navigation BFS;

    double meanShortestPath = 0;
    //int count = 0;
    std::vector<int> d(m_num_vertices,0);  // vector for diamater
    int aux = 0;                           // auxiliary value for the diameter
    
    for(auto u : boost::make_iterator_range(vertices(G))){
        //boost::array<int, 100000> distances{{0}};
        std::vector<int> distances(m_num_vertices,0);  // vector for diamater
        breadth_first_search(G, u, visitor(make_bfs_visitor(record_distances(&distances[0], on_tree_edge()))));
        // breadth_first_search(G, u, visitor(
        //      make_bfs_visitor( record_distances(distances.begin(), 
        //      on_tree_edge{}))));
    
        for (auto e=distances.begin(); e != distances.end(); ++e){
             meanShortestPath += *e;
             //++ count;
             if(*e > d[aux])
                d[aux] = *e;
      }
    }
    int dia = *max_element(d.begin(), d.end());
    double count = num_vertices(G)*(num_vertices(G)-1);
    meanShortestPath /= count;
    BFS.shortestpath = meanShortestPath;
    BFS.diamater = dia;
    
    return BFS;
}

double SamuraI::computeAssortativityCoefficient() {
    int NE = num_vertices(G)-1;   // Total number of degree network
    double T1 = 0.0, T2=T1,T3=T1; // Sum terms
    double R;                     // Assortativity coefficient

    for (auto e : boost::make_iterator_range(boost::edges(G))){
        int u = boost::degree(source(e, G),G);  // degree source
        int v = boost::degree(target(e, G),G);  // degree target
        T1 += u*v;                              // first sum;
        T2 += 0.5*(u+v);                        // second sum;
        T3 += 0.5*(pow(u,2)+pow(v,2));          // third sum;
        }
    T1 /= NE;
    T2 /= NE;
    T3 /= NE;

    R = (T1 - pow(T2,2))/(T3 - pow(T2,2));

    return R;
}

// Navigation SamuraI::computeLocalNavigation(){
//     struct Navigation Local;

//     int NV = num_vertices(G);

//     std::random_device RD;      // Random device for catch random vertex
//     boost::mt19937 gen(RD());

//     vertex_t s = boost::random_vertex(G, gen); // Source random
//     vertex_t t = boost::random_vertex(G, gen); // Target random
    
//     int count = 0;
//     double q = 1.0;
    
//     //std::vector<int> d(num_vertices(G),0);
//     for(int i=0;i<NV;i++){
//         auto neighbours = boost::adjacent_vertices(s, G);
//         std::vector<double> d_nt(boost::degree(s,G), 0.0);
//         // distances between neibor and target
        
//         for (auto vd : make_iterator_range(neighbours)){
//             // Position neighborhood
//             Vector4d p = pos[vd];
//             // Distance between target and neighborhood
//             Vector4d Ruv = pos[t] - p;
//             double Ruv_SQ = Ruv.transpose()*Ruv;
//             d_nt[i] = Ruv_SQ;
//             }
//         // min value in distances
//         auto min_value = std::min_element(std::begin(d_nt), std::end(d_nt));
//         // catch the index of min value and redefine 's' for that node
//         vertex_t k = std::distance(d_nt.begin(), min_value);
//         s = k;
//         }
    
//     Local.shortestpath = q;
//     Local.diamater = count;
//     return Local;
// }

void SamuraI::writeDegrees(std::string fname){
    std::cout << fname << std::endl;

    gzFile fi = gzopen(fname.c_str(),"wb");
    gzprintf(fi,"k,\r\n");
    for(int i=0;i<m_num_vertices;i++){
        vertex_t v = i;
        gzprintf(fi,"%d,\r\n",boost::degree(v,G));
        }
    gzclose(fi);
}

void SamuraI::writeConnections(std::string fname){
    std::cout << fname << std::endl;

    gzFile fi = gzopen(fname.c_str(),"wb");
    gzprintf(fi,"#Node1,");
    gzprintf(fi,"#Node2,\r\n");
    for (auto e : boost::make_iterator_range(boost::edges(G))) {
        gzprintf(fi,"%d,",boost::source(e, G));
        gzprintf(fi,"%d\r\n",boost::target(e, G));
    }
    gzclose(fi);
}

void SamuraI::writeGML(std::string fname){
    std::cout << fname << std::endl;
    
    gzFile fi = gzopen(fname.c_str(),"wb");

    gzprintf(fi, "graph\n");
    gzprintf(fi, "[\n");
    gzprintf(fi, "  Creator \"Gephi\"\n");
    gzprintf(fi, "  undirected 0\n");
    
//    for (auto v : boost::make_iterator_range(boost::vertices(G))) {
      for (int i=0;i<m_num_vertices;i++) {
	vertex_t v = i;
        gzprintf(fi, "node\n");
        gzprintf(fi, "[\n");
        gzprintf(fi, "id (%d)\n", i);
        gzprintf(fi, "label (%d)\n",i);
        gzprintf(fi, "graphics\n");
        gzprintf(fi, "[\n");
        gzprintf(fi, "x %f\n", pos[v](0));
        gzprintf(fi, "y %f\n", pos[v](1));
        gzprintf(fi, "z %f\n", pos[v](2));
        gzprintf(fi, "q %f\n", pos[v](3));
        gzprintf(fi, "degree %d\n", boost::degree(v,G));
        gzprintf(fi,"]\n");
        gzprintf(fi,"]\n");
    }
    
     for (auto e : boost::make_iterator_range(boost::edges(G))){
         Vector4d Ruv = pos[boost::source(e, G)] - pos[boost::target(e, G)];
         gzprintf(fi,"edge\n");
         gzprintf(fi,"[\n");
         gzprintf(fi, "source (%d)\n", boost::source(e, G));
         gzprintf(fi, "target (%d)\n", boost::target(e, G));
         gzprintf(fi, "distance %f\n", pow(Ruv.transpose() * Ruv, 0.5));
         gzprintf(fi, "]\n");
     }
     gzprintf(fi,"]");
     gzclose(fi);
 }

void SamuraI::clear() {
    G.clear();
    pos.clear();
};
