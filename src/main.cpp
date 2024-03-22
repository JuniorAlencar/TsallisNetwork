#include "SamuraI.hpp"
#include "SamuraIConfig.hpp"
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <boost/graph/random.hpp>
#include <random>

#include <chrono>
#include <iomanip>  // for setprecision

// comentarion
using json = nlohmann::json;

namespace fs = boost::filesystem;

using namespace std;

samargs read_parametes(const string& fname) {
    std::ifstream f(fname.c_str());
    json data = json::parse(f);

    samargs xargs;
    xargs.num_vertices = data.value("num_vertices", 128);
    xargs.alpha_a = data.value("alpha_a", 1.0);
    xargs.alpha_g = data.value("alpha_g", 1.0);
    xargs.r_min = data.value("r_min", 1.0);
    xargs.r_max = data.value("r_max", 1e7);
    xargs.dim = data.value("dim", 3);
    xargs.seed = data.value("seed", 1234);
    
    if (xargs.seed < 0) {
        // std::time_t now = std::time(0);
        // xargs.seed = now % 10000;
        std::random_device rd;
        boost::mt19937 Gen(rd());
        boost::random::uniform_int_distribution<int> dist(1, 2147483647);
        int now = dist(Gen);
        xargs.seed = now;
    }
    return xargs;
}

int main(int argc, char* argv[]) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " file.json" << endl;
        return 1;
    }

    // read parameters from json file
    samargs xargs = read_parametes(argv[1]);

    // set filenames
    char n_folder[50], alpha_folder[80], gml_folder[100], prop_folder[100], prop_file[250], gml_file[250], time_process_file[250];
    sprintf(n_folder, "./N_%d", xargs.num_vertices);    
    sprintf(alpha_folder, "%s/dim_%d/alpha_a_%1.1f_alpha_g_%1.1f", n_folder, xargs.dim, xargs.alpha_a, xargs.alpha_g);
    sprintf(gml_folder, "%s/gml", alpha_folder);
    sprintf(prop_folder, "%s/prop", alpha_folder);
    sprintf(prop_file, "%s/prop_%d.csv", prop_folder, xargs.seed);
    sprintf(gml_file, "%s/gml_%d.gml.gz", gml_folder, xargs.seed);
    sprintf(time_process_file, "%s/time_process_seconds.txt", alpha_folder);
    
    // create dir
    fs::create_directories(alpha_folder);
    fs::create_directories(gml_folder);
    fs::create_directories(prop_folder);

    SamuraI S(xargs);
    S.createGraph();
    double l = S.computeGlobalNavigation().shortestpath;
    int d = S.computeGlobalNavigation().diamater;
    
    double r = S.computeAssortativityCoefficient();
	//S.writeConnections(connections_file);
    //S.writeDegrees(degree_file);
    S.writeGML(gml_file);
    cout << prop_file << endl;
    ofstream pout(prop_file);
    pout << "#mean shortest path," << "# diamater," << "#assortativity coefficient\r\n";
    pout << l << "," << d << "," << r << endl;
    pout.close();

    S.clear();
    
    cout << time_process_file << endl;
    // Gen file to calculate time to run process
    // Stop measuring time
    auto end_time = std::chrono::high_resolution_clock::now();

    // Calculate the duration in seconds with 5 decimal places
    auto duration_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();

    // Save the duration to a file
    std::ofstream file(time_process_file, std::ios::app);  // Open file in append mode
    if (file.is_open()) {
        // Set precision to 5 decimal places
        
        file << std::fixed << std::setprecision(5) << duration_seconds << std::endl;
        file.close();
        std::cout << "Execution time saved to file." << std::endl;
    } else {
        std::cerr << "Error opening file for writing." << std::endl;
    }
    // -----------------
    return 0;
}
