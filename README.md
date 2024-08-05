## Preferential Attachment in Euclidian Space
# Networks with preferential attachment immersed in Euclidean space
Implementation of a model proposed by Samurai <i> et. al </i> <p>
  <a href="https://www.nature.com/articles/srep27992">Role of dimensionality in complex networks</a>.
</p>
generating random scale-free networks using a preferential attachment mechanism in Euclidian Space. 

A BFS (Breadth First Search) algorithm was implemented to calculate the shortest paths in the network in C++, along with the calculation of assortativity by degree using an adaptation of the Pearson coefficient by Newman
<p>
  <a href="https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.89.208701">Assortative Mixing in Networks</a>.
</p>


## Requirements
- cmake

- g++ compile

- nlohmann library (json files)

- boost library

- eigen3lib library

- zlib library

## Running code
```bash
# Navigate to the folder where the clone was made
cd TsallisNetwork
# Create build folder to generate executable
mkdir build && cd build
# Generate the cmake files
cmake ...
# Generate the executable
make

# The exe1 executable will be generated, the input parameters must be the standard expressed in example.json. To execute the code, we must follow the pattern
path_to_executable/exe1 path_to_json/example.json

## Data
The data will be stored in <b>/gml_folde</b> and <b>/prop_folder</b>, where the gml file stores all the link pairs in the network, with the Euclidean distance between the site pairs and the degree of each site. While the prop file stores the value of the properties shortest average network path, network diameter and assortativity by degree
