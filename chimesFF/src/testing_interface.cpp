#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <algorithm>
#include <iomanip>

#include "chimesFF.h"

//all functions/code should be roughly equivalent
//to main.py in examples/python

double get_dist(double lx, double ly, double lz,  std::vector<double> const &xcrd, std::vector<double> const &ycrd, std::vector<double> const &zcrd, int i, int j, std::vector<double> &r_ij) {
    r_ij[0] = xcrd[j] - xcrd[i];
    r_ij[0] -= lx*round(r_ij[0]/lx);
    r_ij[1] = ycrd[j] - ycrd[i];
    r_ij[1] -= ly*round(r_ij[1]/ly);
    r_ij[2] = zcrd[j] - zcrd[i];
    r_ij[2] -= lz*round(r_ij[2]/lz);

    double dist_ij = sqrt(r_ij[0]*r_ij[0] + r_ij[1]*r_ij[1] + r_ij[2]*r_ij[2]);
    return dist_ij;
}

int split(string line, vector<string> & items)
{
    // Break a line up into tokens based on space separators.
    // Returns the number of tokens parsed.
    
    string contents;
    stringstream sstream;

    sstream.str(line);
     
    items.clear();

    while ( sstream >> contents ) 
        items.push_back(contents);

    return items.size();
}

int main(int argc, char* argv[]) {
    std::cout << argv[1] << " " << argv[2];
    std::string param_file(argv[1]);
    std::string coord_file(argv[2]);

    chimesFF chimes;
    chimes.init(0); //MPI rank of zero
    chimes.read_parameters(param_file);

    // time to do some file reading for the cord file
    // depression.

    int natoms;
    double lx;
    double ly;
    double lz;

    //atoms is a 1-d vector of atom type (strings)
    //strings/characters
    std::vector<std::string> atoms;
    std::vector<double> xcrd;
    std::vector<double> ycrd;
    std::vector<double> zcrd;

    std::ifstream cord_stream;
    cord_stream.open(coord_file);
    
    std::string input;
    std::getline(cord_stream, input);
    natoms = std::stoi(input);
    atoms.reserve(natoms);
    xcrd.reserve(natoms);
    ycrd.reserve(natoms);
    zcrd.reserve(natoms);

    std::vector<std::string> split_input;
    std::getline(cord_stream, input);
    split(input, split_input);

    lx = std::stod(split_input[0]);
    ly = std::stod(split_input[4]);
    lz = std::stod(split_input[8]);

    //assuming an orthombic system, these are the
    //only ones we care about - all others
    //will be zero and this makes the algebra nicer.



    double energy = 0.0;
    std::vector<double> stress(6, 0.0); // RKL: Changed from 9 elements - chimesFF only computes unique stress tensor componenets - see chimesFF.cpp line 1474
    //Originally using 9 was what led to zeros showing up in the final results,
    //since not all of them were updated.
    //forces should be natoms by 3
    std::vector<std::vector<double> > forces(natoms, std::vector<double>(3));
    //defaults to zero initialization.

    for (int i = 0; i < natoms; i++) {
        //read in each line containing the type
        //of atom and its x y and z coordinates.
        std::getline(cord_stream, input);
        split(input, split_input);
        atoms.push_back(split_input[0]);
        xcrd.push_back(std::stod(split_input[1]));
        ycrd.push_back(std::stod(split_input[2]));
        zcrd.push_back(std::stod(split_input[3]));
    }

    cord_stream.close();

    double maxcut_2b = chimes.max_cutoff_2B();
    double maxcut_3b = chimes.max_cutoff_3B();
    double maxcut_4b = chimes.max_cutoff_4B();

    std::vector<double> maxcuts{maxcut_2b, maxcut_3b, maxcut_4b};

    double maxcut = *max_element(maxcuts.begin(), maxcuts.end());

    int order_2b = chimes.poly_orders[0];
    int order_3b = chimes.poly_orders[1];
    int order_4b = chimes.poly_orders[2];

    std::vector<double> r_ij(3, 0.0); //x,y,z distance
    double dist_ij; //distance between two atoms.
    std::vector<int> typ_idxs{0, 0}; //type indexes to be passed to compute calls
    std::vector<int> all_typ_idxs; //type indexes for each atom.

    //need to setup the type index for each atom.
    for (int i = 0; i < natoms; i++) {
        for (int j = 0; j < chimes.atmtyps.size(); j++) {
            if (atoms[i] == chimes.atmtyps[j]) {
                all_typ_idxs.push_back(j);
                break;
            }
        }
    }
    
    //chimes tmps which are necessary to call the compute methods
    
    chimes2BTmp tmp_2b(order_2b);
    chimes3BTmp tmp_3b(order_3b);
    chimes4BTmp tmp_4b(order_4b);

    for (int i = 0; i < natoms; i++) {
        chimes.compute_1B(all_typ_idxs[i], energy);
        for (int j = i + 1; j < natoms; j++) {
            dist_ij = get_dist(lx, ly, lz, xcrd, ycrd, zcrd, i, j, r_ij);

            if (dist_ij >= maxcut)
                continue;
            
            typ_idxs[0] = all_typ_idxs[i];
            typ_idxs[1] = all_typ_idxs[j];
            
            // RKL: Copying contents of forces into flat forces isn't necessary and leads to double counting when you 
            //      accumulate forces in the loop over k below. 
            //      Replaced these two lines with a simpler flat force
              
            // std::vector<double> flat_force(std::begin(forces[i]), std::end(forces[i]));
            // flat_force.insert(std::end(flat_force), std::begin(forces[j]), std::end(forces[j]));
            
            std::vector<double > flat_force(6,0);
            
            //forces need to be flattened together to 
            //need to flatten the forces we want together.
            chimes.compute_2B(dist_ij, r_ij, typ_idxs, flat_force, stress, energy, tmp_2b);
            //update forces after the call to compute_2B has completed.
            for (int k = 0; k < 3; k++) {
                // RKL: Commented out these lines due to double counting of forces (see comment on line 159 above)
                //      With the old way of doing this, flat_force would already contain the previous value of forces[j][k],
                //      so these two lines have the effect of: forces[j][k] += forces[j][k] + <new contributions computed in chimes.compute_2b>
                // forces[j][k] += flat_force[3 + k];
                // forces[i][k] += flat_force[k];  // this should accumulate due to having multiple atoms?                
                forces[i][k] += flat_force[k];  // this should accumulate due to having multiple atoms?
                forces[j][k] += flat_force[3 + k];
                
            }

            //add three and four body interactions after this later.
        }
    }
    //print the results to a separate file
    std::ofstream out_file;
    out_file.open("single_test_output.txt");
    // TODO will need to update the naming scheme at some point.

    std::setprecision(6);
    out_file << std::fixed << energy << "\n";
    //print stress tensor
    // RKL: Bookkeeping here is a bit off. It's understandably confusing, since the serial_chimes_interace's main.cpp, 
    //      python's main.cpp, and chimesFF.compute_2B all handle stresses differently.
    //      in essence, chimes_FF's compute_2b only computes the 6 unique stress tensor components, in the order xx, xy, xz, yy, yz, zz
    //      whereas the test suite prints them out in the order xx, yy, zz, xy, xz, yz. (see line:
    // Line 1474 in chimes_calculator/chimesFF/src/chimes_FF.cpp
    // Line 1071 in chimes_calculator/serial_interface/src/serial_chimes_interface.cpp
    // Line 155 in chimes_calculator/serial_interface/examples/cpp/main.cpp 
    // That's been fixed below.
    
    
    out_file << stress[0]*6.9479/lx/ly/lz << "\n"; // xx
    out_file << stress[3]*6.9479/lx/ly/lz << "\n"; // yy
    out_file << stress[5]*6.9479/lx/ly/lz << "\n"; // zz
    out_file << stress[1]*6.9479/lx/ly/lz << "\n"; // xy
    out_file << stress[2]*6.9479/lx/ly/lz << "\n"; // xz
    out_file << stress[4]*6.9479/lx/ly/lz << "\n"; // yz
  

    //print forces on atoms.
    for (int i = 0; i < natoms; i++) {
        out_file << std::scientific << forces[i][0] << "\n" << forces[i][1] << "\n" << forces[i][2] << "\n";
    }

}
