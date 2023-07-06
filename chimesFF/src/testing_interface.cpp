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
    std::vector<double> stress(9, 0.0);
    //forces should be natoms by 3
    std::vector<std::vector<double> > forces(natoms, std::vector<double>(3));
    //defaults to zero initialization.

    for (int i = 0; i < natoms; i++) {
        //read in each line containing the type
        //of atom and its coordinates.
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

    //what storage vectors do I need?

    std::vector<double> r_ij(3, 0.0);
    double dist_ij;
    std::vector<int> typ_idxs{0, 0};
    std::vector<int> all_typ_idxs;

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
        for (int j = i + 1; j < natoms; j++) {
            dist_ij = get_dist(lx, ly, lz, xcrd, ycrd, zcrd, i, j, r_ij);

            if (dist_ij >= maxcut)
                continue;
            typ_idxs[0] = all_typ_idxs[i]; //isnt this a string????
            typ_idxs[1] = all_typ_idxs[j];
            //is there something I am missing to convert from string to
            //integer.
            std::vector<double> flat_force(std::begin(forces[i]), std::end(forces[i]));
            flat_force.insert(std::end(flat_force), std::begin(forces[j]), std::end(forces[j]));
            //forces need to be flattened together to 
            //need to flatten the forces we want together.
            chimes.compute_2B(dist_ij, r_ij, typ_idxs, flat_force, stress, energy, tmp_2b);
            //update forces after the call to compute_2B has completed.
            for (int k = 0; k < 3; k++) {
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
    out_file << stress[0]*6.9479/lx/ly/lz << "\n" << stress[4]*6.9479/lx/ly/lz << "\n";
    out_file << stress[8]*6.9479/lx/ly/lz << "\n" << stress[1]*6.9479/lx/ly/lz << "\n";
    out_file << stress[2]*6.9479/lx/ly/lz << "\n" << stress[5]*6.9479/lx/ly/lz << "\n";
    //print forces on atoms.
    for (int i = 0; i < natoms; i++) {
        out_file << std::scientific << forces[i][0] << "\n" << forces[i][1] << "\n" << forces[i][2] << "\n";
    }

    // for (int i = 0; i < 9; i++) {
    //    std::cout << stress[i]/lx/ly/lz*6.9479 << "\n";
    // }

}