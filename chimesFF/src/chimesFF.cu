/* 
    ChIMES Calculator
    Copyright (C) 2020 Rebecca K. Lindsey, Nir Goldman, and Laurence E. Fried
    Contributing Author:  Rebecca K. Lindsey (2020) 
*/

#include<vector>
#include<iostream>
#include<iomanip>
#include<fstream>
#include<string>
#include<sstream>
#include<cstdlib>
#include<algorithm>
#include<cmath>
#include<map>

using namespace std;

#include "chimesFF.h"
#include "gpu_compute.h"   

template <typename T>
int get_index(const vector<T>  & vec, const T  & element)
{
    auto it = find(vec.begin(), vec.end(), element);
 
    if (it != vec.end())
    {
        return distance(vec.begin(), it);
    }
    else
    {
        cout << "chimesFF: " << "ERROR: Could not find element in vector" << endl;
        exit(0);
    }
}

template <typename T>
int get_index_if(const vector<T>  & vec, const T  & element, vector<bool> & disqualified)
{

    if (disqualified.size() != vec.size())
    {
        cout << "chimesFF: " << "ERROR: get_index_if(...): Qualification criteria does not match vector length" << endl;
        cout << "chimesFF: " << "vec.size(): " << vec.size() << endl;
        cout << "chimesFF: " << "disqualified.size(): " << disqualified.size() << endl;
        exit(0);
    }

    for(int i=0; i<vec.size(); i++)
    {
        if ((vec[i]==element) && (!disqualified[i]))
        {
            disqualified[i] = true;
            return i;
        }
    }

    cout << "chimesFF: " << "ERROR: Could not find element in vector: " << element << endl;
    
    for(int i=0; i<vec.size(); i++)
        cout << "chimesFF: " << "\t" << vec[i] << " " << disqualified[i] << endl;
    
    exit(0);
}

int chimesFF::get_proper_pair(string ty1, string ty2)
{

    for(int i=0; i<pair_params_atm_chem_1.size(); i++)
    {
        if (ty1 == pair_params_atm_chem_1[i])
            if (ty2 == pair_params_atm_chem_2[i])
                return i;
        
        if (ty2 == pair_params_atm_chem_1[i])
            if (ty1 == pair_params_atm_chem_2[i])
                return i;
    }
            
    cout << "chimesFF: " << "ERROR: No proper pair name found for atom types" << ty1 << ", " << ty2 << endl;
    exit(0);
}

chimesFF::chimesFF()
{
    natmtyps = 0;
    penalty_params.resize(2);
    
    // Set defaults
    
    fcut_type = fcutType::CUBIC ;
    
    penalty_params[0] = 0.01;
    penalty_params[1] = 1.0E4;

    inner_smooth_distance = 0.01 ;
	
}
chimesFF::~chimesFF(){}

void chimesFF::init(int mpi_rank)
{
    rank = mpi_rank;
    print_pretty_stuff();
}

void chimesFF::print_pretty_stuff()
{
    if (rank == 0)
    {
        cout << "chimesFF: " <<  endl;
        cout << "chimesFF: " << "01000011011010001001001010011010100010101010011 0100010101101110110011101101001011011101100101  " << endl;
        cout << "chimesFF: " <<  endl;
        cout << "chimesFF: " << "      _____  _      _____  __  __  ______   _____   ______                _                      " << endl;
        cout << "chimesFF: " << "     / ____|| |    |_   _||  \\/  ||  ____| / ____| |  ____|              (_)                    " << endl;
        cout << "chimesFF: " << "    | |     | |__    | |  | \\  / || |__   | (___   | |__    _ __    __ _  _  _ __    ___        " << endl;
        cout << "chimesFF: " << "    | |     | '_ \\   | |  | |\\/| ||  __|   \\___ \\  |  __|  | '_ \\  / _` || || '_ \\  / _ \\ " << endl;
        cout << "chimesFF: " << "    | |____ | | | | _| |_ | |  | || |____  ____) | | |____ | | | || (_| || || | | ||  __/        " << endl;
        cout << "chimesFF: " << "     \\_____||_| |_||_____||_|  |_||______||_____/  |______||_| |_| \\__, ||_||_| |_| \\___|     " << endl;
        cout << "chimesFF: " << "                                                                    __/ |                        " << endl;
        cout << "chimesFF: " << "                                                                   |___/                         " << endl;  
        cout << "chimesFF: " << endl;
        cout << "chimesFF: " << "                     Copyright (C) 2020 R.K. Lindsey, L.E. Fried, N. Goldman                     " << endl;    
        cout << "chimesFF: " << endl;
        cout << "chimesFF: " << "01000011011010001001001010011010100010101010011 0100010101101110110011101101001011011101100101   " << endl;
        cout << "chimesFF: " << endl;
    }
      
}

int chimesFF::split_line(string line, vector<string> & items)
{
    // Break a line up into tokens based on space separators.
    // Returns the number of tokens parsed.
    
    string       contents;
    stringstream sstream;

    // Strip comments beginining with ! or ## and terminal new line

    int pos = line.find('!');
      
    if ( pos != string::npos ) 
        line.erase(pos, line.length() - pos);

    pos = line.find("##");
    if ( pos != string::npos ) 
        line.erase(pos, line.length()-pos);

    pos = line.find('\n');
    if ( pos != string::npos ) 
        line.erase(pos, 1);

    sstream.str(line);
     
    items.clear();

    while ( sstream >> contents ) 
        items.push_back(contents);

    return items.size();
}

string chimesFF::get_next_line(istream& str)
{
    // Read a line and return it, with error checking.
    
    string line;

    getline(str, line);
    
    if ( ! str.good() )
    {
        if (rank == 0)
            cout << "chimesFF: " << "Error reading line" << line << endl;
        exit(0);
    } 

    return line;
}

void chimesFF::read_parameters(string paramfile)
{
    // Open the parameter file, run sanity checks
    
    ifstream param_file;
    param_file.open(paramfile.data());
    
    if (rank == 0)
        cout << "chimesFF: " << "Reading parameters from file: " << paramfile << endl;
    
    if(!param_file.is_open())
    {
        if (rank == 0)
            cout << "chimesFF: " << "ERROR: Cannot open parameter file: " << paramfile << endl;
        exit(0);
    }
    
    // Declare parsing variables
    
    
    bool           found_end = false;
    string         line;
    string         tmp_str;
    vector<string> tmp_str_items;
    int            tmp_no_items;
    int            tmp_int;
    int            no_pairs;
    
    // Check that this is actually a chebyshev parameter set

    while (!found_end)
    {
        line = get_next_line(param_file);

           // Break out of loop

           if(line.find("ENDFILE") != string::npos)
        {
            if (rank == 0)
            {
                cout << "chimesFF: " << "ERROR: Could not find line containing: \" PAIRTYP: CHEBYSHEV\" " << endl;
                cout << "chimesFF: " << "       ...Is this a ChIMES force field parameter file?" << endl;
            }
            exit(0);
        }
        
        if(line.find("PAIRTYP: CHEBYSHEV") != string::npos)
        {
            tmp_no_items = split_line(line, tmp_str_items);
            
            if (tmp_no_items < 3)
            {    
                if (rank == 0)
                    cout << "chimesFF: " << "ERROR: \"PAIRTYP: CHEBYSHEV\" line must at least contain the 2-body order" << endl;
                exit(0);
            }
            
            poly_orders.push_back(stoi(tmp_str_items[2]));
            
            if (tmp_no_items >= 4)
                poly_orders.push_back(stoi(tmp_str_items[3]));

            if (tmp_no_items >= 5)
                poly_orders.push_back(stoi(tmp_str_items[4]));    
            
            while (poly_orders.size() < 3)
                poly_orders.push_back(0);
            
            if (rank == 0)
            {
                cout << "chimesFF: " << "Using respective 2, 3, and 4-body orders of: " << poly_orders[0] << " " << poly_orders[1] << " " << poly_orders[2] << endl;
            
                cout << "chimesFF: " << "Note: Ignoring polynomial domain; assuming [-1,1]" << endl;    
            }
            
            break;    
        }
    }
    
    // If we've made it to here, then this should contain Chebyshev params. Rewind and start looking for general information
        
    param_file.seekg(0);
    
    found_end = false;
    
    while (!found_end)
    {
        line = get_next_line(param_file);
        
           if(line.find("ENDFILE") != string::npos)
            break;        
    
        if(line.find("ATOM TYPES:") != string::npos)
        {
            tmp_no_items = split_line(line, tmp_str_items);
        
            natmtyps = stoi(tmp_str_items[2]);
        
            if (rank == 0)
                cout << "chimesFF: " << "Will consider " << natmtyps << " atom types:" << endl;
                
            energy_offsets.resize(natmtyps);
            
            for(int i=0; i<natmtyps; i++)
                energy_offsets[i] = 0.0;
        }
        
        if(line.find("# TYPEIDX #") != string::npos)
        {
            atmtyps.resize(natmtyps);
			masses.resize(natmtyps);
            for (int i=0; i<natmtyps; i++)
            {
                line = get_next_line(param_file);
                split_line(line, tmp_str_items);
                atmtyps[i] = tmp_str_items[1];
				masses[i]  = stod(tmp_str_items[3]);
                
                if (rank == 0)
                    cout << "chimesFF: " << "\t" << i << " " << atmtyps[i] << endl;
            }
            
        }
            
        if(line.find("ATOM PAIRS:") != string::npos)
        {
            tmp_no_items = split_line(line, tmp_str_items);
        
            no_pairs = stoi(tmp_str_items[2]);
        
            if (rank == 0)
                cout << "chimesFF: " << "Will consider " << no_pairs << " atom pair types" << endl;        
        }    
        
        if(line.find("# PAIRIDX #") != string::npos)
        {
            if(line.find("# USEOVRP #") != string::npos)
                continue;
        
            pair_params_atm_chem_1.resize(no_pairs);
            pair_params_atm_chem_2.resize(no_pairs);
            chimes_2b_cutoff      .resize(no_pairs);
            morse_var             .resize(no_pairs);
            
            ncoeffs_2b            .resize(no_pairs);
            chimes_2b_pows        .resize(no_pairs);
            chimes_2b_params      .resize(no_pairs);
            chimes_2b_cutoff      .resize(no_pairs);

            string tmp_xform_style;
            
            for (int i=0; i<no_pairs; i++)
            {
                line = get_next_line(param_file);
                
                tmp_no_items = split_line(line, tmp_str_items);

                int pair_input_version = 0;
				
                if ( tmp_no_items == 8 )
                {
					if ( rank == 0 && i == 0 ) cout << "chimesFF: Detected version 1 pair specification (with S_DELTA)\n";
					pair_input_version = 1;
                }
                else if ( tmp_no_items == 7 )
                {
					if ( rank == 0 && i == 0 ) cout << "chimesFF: Detected version 2 pair specification (no S_DELTA)\n";
					pair_input_version = 2;
                }
                else
                {
					if ( rank == 0 )
					{
						cout << "Incorrect input in line: " << line << endl;
						cout << "Expect 7 or 8 entries\n";
					}
					exit(0);
                }
            
                pair_params_atm_chem_1[i] = tmp_str_items[1];
                pair_params_atm_chem_2[i] = tmp_str_items[2];
                
                if (rank == 0)
                    cout << "chimesFF: " << "\t" << i << " " << pair_params_atm_chem_1[i] << " " << pair_params_atm_chem_2[i]<< endl;
                
                chimes_2b_cutoff[i].push_back(stod(tmp_str_items[3])); // Inner cutoff    
                chimes_2b_cutoff[i].push_back(stod(tmp_str_items[4])); // Outer cutoff

                int xform_style_idx, morse_idx;
				
                if ( pair_input_version == 1 )
                {
					xform_style_idx = 6;
					morse_idx = 7;
                }
                else if ( pair_input_version == 2 )
                {
					xform_style_idx = 5;
					morse_idx = 6;
                } 
                else
                {
					if ( rank == 0 ) cout << "Bad pair input version\n";
					exit(0);
                }
                    
                if (i==0)
                {
                    tmp_xform_style = tmp_str_items[xform_style_idx];
                }
                else if ( tmp_str_items[xform_style_idx] != tmp_xform_style)    
                {
					if (rank == 0)
						cout << "chimesFF: " << "Distance transformation style must be the same for all pair types" << endl;
					exit(0);
                }

                if (tmp_xform_style == "MORSE" )
                {
					if ( tmp_no_items > morse_idx )
						morse_var[i] = stod(tmp_str_items[morse_idx]);
					else {
						if ( rank == 0 )
							cout << "chimesFF: Missing morse lambda value in line: \n" << line << endl;
						exit(0);
					}
				}
            }
                
            xform_style = tmp_xform_style;
            
            if (rank == 0)
                cout << "chimesFF: " << "Read the following pair type information:" << endl;
            
            for (int i=0; i<no_pairs; i++)
            {
				if (rank == 0)
					cout << "chimesFF: " << "\t" << pair_params_atm_chem_1[i] << " " << pair_params_atm_chem_2[i] << " r_cut_in: " << fixed << right << setprecision(5) << chimes_2b_cutoff[i][0] << " r_cut_out: " << chimes_2b_cutoff[i][1] << " " <<  xform_style;
                
				if (xform_style == "MORSE")
				{
					if (rank == 0)
						cout << " " << morse_var[i] << endl;
				}
				else
					if (rank == 0)
						cout << endl;
			}
        }
            
        if(line.find("FCUT TYPE:") != string::npos)
        {
            tmp_no_items = split_line(line, tmp_str_items);
        
            if ( tmp_str_items[2] == "CUBIC" )
                fcut_type = fcutType::CUBIC ;
            else if ( tmp_str_items[2] == "TERSOFF" )
                fcut_type = fcutType::TERSOFF ;
            else
            {
                if ( rank == 0 ) 
                    cout << "Error: unknown FCUT TYPE: " << tmp_str_items[2] << endl ;
                exit(1) ;
            }
                    
            if (rank == 0)
                cout << "chimesFF: " << "Will use cutoff style " << tmp_str_items[2] << endl ;
            
            if (fcut_type == fcutType::TERSOFF )
            {
                fcut_var = stod(tmp_str_items[3]);
                
                if (rank == 0)
                    cout << " " << fcut_var << endl;
            }
            else
                if (rank == 0)
                    cout << endl;
        }
        
        if(line.find("PAIR CHEBYSHEV PENALTY DIST:") != string::npos)
        {    
            tmp_no_items = split_line(line, tmp_str_items);
            
            penalty_params[0] = stod(tmp_str_items[4]);
            
            if (rank == 0)
                cout << "chimesFF: " << "Will use penalty distance: " << penalty_params[0] << endl;
        }
        
        if(line.find("PAIR CHEBYSHEV PENALTY SCALING:") != string::npos)
        {    
            tmp_no_items = split_line(line, tmp_str_items);
            
            penalty_params[1] = stod(tmp_str_items[4]);
            
            if (rank == 0)
                cout << "chimesFF: " << "Will use penalty scaling: " << penalty_params[1] << endl;
        }
        
        if(line.find("NO ENERGY OFFSETS:") != string::npos)
        {
            int tmp_no = split_line(line, tmp_str_items);
                        
            if(stoi(tmp_str_items[tmp_no-1]) != natmtyps)
            {
                cout << "chimesFF: " << "ERROR: Number of energy offsets do not match number of atom types" << endl;
                exit(0);
            }

            // Expects atom offsets in the same order as atom types were provided originally
            
            if (rank == 0)
                cout << "chimesFF: " << "Will use single atom energy offsets: "<< endl;
            
            int tmp_idx;
            
            for (int i=0; i<natmtyps; i++)
            {
                line = get_next_line(param_file);
                split_line(line, tmp_str_items);
                tmp_idx = stoi(tmp_str_items[2]);
                
                energy_offsets[tmp_idx-1] = stod(tmp_str_items[3]);
                
                if (rank == 0)
                    cout << "chimesFF: " << "\t" << tmp_idx << " " << atmtyps[tmp_idx-1] << " " << energy_offsets[tmp_idx-1] << endl;
            }
            
        }                
    }
    
    // Rewind and read the 2-body Chebyshev pair parameters
    
    param_file.seekg(0);
    
    found_end = false;
    
    while (!found_end)
    {
		line = get_next_line(param_file);

		if(line.find("ENDFILE") != string::npos)
			break;            
        
        if(line.find("PAIRTYPE PARAMS:") != string::npos)
        {
            tmp_no_items = split_line(line, tmp_str_items);
            
            tmp_int = stoi(tmp_str_items[2]);
            
            if (rank == 0)
                cout << "chimesFF: " << "Read 2B parameters for pair: " << tmp_int << " " << tmp_str_items[3] << " " << tmp_str_items[4] << endl;
            
            line = get_next_line(param_file);
            
            split_line(line, tmp_str_items); // Empty line
            
            ncoeffs_2b[tmp_int] = poly_orders[0];
            
            for(int i=0; i<poly_orders[0]; i++)
            {
                line = get_next_line(param_file);
                split_line(line, tmp_str_items);
                
                chimes_2b_pows  [tmp_int].push_back(stoi(tmp_str_items[0]));                
                chimes_2b_params[tmp_int].push_back(stod(tmp_str_items[1]));
                
                if (rank == 0)
                    cout << "chimesFF: " << "\t" << chimes_2b_pows[tmp_int][i] << " " << chimes_2b_params[tmp_int][i] << endl;
            }
        }
        
        if(line.find("PAIRMAPS:") != string::npos)
        {
            // Read the slow map and build the fast map
            
            tmp_no_items = split_line(line, tmp_str_items);
            
            n_pair_maps = stoi(tmp_str_items[1]);
            
            atom_typ_pair_map.resize(n_pair_maps);
            atom_idx_pair_map.resize(n_pair_maps);
            
            atom_int_prpr_map.resize(n_pair_maps);
            
            if (rank == 0)
                cout << "chimesFF: " << "Built the following 2-body pair \"slow\" map:" << endl;
            
            for(int i=0; i<n_pair_maps; i++)
            {
                line = get_next_line(param_file);
                split_line(line, tmp_str_items);
                
                atom_idx_pair_map[i] = stoi(tmp_str_items[0]);
                atom_typ_pair_map[i] =      tmp_str_items[1];
                
                if (rank == 0)
                    cout << "chimesFF: " << "\t" << atom_idx_pair_map[i] << " " << atom_typ_pair_map[i] << "(i: " << i << ")" << endl;

            }

            if (rank == 0)
                cout << "chimesFF: " << "Built the following 2-body pair \"fast\" map:" << endl;
            
            atom_int_pair_map.resize((natmtyps-1)*natmtyps + (natmtyps-1) + 1); // Maximum possible pair value + a small buffer
            

            for(int i=0; i<natmtyps; i++)
            {
                for (int j=0; j<natmtyps; j++)
                {
                    // Get the pair type name for the set of atoms
                    
                    tmp_str = atmtyps[i] + atmtyps[j];

                    tmp_int = get_index(atom_typ_pair_map, tmp_str);
                    
                    atom_int_pair_map[ i*natmtyps + j ] = atom_idx_pair_map[tmp_int];
                    

                    tmp_int = get_proper_pair(atmtyps[i],atmtyps[j]);
                    
                    atom_int_prpr_map [ i*natmtyps + j ] = pair_params_atm_chem_1[tmp_int] + pair_params_atm_chem_2[tmp_int];

                    
                    if (rank == 0)
                        cout << "chimesFF: " << "\t" << tmp_str << ": " << i*natmtyps + j << " " << atom_int_pair_map[ i*natmtyps + j ] << endl;

                }
            }                        
        }
    }
    
    // Rewind and read the 3-body Chebyshev pair parameters
    
    if (poly_orders[1] > 0)
    {
        int ntrips;
        int tmp_idx;
        
        // Read parameters
        
        param_file.seekg(0);
        
        found_end = false;
    
        while (!found_end)
        {
            line = get_next_line(param_file);
        
               if(line.find("ENDFILE") != string::npos)
                break;    
            
            if(line.find("ATOM PAIR TRIPLETS:") != string::npos)
            {
                split_line(line, tmp_str_items);
                
                ntrips = stoi(tmp_str_items[3]);
                                
                ncoeffs_3b      .resize(ntrips);
                chimes_3b_powers.resize(ntrips);                
                chimes_3b_params.resize(ntrips);
                chimes_3b_cutoff.resize(ntrips);
    
                
                trip_params_atm_chems.resize(ntrips);                
                trip_params_pair_typs.resize(ntrips);
            }
            
            if(line.find("TRIPLETTYPE PARAMS:") != string::npos)
            {
                vector<int> tmp_int_vec(3);

                line = get_next_line(param_file);
                
                split_line(line, tmp_str_items);

                tmp_int = stoi(tmp_str_items[1]);
                
                trip_params_atm_chems[tmp_int].push_back(tmp_str_items[3]);
                trip_params_atm_chems[tmp_int].push_back(tmp_str_items[4]);
                trip_params_atm_chems[tmp_int].push_back(tmp_str_items[5]);

                if (rank == 0)
                    cout << "chimesFF: " << "Read 3B parameters for triplet: " << tmp_int << " " << trip_params_atm_chems[tmp_int][0] << " " << trip_params_atm_chems[tmp_int][1] << " " << trip_params_atm_chems[tmp_int][2] << endl;
                
                line = get_next_line(param_file);
                
                split_line(line, tmp_str_items);
            
                trip_params_pair_typs[tmp_int].push_back(tmp_str_items[1]);
                trip_params_pair_typs[tmp_int].push_back(tmp_str_items[2]);
                trip_params_pair_typs[tmp_int].push_back(tmp_str_items[3]);
		
		// Check for excluded triplet types
	
		if(tmp_str_items[4] != "EXCLUDED:")
		{
                	ncoeffs_3b[tmp_int] = stoi(tmp_str_items[7]);    
	
        	        get_next_line(param_file);
        	        get_next_line(param_file);
            
        	        for(int i=0; i<ncoeffs_3b[tmp_int]; i++)
        	        {
        	            line = get_next_line(param_file);
        	            split_line(line, tmp_str_items);
                    
        	            tmp_int_vec[0] = stoi(tmp_str_items[1]);
        	            tmp_int_vec[1] = stoi(tmp_str_items[2]);
        	            tmp_int_vec[2] = stoi(tmp_str_items[3]);
                    
        	            chimes_3b_powers[tmp_int].push_back(tmp_int_vec);                    
        	            chimes_3b_params[tmp_int].push_back(stod(tmp_str_items[6]));
                
        	            if (rank == 0)
        	                cout << "chimesFF: " << "\t" << chimes_3b_powers[tmp_int][i][0] << " " << chimes_3b_powers[tmp_int][i][1] << " " << chimes_3b_powers[tmp_int][i][2] << " " << chimes_3b_params[tmp_int][i] << endl;
        	        }
		}
		else
		{
			cout << "chimesFF: \tType is excluded... skipping." << endl;
		}
            }    
            
            if(line.find("TRIPMAPS:") != string::npos)
            {
                split_line(line, tmp_str_items);
                
                n_trip_maps = stoi(tmp_str_items[1]);
                
                atom_idx_trip_map.resize(n_trip_maps);
                atom_typ_trip_map.resize(n_trip_maps);
                
                if (rank == 0)                
                    cout << "chimesFF: " << "Built the following 3-body pair \"slow\" map:" << endl;
            
                for(int i=0; i<n_trip_maps; i++)
                {
                    line = get_next_line(param_file);
                    split_line(line, tmp_str_items);
                
                    atom_idx_trip_map[i] = stoi(tmp_str_items[0]);
                    atom_typ_trip_map[i] =      tmp_str_items[1];
                
                    if (rank == 0)
                        cout << "chimesFF: " << "\t" << atom_idx_trip_map[i] << " " << atom_typ_trip_map[i] << endl;
                }        
                
                if (rank == 0)
                    cout << "chimesFF: " << "Built the following 3-body pair \"fast\" map:" << endl;

                atom_int_trip_map.resize(natmtyps*natmtyps*natmtyps);
                
                for(int i=0; i<natmtyps; i++)
                {
                    for (int j=0; j<natmtyps; j++)
                    {
                        for(int k=0; k<natmtyps; k++)
                        {
                            // Get the trip type name for the set of atoms
                            
                            tmp_str = "";

                            tmp_int  = get_proper_pair(atmtyps[i], atmtyps[j]);
                            tmp_str += pair_params_atm_chem_1[tmp_int] + pair_params_atm_chem_2[tmp_int];    
                
                            tmp_int  = get_proper_pair(atmtyps[i], atmtyps[k]);
                            tmp_str += pair_params_atm_chem_1[tmp_int] + pair_params_atm_chem_2[tmp_int];    
                
                            tmp_int  = get_proper_pair(atmtyps[j], atmtyps[k]);
                            tmp_str += pair_params_atm_chem_1[tmp_int] + pair_params_atm_chem_2[tmp_int];            
                                            
                            tmp_int = get_index(atom_typ_trip_map, tmp_str);

                            tmp_idx = i*natmtyps*natmtyps + j*natmtyps + k;

                            atom_int_trip_map[ tmp_idx ] = atom_idx_trip_map[tmp_int];
                                                        
                            if (rank == 0)
                                cout << "chimesFF: " << "\t" << tmp_idx << " " << atom_int_trip_map[ tmp_idx  ]  << endl;
                        }
                    }
                }
            }            
        }
        
        // Set up cutoffs ... First set to match 2-body, then read special if they exist
        
        int atmtyp_1,  atmtyp_2,  atmtyp_3;
        int pairtyp_1, pairtyp_2, pairtyp_3;

        for(int i=0; i<ntrips; i++) 
        {
            // Figure out the atom type index for each atom in the triplet type 
                        
            atmtyp_1 = distance(atmtyps.begin(), find(atmtyps.begin(), atmtyps.end(), trip_params_atm_chems[i][0]));    
            atmtyp_2 = distance(atmtyps.begin(), find(atmtyps.begin(), atmtyps.end(), trip_params_atm_chems[i][1]));    
            atmtyp_3 = distance(atmtyps.begin(), find(atmtyps.begin(), atmtyps.end(), trip_params_atm_chems[i][2]));    
                        
            // Figure out the corresponding 2-body pair type
            
            pairtyp_1 = atom_int_pair_map[ atmtyp_1*natmtyps + atmtyp_2 ];
            pairtyp_2 = atom_int_pair_map[ atmtyp_1*natmtyps + atmtyp_3 ];
            pairtyp_3 = atom_int_pair_map[ atmtyp_2*natmtyps + atmtyp_3 ];
    
            // Set the default inner/outer cutoffs to the corresponding 2-body value

            chimes_3b_cutoff[i].resize(2);

            chimes_3b_cutoff[i][0].push_back(chimes_2b_cutoff[pairtyp_1][0]);
            chimes_3b_cutoff[i][0].push_back(chimes_2b_cutoff[pairtyp_2][0]);
            chimes_3b_cutoff[i][0].push_back(chimes_2b_cutoff[pairtyp_3][0]);
            
            chimes_3b_cutoff[i][1].push_back(chimes_2b_cutoff[pairtyp_1][1]);
            chimes_3b_cutoff[i][1].push_back(chimes_2b_cutoff[pairtyp_2][1]);
            chimes_3b_cutoff[i][1].push_back(chimes_2b_cutoff[pairtyp_3][1]);                    
        }
        
        param_file.seekg(0);
        
        int    nentries;
        double cutval;
        
        found_end = false;
        
        while (!found_end)
        {
            line = get_next_line(param_file);
        
			if(line.find("ENDFILE") != string::npos)
                break;                
            
            if(line.find("SPECIAL 3B S_MAXIM:") != string::npos)
            {
                split_line(line, tmp_str_items);
                
                if (rank == 0)
                    cout << "chimesFF: " << "Set the following special 3-body outer cutoffs: " << endl;
                
                if(tmp_str_items[3] == "ALL")
                {
                    cutval = stod(tmp_str_items[4]);
                                        
                    for(int i=0; i<ntrips; i++)
                    {
                        chimes_3b_cutoff[i][1][0] = cutval;
                        chimes_3b_cutoff[i][1][1] = cutval;
                        chimes_3b_cutoff[i][1][2] = cutval;                    
                    }
                }
                else
                {
                    nentries = stoi(tmp_str_items[4]);
                    
                    vector<string> pair_name(3);
                    vector<double> cutoffval(3);

    
                    for(int i=0; i<nentries; i++)
                    {
                        line = get_next_line(param_file);
                        
                        split_line(line, tmp_str_items);
                        
                        tmp_int = atom_idx_trip_map[distance(atom_typ_trip_map.begin(), find(atom_typ_trip_map.begin(), atom_typ_trip_map.end(), tmp_str_items[0]))];

                        pair_name[0] = tmp_str_items[1];
                        pair_name[1] = tmp_str_items[2];
                        pair_name[2] = tmp_str_items[3];
                        
                        cutoffval[0] = stod(tmp_str_items[4]);
                        cutoffval[1] = stod(tmp_str_items[5]);
                        cutoffval[2] = stod(tmp_str_items[6]);
                        
                        vector<bool>   disqualified(3,false);
                        
                        chimes_3b_cutoff[tmp_int][1][ get_index_if(trip_params_pair_typs[tmp_int], pair_name[0], disqualified) ] = cutoffval[0];
                        chimes_3b_cutoff[tmp_int][1][ get_index_if(trip_params_pair_typs[tmp_int], pair_name[1], disqualified) ] = cutoffval[1];
                        chimes_3b_cutoff[tmp_int][1][ get_index_if(trip_params_pair_typs[tmp_int], pair_name[2], disqualified) ] = cutoffval[2];
                                        
                    }
                }
                
                for(int i=0; i<ntrips; i++)
                    if (rank == 0)
                        cout << "chimesFF: " << "\t" << i << " " << chimes_3b_cutoff[i][1][0] << " " << chimes_3b_cutoff[i][1][1] << " " << chimes_3b_cutoff[i][1][2] << endl;
                
            }

            if(line.find("SPECIAL 3B S_MINIM:") != string::npos)
            {
                split_line(line, tmp_str_items);
                
                if (rank == 0)
                    cout << "chimesFF: " << "Set the following special 3-body inner cutoffs: " << endl;
                
                if(tmp_str_items[3] == "ALL")
                {
                    cutval = stod(tmp_str_items[4]);
                    
                    for(int i=0; i<ntrips; i++)
                    {
                        chimes_3b_cutoff[i][0][0] = cutval;
                        chimes_3b_cutoff[i][0][1] = cutval;
                        chimes_3b_cutoff[i][0][2] = cutval;
                        
                    }
                }
                else
                {
                    nentries = stoi(tmp_str_items[4]);
                    
                    vector<string> pair_name(3);
                    vector<double> cutoffval(3);


                    for(int i=0; i<nentries; i++)
                    {
                        line = get_next_line(param_file);
                        
                        split_line(line, tmp_str_items);
                        
                        tmp_int = atom_idx_trip_map[distance(atom_typ_trip_map.begin(), find(atom_typ_trip_map.begin(), atom_typ_trip_map.end(), tmp_str_items[0]))];
                        
                        pair_name[0] = tmp_str_items[1];
                        pair_name[1] = tmp_str_items[2];
                        pair_name[2] = tmp_str_items[3];
                        
                        cutoffval[0] = stod(tmp_str_items[4]);
                        cutoffval[1] = stod(tmp_str_items[5]);
                        cutoffval[2] = stod(tmp_str_items[6]);
                        
                        vector<bool>   disqualified(3,false);
                        
                        chimes_3b_cutoff[tmp_int][0][ get_index_if(trip_params_pair_typs[tmp_int], pair_name[0], disqualified) ] = cutoffval[0];
                        chimes_3b_cutoff[tmp_int][0][ get_index_if(trip_params_pair_typs[tmp_int], pair_name[1], disqualified) ] = cutoffval[1];
                        chimes_3b_cutoff[tmp_int][0][ get_index_if(trip_params_pair_typs[tmp_int], pair_name[2], disqualified) ] = cutoffval[2];
                
                    }
                }
                
                for(int i=0; i<ntrips; i++)
                    if (rank == 0)
                        cout << "chimesFF: " << "\t" << i << " " << chimes_3b_cutoff[i][0][0] << " " << chimes_3b_cutoff[i][0][1] << " " << chimes_3b_cutoff[i][0][2] << endl;
            }            
        }
        // With all of the 3 body stuff safely done at this point, we want to flatten
        // chimes_3b_powers by one dimension to make it friendly to the GPU.
        chimes_3b_powers_flat.resize(chimes_3b_powers.size());
        for (int i = 0; i < chimes_3b_powers.size(); i++) {
            for (int j = 0; j < chimes_3b_powers[i].size(); j++) {
                for (int k = 0; k < chimes_3b_powers[i][j].size(); k++) {
                    chimes_3b_powers_flat[i].push_back(chimes_3b_powers[i][j][k]); // should this be j or i?
                }
            }
        }
        
          
    }
    
    // Rewind and read the 4-body Chebyshev pair parameters
    
    if (poly_orders[2] > 0)
    {
        int nquads;
        int tmp_idx;
        
        // Read parameters
        
        param_file.seekg(0);
        
        found_end = false;
    
        while (!found_end)
        {
            line = get_next_line(param_file);
        
			if(line.find("ENDFILE") != string::npos)
                break;    
            
            if(line.find("ATOM PAIR QUADRUPLETS:") != string::npos)
            {
                split_line(line, tmp_str_items);
                
                nquads = stoi(tmp_str_items[3]);
                                
                ncoeffs_4b      .resize(nquads);                                 
                chimes_4b_powers.resize(nquads);                                              
                chimes_4b_params.resize(nquads);                       
                chimes_4b_cutoff.resize(nquads);                            
                
                quad_params_atm_chems.resize(nquads);                
                quad_params_pair_typs.resize(nquads);
            }
            
            if(line.find("QUADRUPLETYPE PARAMS:") != string::npos)
            {            
                line = get_next_line(param_file);
                
                split_line(line, tmp_str_items);
                
                tmp_int = stoi(tmp_str_items[1]);
                
                quad_params_atm_chems[tmp_int].push_back(tmp_str_items[3]);
                quad_params_atm_chems[tmp_int].push_back(tmp_str_items[4]);
                quad_params_atm_chems[tmp_int].push_back(tmp_str_items[5]);
                quad_params_atm_chems[tmp_int].push_back(tmp_str_items[6]);

                if (rank == 0)
                    cout << "chimesFF: " << "Read 4B parameters for quadruplets: " << tmp_int << " " << quad_params_atm_chems[tmp_int][0] << " " << quad_params_atm_chems[tmp_int][1] << " " << quad_params_atm_chems[tmp_int][2] << " " << quad_params_atm_chems[tmp_int][3]<< endl;
                
                line = get_next_line(param_file);
                
                split_line(line, tmp_str_items);
            
                quad_params_pair_typs[tmp_int].push_back(tmp_str_items[1]);
                quad_params_pair_typs[tmp_int].push_back(tmp_str_items[2]);
                quad_params_pair_typs[tmp_int].push_back(tmp_str_items[3]);
                quad_params_pair_typs[tmp_int].push_back(tmp_str_items[4]);
                quad_params_pair_typs[tmp_int].push_back(tmp_str_items[5]);
                quad_params_pair_typs[tmp_int].push_back(tmp_str_items[6]);   
		
		// Check for excluded triplet types
	
		if(tmp_str_items[7] != "EXCLUDED:")
		{		                         
	                ncoeffs_4b[tmp_int] = stoi(tmp_str_items[10]);    

	                get_next_line(param_file);
	                get_next_line(param_file);
            
	                vector<int> tmp_int_vec(6);
                
	                for(int i=0; i<ncoeffs_4b[tmp_int]; i++)
	                {                
	                    line = get_next_line(param_file);
	                    split_line(line, tmp_str_items);
                    
	                    tmp_int_vec[0] = stoi(tmp_str_items[1]);
	                    tmp_int_vec[1] = stoi(tmp_str_items[2]);
	                    tmp_int_vec[2] = stoi(tmp_str_items[3]);
	                    tmp_int_vec[3] = stoi(tmp_str_items[4]);
	                    tmp_int_vec[4] = stoi(tmp_str_items[5]);
	                    tmp_int_vec[5] = stoi(tmp_str_items[6]);
                    
	                    chimes_4b_powers[tmp_int].push_back(tmp_int_vec);                 
                    
	                    chimes_4b_params[tmp_int].push_back(stod(tmp_str_items[9]));
                
	                    if (rank == 0)
                        	cout << "chimesFF: " << "\t" << 
                        	chimes_4b_powers[tmp_int][i][0] << " " << 
                	        chimes_4b_powers[tmp_int][i][1] << " " << 
                	        chimes_4b_powers[tmp_int][i][2] << " " << 
                	        chimes_4b_powers[tmp_int][i][3] << " " << 
                	        chimes_4b_powers[tmp_int][i][4] << " " << 
                	        chimes_4b_powers[tmp_int][i][5] << " " <<                                
                	        chimes_4b_params[tmp_int][i] << endl;
                	}
		}
		else
		{
			cout << "chimesFF: \tType is excluded... skipping." << endl;		
		}
            }    
            
            if(line.find("QUADMAPS:") != string::npos)
            {
                split_line(line, tmp_str_items);
                
                n_quad_maps = stoi(tmp_str_items[1]);
                
                atom_idx_quad_map.resize(n_quad_maps);
                atom_typ_quad_map.resize(n_quad_maps);
                    
                if (rank == 0)            
                    cout << "chimesFF: " << "Built the following 4-body pair \"slow\" map:" << endl;
            
                for(int i=0; i<n_quad_maps; i++)
                {
                    line = get_next_line(param_file);
                    split_line(line, tmp_str_items);
                
                    atom_idx_quad_map[i] = stoi(tmp_str_items[0]);
                    atom_typ_quad_map[i] =      tmp_str_items[1];
                
                    if (rank == 0)
                        cout << "chimesFF: " << "\t" << atom_idx_quad_map[i] << " " << atom_typ_quad_map[i] << endl;
                }        
                
                if (rank == 0)
                    cout << "chimesFF: " << "Built the following 4-body pair \"fast\" map:" << endl;

                atom_int_quad_map.resize(natmtyps*natmtyps*natmtyps*natmtyps);
                
                for(int i=0; i<natmtyps; i++)
                {
                    for (int j=0; j<natmtyps; j++)
                    {
                        for(int k=0; k<natmtyps; k++)
                        {
                            for(int l=0; l<natmtyps; l++)
                            {                            
                                // Get the quad type name for the set of atoms
                            
                                tmp_str = "";
                                
                                
                                tmp_int  = get_proper_pair(atmtyps[i], atmtyps[j]);
                                tmp_str += pair_params_atm_chem_1[tmp_int] + pair_params_atm_chem_2[tmp_int];    

                                tmp_int  = get_proper_pair(atmtyps[i], atmtyps[k]);
                                tmp_str += pair_params_atm_chem_1[tmp_int] + pair_params_atm_chem_2[tmp_int];    
                
                                tmp_int  = get_proper_pair(atmtyps[i], atmtyps[l]);
                                tmp_str += pair_params_atm_chem_1[tmp_int] + pair_params_atm_chem_2[tmp_int];    
                            
                                tmp_int  = get_proper_pair(atmtyps[j], atmtyps[k]);
                                tmp_str += pair_params_atm_chem_1[tmp_int] + pair_params_atm_chem_2[tmp_int];    
                            
                                tmp_int  = get_proper_pair(atmtyps[j], atmtyps[l]);
                                tmp_str += pair_params_atm_chem_1[tmp_int] + pair_params_atm_chem_2[tmp_int];
                            
                                tmp_int  = get_proper_pair(atmtyps[k], atmtyps[l]);
                                tmp_str += pair_params_atm_chem_1[tmp_int] + pair_params_atm_chem_2[tmp_int];                                                                                                

                                tmp_int = get_index(atom_typ_quad_map, tmp_str);
                            
                                tmp_idx = i*natmtyps*natmtyps*natmtyps + j*natmtyps*natmtyps + k*natmtyps + l;

                                atom_int_quad_map[ tmp_idx ] = atom_idx_quad_map[tmp_int];

                                if (rank == 0)
                                    cout << "chimesFF: " << "\t" << tmp_idx << " " << atom_int_quad_map[ tmp_idx  ]  << endl;
                            }
                        }
                    }
                }
            }            
        }
        
        // Set up cutoffs ... First set to match 2-body, then read special if they exist
        
        int atmtyp_1,  atmtyp_2,  atmtyp_3,  atmtyp_4;
        int pairtyp_1, pairtyp_2, pairtyp_3, pairtyp_4, pairtyp_5, pairtyp_6;
        
        for(int i=0; i<nquads; i++) 
        {
            // Figure out the atom type index for each atom in the quadruplet type 
                        
            atmtyp_1 = distance(atmtyps.begin(), find(atmtyps.begin(), atmtyps.end(), quad_params_atm_chems[i][0]));    
            atmtyp_2 = distance(atmtyps.begin(), find(atmtyps.begin(), atmtyps.end(), quad_params_atm_chems[i][1]));    
            atmtyp_3 = distance(atmtyps.begin(), find(atmtyps.begin(), atmtyps.end(), quad_params_atm_chems[i][2]));    
            atmtyp_4 = distance(atmtyps.begin(), find(atmtyps.begin(), atmtyps.end(), quad_params_atm_chems[i][3]));    
                        
            // Figure out the corresponding 2-body pair type
            
            pairtyp_1 = atom_int_pair_map[ atmtyp_1*natmtyps + atmtyp_2 ];
            pairtyp_2 = atom_int_pair_map[ atmtyp_1*natmtyps + atmtyp_3 ];
            pairtyp_3 = atom_int_pair_map[ atmtyp_1*natmtyps + atmtyp_4 ];
            pairtyp_4 = atom_int_pair_map[ atmtyp_2*natmtyps + atmtyp_3 ];
            pairtyp_5 = atom_int_pair_map[ atmtyp_2*natmtyps + atmtyp_4 ];
            pairtyp_6 = atom_int_pair_map[ atmtyp_3*natmtyps + atmtyp_4 ];            
    
            // Set the default inner/outer cutoffs to the corresponding 2-body value                    

            chimes_4b_cutoff[i].resize(2);            

            chimes_4b_cutoff[i][0].push_back(chimes_2b_cutoff[pairtyp_1][0]);
            chimes_4b_cutoff[i][0].push_back(chimes_2b_cutoff[pairtyp_2][0]);
            chimes_4b_cutoff[i][0].push_back(chimes_2b_cutoff[pairtyp_3][0]);
            chimes_4b_cutoff[i][0].push_back(chimes_2b_cutoff[pairtyp_4][0]);
            chimes_4b_cutoff[i][0].push_back(chimes_2b_cutoff[pairtyp_5][0]);
            chimes_4b_cutoff[i][0].push_back(chimes_2b_cutoff[pairtyp_6][0]);              
            
            chimes_4b_cutoff[i][1].push_back(chimes_2b_cutoff[pairtyp_1][1]);
            chimes_4b_cutoff[i][1].push_back(chimes_2b_cutoff[pairtyp_2][1]);
            chimes_4b_cutoff[i][1].push_back(chimes_2b_cutoff[pairtyp_3][1]);          
            chimes_4b_cutoff[i][1].push_back(chimes_2b_cutoff[pairtyp_4][1]);
            chimes_4b_cutoff[i][1].push_back(chimes_2b_cutoff[pairtyp_5][1]);
            chimes_4b_cutoff[i][1].push_back(chimes_2b_cutoff[pairtyp_6][1]);                                              
        }
        
        param_file.seekg(0);
        
        int    nentries;
        double cutval;
        
        found_end = false;
        
        while (!found_end)
        {
			line = get_next_line(param_file);
        
			if(line.find("ENDFILE") != string::npos)
                break;                
            
            if(line.find("SPECIAL 4B S_MAXIM:") != string::npos)
            {
                split_line(line, tmp_str_items);
                
                if (rank == 0)
                    cout << "chimesFF: " << "Set the following special 4-body outer cutoffs: " << endl;
                
                if(tmp_str_items[3] == "ALL")
                {
                    cutval = stod(tmp_str_items[4]);
                                        
                    for(int i=0; i<nquads; i++)
                    {                
                        chimes_4b_cutoff[i][1][0] = cutval;
                        chimes_4b_cutoff[i][1][1] = cutval;
                        chimes_4b_cutoff[i][1][2] = cutval;
                        chimes_4b_cutoff[i][1][3] = cutval;
                        chimes_4b_cutoff[i][1][4] = cutval;
                        chimes_4b_cutoff[i][1][5] = cutval;                                                      
                    }
                }
                else
                {
                    nentries = stoi(tmp_str_items[4]);
                    
                    vector<string> pair_name(6);
                    vector<double> cutoffval(6);

                    for(int i=0; i<nentries; i++)
                    {
                        line = get_next_line(param_file);
                        
                        split_line(line, tmp_str_items);
                        
                        tmp_int = atom_idx_quad_map[distance(atom_typ_quad_map.begin(), find(atom_typ_quad_map.begin(), atom_typ_quad_map.end(), tmp_str_items[0]))];

                        pair_name[0] = tmp_str_items[1];
                        pair_name[1] = tmp_str_items[2];
                        pair_name[2] = tmp_str_items[3];
                        pair_name[3] = tmp_str_items[4];
                        pair_name[4] = tmp_str_items[5];
                        pair_name[5] = tmp_str_items[6];
                        
                        cutoffval[0] = stod(tmp_str_items[7 ]);
                        cutoffval[1] = stod(tmp_str_items[8 ]);
                        cutoffval[2] = stod(tmp_str_items[9 ]);
                        cutoffval[3] = stod(tmp_str_items[10]);
                        cutoffval[4] = stod(tmp_str_items[11]);
                        cutoffval[5] = stod(tmp_str_items[12]);
                        
                        vector<bool>   disqualified(6,false);
                        
                        chimes_4b_cutoff[tmp_int][1][ get_index_if(quad_params_pair_typs[tmp_int], pair_name[0], disqualified) ] = cutoffval[0];
                        chimes_4b_cutoff[tmp_int][1][ get_index_if(quad_params_pair_typs[tmp_int], pair_name[1], disqualified) ] = cutoffval[1];
                        chimes_4b_cutoff[tmp_int][1][ get_index_if(quad_params_pair_typs[tmp_int], pair_name[2], disqualified) ] = cutoffval[2];    
                        chimes_4b_cutoff[tmp_int][1][ get_index_if(quad_params_pair_typs[tmp_int], pair_name[3], disqualified) ] = cutoffval[3];
                        chimes_4b_cutoff[tmp_int][1][ get_index_if(quad_params_pair_typs[tmp_int], pair_name[4], disqualified) ] = cutoffval[4];
                        chimes_4b_cutoff[tmp_int][1][ get_index_if(quad_params_pair_typs[tmp_int], pair_name[5], disqualified) ] = cutoffval[5];
					}
                }
                
                for(int i=0; i<nquads; i++)
                {                
                    if (rank == 0)
                        cout << "chimesFF: " << "\t" << i << " " 
                        << chimes_4b_cutoff[i][1][0] << " " 
                        << chimes_4b_cutoff[i][1][1] << " " 
                        << chimes_4b_cutoff[i][1][2] << " " 
                        << chimes_4b_cutoff[i][1][3] << " " 
                        << chimes_4b_cutoff[i][1][4] << " " 
                        << chimes_4b_cutoff[i][1][5] << endl;
                }                
            }

            if(line.find("SPECIAL 4B S_MINIM:") != string::npos)
            {
                split_line(line, tmp_str_items);
                
                if (rank == 0)
                    cout << "chimesFF: " << "Set the following special 4-body inner cutoffs: " << endl;
                
                if(tmp_str_items[3] == "ALL")
                {
                    cutval = stod(tmp_str_items[4]);
                    
                    for(int i=0; i<nquads; i++)
                    {                
                        chimes_4b_cutoff[i][0][0] = cutval;
                        chimes_4b_cutoff[i][0][1] = cutval;
                        chimes_4b_cutoff[i][0][2] = cutval;
                        chimes_4b_cutoff[i][0][3] = cutval;
                        chimes_4b_cutoff[i][0][4] = cutval;
                        chimes_4b_cutoff[i][0][5] = cutval;                         
                    }
                }
                else
                {
                    nentries = stoi(tmp_str_items[4]);
                    
                    vector<string> pair_name(6);
                    vector<double> cutoffval(6);

					for(int i=0; i<nquads; i++)
					{
						chimes_4b_cutoff[i][0][0] = -1.0;
						chimes_4b_cutoff[i][0][1] = -1.0;
						chimes_4b_cutoff[i][0][2] = -1.0;
						chimes_4b_cutoff[i][0][3] = -1.0;
						chimes_4b_cutoff[i][0][4] = -1.0;
						chimes_4b_cutoff[i][0][5] = -1.0;
					}

                    for(int i=0; i<nentries; i++)
                    {
                        line = get_next_line(param_file);
                        
                        split_line(line, tmp_str_items);
                        
                        tmp_int = atom_idx_quad_map[distance(atom_typ_quad_map.begin(), find(atom_typ_quad_map.begin(), atom_typ_quad_map.end(), tmp_str_items[0]))];

                        pair_name[0] = tmp_str_items[1];
                        pair_name[1] = tmp_str_items[2];
                        pair_name[2] = tmp_str_items[3];
                        pair_name[3] = tmp_str_items[4];
                        pair_name[4] = tmp_str_items[5];
                        pair_name[5] = tmp_str_items[6];
                        
                        cutoffval[0] = stod(tmp_str_items[7 ]);
                        cutoffval[1] = stod(tmp_str_items[8 ]);
                        cutoffval[2] = stod(tmp_str_items[9 ]);
                        cutoffval[3] = stod(tmp_str_items[10]);
                        cutoffval[4] = stod(tmp_str_items[11]);
                        cutoffval[5] = stod(tmp_str_items[12]);
                        
                        vector<bool>   disqualified(6,false);
                        
                        chimes_4b_cutoff[tmp_int][0][ get_index_if(quad_params_pair_typs[tmp_int], pair_name[0], disqualified) ] = cutoffval[0];
                        chimes_4b_cutoff[tmp_int][0][ get_index_if(quad_params_pair_typs[tmp_int], pair_name[1], disqualified) ] = cutoffval[1];
                        chimes_4b_cutoff[tmp_int][0][ get_index_if(quad_params_pair_typs[tmp_int], pair_name[2], disqualified) ] = cutoffval[2];    
                        chimes_4b_cutoff[tmp_int][0][ get_index_if(quad_params_pair_typs[tmp_int], pair_name[3], disqualified) ] = cutoffval[3];
                        chimes_4b_cutoff[tmp_int][0][ get_index_if(quad_params_pair_typs[tmp_int], pair_name[4], disqualified) ] = cutoffval[4];
                        chimes_4b_cutoff[tmp_int][0][ get_index_if(quad_params_pair_typs[tmp_int], pair_name[5], disqualified) ] = cutoffval[5];                     
                    }
                }
                
                for(int i=0; i<nquads; i++)
                {                
                    if (rank == 0)
                        cout << "chimesFF: " << "\t" << i << " " 
                        << chimes_4b_cutoff[i][1][0] << " " 
                        << chimes_4b_cutoff[i][1][1] << " " 
                        << chimes_4b_cutoff[i][1][2] << " " 
                        << chimes_4b_cutoff[i][1][3] << " " 
                        << chimes_4b_cutoff[i][1][4] << " " 
                        << chimes_4b_cutoff[i][1][5] << endl;
                }                
            }            
        }

        //set up flat version of powers for 4 body.
        chimes_4b_powers_flat.resize(chimes_4b_powers.size());
        for (int i = 0; i < chimes_4b_powers.size(); i++) {
            for (int j = 0; j < chimes_4b_powers[i].size(); j++) {
                for (int k = 0; k < chimes_4b_powers[i][j].size(); k++) {
                    chimes_4b_powers_flat[i].push_back(chimes_4b_powers[i][j][k]);
                }
            }
        }    
    }
    
    param_file.close();    
}

void chimesFF::set_polys_out_of_range(vector<double> &Tn, vector<double> &Tnd, double dx, double x, int poly_order, double inner_cutoff, double exprlen, double dx_dr)
{
    //  Sets the value of the Chebyshev polynomials (Tn) and their derivatives (Tnd) when dx is < inner_cutoff.
    //  Tnd is the derivative with respect to the interatomic distance, not the transformed distance (x).
    //	
    //  The derivative Tnd is continuously set to zero inside the cutoff.
    //  The exponential smoothing distance is set to ChimesFF::inner_smooth_distance.
    //  x, exprlen, and dx_dr are evaluated at the inner cutoff.
    //	
    //  dx is the pair distance, which is assumed to be less than inner_cutoff.
    Tn[0] = 1.0;
    Tn[1] = x;

    // Start the derivative setup. Set the first two 1st-kind Cheby's equal to the first two of the 2nd-kind

    Tnd[0] = 1.0;
    Tnd[1] = 2.0 * x;
    
    // Use recursion to set up the higher n-value Tn and Tnd's
    for ( int i = 2; i <= poly_order; i++ ) 
    {
        Tn[i]  = 2.0 * x *  Tn[i-1] -  Tn[i-2];
        Tnd[i] = 2.0 * x * Tnd[i-1] - Tnd[i-2];
    }
    
    // Now multiply by n to convert Tnd's to actual derivatives of Tn

    for ( int i = poly_order; i >= 1; i-- ) 
        Tnd[i] = i * dx_dr * Tnd[i-1];

    Tnd[0] = 0.0;

    // Exponential damping of the derivative.
    double damp_fac = exp( (dx-inner_cutoff) / inner_smooth_distance ) ;
      
    // Correct Tn outside of the range using the damping factor.
    for ( int i = 0 ; i <= poly_order ; i++ )
    {
        Tn[i]  += inner_smooth_distance * (damp_fac-1.0)  * Tnd[i] ;
        Tnd[i] *= damp_fac ;
    }     
}

inline double chimesFF::dr2_3B(const double *dr2, int i, int j, int k, int l)
{
    // Access the dr2 distance tensor for a 3 body interaction.
    return(dr2[i*CHDIM*3*CHDIM + j*3*CHDIM + k*CHDIM + l]) ;
}

inline double chimesFF::dr2_4B(const double *dr2, int i, int j, int k, int l)
{
    // Access the dr2 distance tensor for a 4 body interaction.
    return(dr2[i*CHDIM*6*CHDIM + j*6*CHDIM + k*CHDIM + l]) ;
}

inline void chimesFF::init_distance_tensor(double *dr2, const vector<double> & dr, int npairs)
{
    for ( int i = 0 ; i < npairs ; i++ )
        for ( int j = 0 ; j < CHDIM ; j++ )
            for ( int k = 0 ; k < npairs ; k++ )
                for ( int l = 0 ; l < CHDIM ; l++ )
                    dr2[i* CHDIM * npairs * CHDIM + j * npairs * CHDIM + k * CHDIM + l] = dr[i*CHDIM+j] * dr[k*CHDIM+l] ;
}

void chimesFF::compute_1B(const int typ_idx, double & energy )
{
    // Compute 1b (input: a single atom type index... outputs (updates) energy

    energy += energy_offsets[typ_idx];
}



// Overload for calls from LAMMPS                 
void chimesFF::compute_2B(const double dx, const vector<double> & dr, const vector<int> typ_idxs, vector<double> & force, vector<double> & stress, double & energy, chimes2BTmp &tmp)
{              
    double dummy_force_scalar;
    compute_2B(dx, dr, typ_idxs, force, stress, energy, tmp, dummy_force_scalar);                                                               
}
void chimesFF::compute_2B(const double dx, const vector<double> & dr, const vector<int> typ_idxs, vector<double> & force, vector<double> & stress, double & energy, chimes2BTmp &tmp, double & force_scalar_in)
{
    // Compute 2b (input: 2 atoms or distances, corresponding types... outputs (updates) force, acceleration, energy, stress
    //
    // Input parameters:
    //
    // dx: Scalar (pair distance)
    // dr: 1d-Array (pair distance: [x, y, and z-component]) 
    // Force: [natoms in interaction set][x,y, and z-component] *note
    // Stress [sxx, sxy, sxz, syy, syz, szz]  *note
    // Energy: Scalar; energy for interaction set
    // Tmp: Temporary storage for calculation.
    
    // Assumes atom indices start from zero
    // Assumes distances are atom_2 - atom_1
    //
    // *note: force is a packed array of coordinates.

    int     pair_idx;    
    double  fcut;
    double  fcutderiv;

    // tmp.resize(poly_orders[0]+1) ;
    
    // Use references for readability.
    vector<double> &Tn = tmp.Tn ;
    vector<double> &Tnd = tmp.Tnd ;
    
    pair_idx = atom_int_pair_map[ typ_idxs[0]*natmtyps + typ_idxs[1] ];

    if (dx >= chimes_2b_cutoff[pair_idx][1])
        return;    

    set_cheby_polys(Tn, Tnd, dx, pair_idx, chimes_2b_cutoff[pair_idx][0], chimes_2b_cutoff[pair_idx][1], 0);
    
    get_fcut(dx, chimes_2b_cutoff[pair_idx][1], fcut, fcutderiv);

    double dx_inv = ( dx > 0.0 ) ? 1.0 / dx : 1e20 ;

    // need to create some C-style arrays/pointers to feed into GPU
    // and receive results.

    // Host and GPU pointers - these all need to be freed
    // at the end.

    // pointers to get force and stress data back at the end of calculations
    double *host_stress, *host_force;
    // device/GPU pointers for the items being transferred over.
    double *device_chimes_params, *device_Tn, *device_Tnd, *device_force, *device_stress;
    int *device_chimes_pows;
    // GPU memory allocation

    // Start with copying dr to constant memory - it doesnt
    // change for the duration of the call to the GPU and it 
    // is used by every single thread.
   
    cudaMemcpyToSymbol(dr_gpu, dr.data(), CHDIM*sizeof(double));
    

    //set energy to whatever the input value should be
    //ok this finally transfers properly but I think if I want
    //to use this for a multi threaded application I need to switch this to
    //https://stackoverflow.com/a/2620144
    cudaMemcpyToSymbol(gpu_energy, &energy, sizeof(double));

    // allocate GPU memory for device pointers

    cudaMalloc(&device_chimes_params, chimes_2b_params[pair_idx].size() * sizeof(double));
    cudaMalloc(&device_Tn, Tn.size() * sizeof(double));
    cudaMalloc(&device_Tnd, Tnd.size() * sizeof(double));
    cudaMalloc(&device_force, force.size() * sizeof(double));
    cudaMalloc(&device_stress, stress.size() * sizeof(double));
    cudaMalloc(&device_chimes_pows, chimes_2b_pows[pair_idx].size() * sizeof(int));

    // Transfer memory from host to GPU.

    cudaMemcpy(device_chimes_params, chimes_2b_params[pair_idx].data(), chimes_2b_params[pair_idx].size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_Tn, Tn.data(), Tn.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_Tnd, Tnd.data(), Tnd.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_force, force.data(), force.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_stress, stress.data(), stress.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_chimes_pows, chimes_2b_pows[pair_idx].data(), chimes_2b_pows[pair_idx].size() * sizeof(int), cudaMemcpyHostToDevice);

    //grid and block dimension setup - for controlling threads on the gpu.

    int blockSize = 512;  // start with this - can consider other options depending on some of the other
    // strategies I end up using.
    dim3 dimBlock(blockSize, 1, 1);
    dim3 dimGrid(ceil((double)ncoeffs_2b[pair_idx]/blockSize), 1 , 1);

    // Starting the kernel - houston, we are cleared for liftoff
    compute2B_helper<<<dimGrid, dimBlock>>>(ncoeffs_2b[pair_idx], fcut, fcutderiv, dx_inv, device_chimes_params,
    device_chimes_pows, device_Tn, device_Tnd, device_force, device_stress);
    cudaDeviceSynchronize();  // do not proceed on the CPU until the GPU has finished its calculations.
    // kernel completion - update host memory with results
   
    //allocate memory for our host pointers to receive the newly updated data
    //TODO - I'm not convinced that this is the best way to handle this, maybe return to this
    //in the future when I'm wiser and I'm older

    host_stress = (double *)malloc(stress.size() * sizeof(double));
    host_force = (double *)malloc(force.size() * sizeof(double));

    //transfer new force and stress values from the GPU to the CPU
    //perhaps try putting stress.data() in the recieving spot and seeing what happens?  I've never
    //really tried that before.
    cudaMemcpy(host_stress, device_stress, stress.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_force, device_force, force.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&energy, gpu_energy, sizeof(double));
    

    //update the storage vectors on the host - replace old data with new data from GPU calculations.
    stress = std::vector<double>(host_stress, host_stress + stress.size());
    force = std::vector<double>(host_force, host_force + force.size());


    // just need to solve getting energy - 
    // https://stackoverflow.com/questions/2619296/how-to-return-a-single-variable-from-a-cuda-kernel-function
    // answers in this thread may be helpful

    // free memory used - be very careful not to cause
    // memory leaks since this will almost certainly be called
    // multiple times by host programs - memory leaks
    // will very rapidly cause out of memory errors.

    //clean up time!  Free takes care of malloc memory, cudaFree takes care
    //of cudaMalloc memory on the GPU.
    free(host_stress);
    free(host_force);
    cudaFree(device_chimes_params);
    cudaFree(device_Tn);
    cudaFree(device_Tnd);
    cudaFree(device_chimes_pows);
    cudaFree(device_force);
    cudaFree(device_stress);
    
    /*
    for(int coeffs=0; coeffs<ncoeffs_2b[pair_idx]; coeffs++)
    {
        double coeff_val = [coeffs];        
        
        energy += coeff_val * fcut * Tn[ chimes_2b_pows[pair_idx][coeffs]+1 ];
                                                
        double deriv = fcut * Tnd[ chimes_2b_pows[pair_idx][coeffs]+1 ]  + fcutderiv * Tn[ chimes_2b_pows[pair_idx][coeffs]+1 ];    

        double force_scalar = coeff_val * deriv * dx_inv ; 

        force[0*CHDIM+0] += force_scalar * dr[0];
        force[0*CHDIM+1] += force_scalar * dr[1];
        force[0*CHDIM+2] += force_scalar * dr[2];
        
        force[1*CHDIM+0] -= force_scalar * dr[0];
        force[1*CHDIM+1] -= force_scalar * dr[1];
        force[1*CHDIM+2] -= force_scalar * dr[2];
        
        // xx xy xz yy yz zz
        // 0  1  2  3  4  5
        
        // xx xy xz yx yy yz zx zy zz
        // 0  1  2  3  4  5  6  7  8
        // *           *           *
        
        stress[0] -= force_scalar * dr[0] * dr[0]; // xx tensor component
        stress[1] -= force_scalar * dr[0] * dr[1]; // xy tensor component 
        stress[2] -= force_scalar * dr[0] * dr[2]; // xz tensor component
        stress[3] -= force_scalar * dr[1] * dr[1]; // yy tensor component
        stress[4] -= force_scalar * dr[1] * dr[2]; // yz tensor component
        stress[5] -= force_scalar * dr[2] * dr[2]; // zz tensor component
            
    } */


    double E_penalty = 0.0 ;
    double force_scalar ;
    get_penalty(dx, pair_idx, E_penalty , force_scalar); 

    if ( E_penalty > 0.0 ) 
    {
        energy += E_penalty;

        force_scalar /= dx ;

        // Note: force_scalar is negative (LEF) 7/30/21.
        force[0*CHDIM+0] += force_scalar * dr[0];
        force[0*CHDIM+1] += force_scalar * dr[1];
        force[0*CHDIM+2] += force_scalar * dr[2];
        
        force[1*CHDIM+0] -= force_scalar * dr[0];
        force[1*CHDIM+1] -= force_scalar * dr[1];
        force[1*CHDIM+2] -= force_scalar * dr[2];

        // Update stress according to penalty force. (LEF) 07/30/21
        stress[0] -= force_scalar  * dr[0] * dr[0]; // xx tensor component
        stress[1] -= force_scalar  * dr[0] * dr[1]; // xy tensor component 
        stress[2] -= force_scalar  * dr[0] * dr[2]; // xz tensor component
        stress[3] -= force_scalar  * dr[1] * dr[1]; // yy tensor component
        stress[4] -= force_scalar  * dr[1] * dr[2]; // yz tensor component
        stress[5] -= force_scalar  * dr[2] * dr[2]; // zz tensor component

    }
    
    force_scalar_in = force_scalar;
}

// Overload for calls from LAMMPS  
void chimesFF::compute_3B(const vector<double> & dx, const vector<double> & dr, const vector<int> & typ_idxs, vector<double> & force, vector<double> & stress, double & energy, chimes3BTmp &tmp)
{
	vector<double> dummy_force_scalar(3);
	compute_3B(dx, dr, typ_idxs, force, stress, energy, tmp, dummy_force_scalar);
}
void chimesFF::compute_3B(const vector<double> & dx, const vector<double> & dr, const vector<int> & typ_idxs, vector<double> & force, vector<double> & stress, double & energy, chimes3BTmp &tmp, vector<double> & force_scalar_in)
{
    // Compute 3b (input: 3 atoms or distances, corresponding types... outputs (updates) force, acceleration, energy, stress
    //
    // Input parameters:
    //
    // dx_ij: Scalar (pair distance)
    // dr_ij: 1d-Array (pair distance: [x, y, and z-component])
    // Force: [natoms in interaction set][x,y, and z-component] *note
    // Stress [sxx, sxy, sxz, syy, syz, szz] 
    // Energy: Scalar; energy for interaction set
    // Tmp: Temporary storage for 3-body interactions.
    
    // Assumes atom indices start from zero
    // Assumes distances are atom_2 - atom_1
    //
    // *note: force and dr are packed vectors of coordinates.
    
    const int natoms = 3;                   // Number of atoms in an interaction set
    const int npairs = natoms*(natoms-1)/2; // Number of pairs in an interaction set
    
    // tmp.resize(poly_orders[1]) ;
    
    vector<double> &Tn_ij  = tmp.Tn_ij ;
    vector<double> &Tn_ik  = tmp.Tn_ik ;
    vector<double> &Tn_jk  = tmp.Tn_jk ;   // The Chebyshev polymonials
    vector<double> &Tnd_ij = tmp.Tnd_ij ;
    vector<double> &Tnd_ik = tmp.Tnd_ik ;
    vector<double> &Tnd_jk = tmp.Tnd_jk ;  // The Chebyshev polymonial derivatives

    //I think I want to store all of these device pointers in a struct of some sort, especially for four body, to cut
    //down on the number of arguments I have to pass, especially since there is a limit of 32 pointers/256 bytes I think.

    // Avoid allocating std::vector quantities.  Heap memory allocation is slow on the GPU.
    // fixed-length C arrays are allocated on the stack.
    double fcut[npairs] ;
    double fcutderiv[npairs] ;
    //double deriv[npairs];

    

#if DEBUG == 1  
    if ( dr.size() != 9 )
    {
        cout << "Error: dr should have length = 9.  Current length = " << dr.size() << endl ;
        exit(0) ;
    }
#endif


    int type_idx =  typ_idxs[0]*natmtyps*natmtyps + typ_idxs[1]*natmtyps + typ_idxs[2] ;
    int tripidx = atom_int_trip_map[type_idx];

    if(tripidx < 0)    // Skipping an excluded interaction
        return;
    
    // Check whether cutoffs are within allowed ranges
    vector<int> & mapped_pair_idx = pair_int_trip_map[type_idx] ;
        
    if (dx[0] >= chimes_3b_cutoff[ tripidx ][1][mapped_pair_idx[0]])    // ij
        return;    
    if (dx[1] >= chimes_3b_cutoff[ tripidx ][1][mapped_pair_idx[1]])    // ik
        return;    
    if (dx[2] >= chimes_3b_cutoff[ tripidx ][1][mapped_pair_idx[2]])    // jk
        return;    
    
    // At this point, all distances are within allowed ranges. We can now proceed to the force/stress/energy calculation

#ifdef USE_DISTANCE_TENSOR  
    // Tensor product of displacement vectors.
    double dr2[CHDIM*CHDIM*npairs*npairs] ;
    init_distance_tensor(dr2, dr, npairs) ;
#endif

    // Set up the polynomials

    set_cheby_polys(Tn_ij, Tnd_ij, dx[0], atom_int_pair_map[ typ_idxs[0]*natmtyps + typ_idxs[1] ], chimes_3b_cutoff[tripidx][0][mapped_pair_idx[0]], chimes_3b_cutoff[tripidx][1][mapped_pair_idx[0]], 1);
    set_cheby_polys(Tn_ik, Tnd_ik, dx[1], atom_int_pair_map[ typ_idxs[0]*natmtyps + typ_idxs[2] ], chimes_3b_cutoff[tripidx][0][mapped_pair_idx[1]], chimes_3b_cutoff[tripidx][1][mapped_pair_idx[1]], 1);
    set_cheby_polys(Tn_jk, Tnd_jk, dx[2], atom_int_pair_map[ typ_idxs[1]*natmtyps + typ_idxs[2] ], chimes_3b_cutoff[tripidx][0][mapped_pair_idx[2]], chimes_3b_cutoff[tripidx][1][mapped_pair_idx[2]], 1);
    
    
    // Set up the smoothing functions
        
    get_fcut(dx[0], chimes_3b_cutoff[tripidx][1][mapped_pair_idx[0]], fcut[0], fcutderiv[0]);
    get_fcut(dx[1], chimes_3b_cutoff[tripidx][1][mapped_pair_idx[1]], fcut[1], fcutderiv[1]);
    get_fcut(dx[2], chimes_3b_cutoff[tripidx][1][mapped_pair_idx[2]], fcut[2], fcutderiv[2]);
    double fcut_all =  fcut[0] * fcut[1] * fcut[2] ;

    // Product of 2 fcuts divided by dx. Index i = product of all fcuts except i.
    double fcut_2[npairs] ;
    fcut_2[0] = fcut[1] * fcut[2] / dx[0] ;
    fcut_2[1] = fcut[0] * fcut[2] / dx[1] ;
    fcut_2[2] = fcut[0] * fcut[1] / dx[2] ;

    // Start the force/stress/energy calculation
    //double coeff;
    //int powers[npairs] ;
    //double force_scalar[npairs] ;

    // GPU IMPLEMENTATION

    // Let the book keeping begin :(
    
    // What needs to go to the GPU?
    // ncoeffs_3b[tripidx] can be passed in the kernel call.
    // chimes_3b_params[tripidx] needs memcpy
    // chimes_3b_powers[tripidx] needs memcpy and HAS TO BE FLATTENED :(
    // flattening this every time may be expensive?  Talk with Professor Lindsey about where else
    // the chimes_3b_powers is used in the code and if I can set it up to be flat initally.
    // energy has to be either the __device__ energy that I have been using or it
    // just needs to be an allocated double, which is what I will have to do if I want it to be MPI capable.
    // fcut_all can be passed in the kernel call.
    // fcut array should be sent to constant memory
    // fcutderiv array should also be sent to constant memory.
    // fcut2 should also be sent to constant memory.
    // Wait - I'm not sure how constant memory interacts with MPI calls.
    // I think it may be fine since at least on the CPU each MPI thing is a
    // copy and has their own copies of the variables/their own context?
    // thats a problem for later, its fixable if I am wrong.
    // dr and dr2_3b should both be sent to constant memory.


    // need to create some C-style arrays/pointers to feed into GPU
    // and receive results.
    // Host and GPU pointers - these all need to be freed
    // at the end.

    // host pointers to update stress and force at the end.
    // double *host_stress, *host_force;
    // device/GPU pointers for the items being transferred over.
    //poly_pointers_3b *device_polys; // I think the whole struct now has to
    // be copied over to memory, increasing memory accesses?
    // Maybe more arguments is the answer?
    // If I do it this way, I pass a pointer to a struct.
    // I then have a memory access to the struct to get the pointers in the struct.
    // Or I can pass all of those pointers to the kernel call (assuming they fit)
    // and they should end up as part of the registers for each thread.
    // However, that's a lot of arguments... which is usually considered
    // to be poor practice for other languages at least.  Hmm
    // Its not that big so I think I start with trying to pass by value
    // cause looking at the profiling from the two body version the
    // cudaMalloc and free calls are very expensive.

    double *device_chimes_params, *device_force, *device_stress;
    int *device_chimes_pows; // need this but would like it to be flattened
    // before the compute call if possible.
    
    //device_polys = (poly_pointers_3b *)malloc(sizeof(poly_pointers_3b));
    struct poly_pointers_3b device_polys;
    // Allocate GPU memory for device pointers, including the ones in device_polys.
    // Ouch this is a lot of malloc calls.
    // Could put all of these in a struct and then copy the struct to the gpu instead
    // of sending a struct with pointers to each one - that would cut down on cudaMalloc calls,
    // but will mean alot of malloc calls on the cpu - I have to think those would be faster
    // though?
    // Lets go have lunch and think about this, I dont like making decisions
    // while hungry.

    cudaMalloc(&device_chimes_params, chimes_3b_params[tripidx].size() * sizeof(double));
    cudaMalloc(&device_force, force.size() * sizeof(double));
    cudaMalloc(&device_stress, stress.size() * sizeof(double));
    cudaMalloc(&device_chimes_pows, chimes_3b_powers_flat[tripidx].size() * sizeof(double));
    // mallocs all done except for powers, which I want to flatten first.

    // Tn and Tnd mallocs
    cudaMalloc(&device_polys.Tn_ij, Tn_ij.size() * sizeof(double));
    cudaMalloc(&device_polys.Tn_ik, Tn_ik.size() * sizeof(double));
    cudaMalloc(&device_polys.Tn_jk, Tn_jk.size() * sizeof(double));
    cudaMalloc(&device_polys.Tnd_ij, Tnd_ij.size() * sizeof(double));
    cudaMalloc(&device_polys.Tnd_ik, Tnd_ik.size() * sizeof(double));
    cudaMalloc(&device_polys.Tnd_jk, Tnd_jk.size() * sizeof(double));

    // Begin transferring memory over to the GPU.

    // Start off with the stuff going over to constant memory.
    if (cudaMemcpyToSymbol(fcut_3b, fcut, npairs * sizeof(double)) != cudaSuccess) {
        std::cout << "An error occured at line 1810" << std::endl;
        exit(1);
    }

    if (cudaMemcpyToSymbol(fcutderiv_3b, fcutderiv, npairs * sizeof(double)) != cudaSuccess) {
        std::cout << "An error occured at line 1815" << std::endl;
        exit(1);
    }

    if (cudaMemcpyToSymbol(fcut2_3b, fcut_2, npairs * sizeof(double)) != cudaSuccess) {
        std::cout << "An error occured at line 1820" << std::endl;
        exit(1);
    }
    
    if (cudaMemcpyToSymbol(dr_3b, dr.data(), dr.size() * sizeof(double)) != cudaSuccess) {
        std::cout << "An error occured at line 1825" << std::endl;
        exit(1);
    }

    if (cudaMemcpyToSymbol(pair_idx_3b, mapped_pair_idx.data(), mapped_pair_idx.size() * sizeof(int)) != cudaSuccess) {
        std::cout << "An error occured at line 1830" << std::endl;
        exit(1);
    }
    
    
    
    
    #ifdef USE_DISTANCE_TENSOR
        if (cudaMemcpyToSymbol(dr2_3b, dr2, sizeof(dr2_3b)) != cudaSuccess) {
            std::cout << "An error occured at line 1842" << std::endl;
            exit(1);
        }
        
    #endif
    if (cudaMemcpyToSymbol(gpu_energy, &energy, sizeof(double)) != cudaSuccess) {
        std::cout << "An error occured at line 1848" << std::endl;
        exit(1);
    }
    

    // constant memory stuff is completed, time for dynamically allocated stuff

    cudaMemcpy(device_chimes_params, chimes_3b_params[tripidx].data(), chimes_3b_params[tripidx].size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_force, force.data(), force.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_stress, stress.data(), stress.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_chimes_pows, chimes_3b_powers_flat[tripidx].data(), chimes_3b_powers_flat[tripidx].size() * sizeof(int), cudaMemcpyHostToDevice);

    //memcpy for the Tn and Tnds
    cudaMemcpy(device_polys.Tn_ij, Tn_ij.data(), Tn_ij.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_polys.Tn_ik, Tn_ik.data(), Tn_ik.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_polys.Tn_jk, Tn_jk.data(), Tn_jk.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_polys.Tnd_ij, Tnd_ij.data(), Tnd_ij.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_polys.Tnd_ik, Tnd_ik.data(), Tnd_ik.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_polys.Tnd_jk, Tnd_jk.data(), Tnd_jk.size() * sizeof(double), cudaMemcpyHostToDevice);


    // Memory transfer should now be done - time to do the grid and block dimension setup
    int blockSize = 512;
    // Can start with this
    dim3 dimBlock(blockSize, 1, 1);
    dim3 dimGrid(ceil((double)ncoeffs_3b[tripidx]/blockSize), 1 , 1);

    // Start the kernel.
    compute3b_helper<<<dimGrid, dimBlock>>>(ncoeffs_3b[tripidx], fcut_all, device_chimes_params, device_chimes_pows, device_force, device_stress, device_polys);
    cudaDeviceSynchronize();

    // kernel is now finished at this point - time to get the results we
    // need from the GPU.

    cudaMemcpyFromSymbol(&energy, gpu_energy, sizeof(double));
    cudaMemcpy(stress.data(), device_stress, stress.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(force.data(), device_force, force.size() * sizeof(double), cudaMemcpyDeviceToHost);
    // testing this.

    // memory clean up
    cudaFree(device_chimes_params);
    cudaFree(device_chimes_pows);
    cudaFree(device_force);
    cudaFree(device_stress);
    cudaFree(device_polys.Tn_ij);
    cudaFree(device_polys.Tn_ik);
    cudaFree(device_polys.Tn_jk);
    cudaFree(device_polys.Tnd_ij);
    cudaFree(device_polys.Tnd_ik);
    cudaFree(device_polys.Tnd_jk);
    //free(device_polys);
    
    /*
    for(int coeffs=0; coeffs<ncoeffs_3b[tripidx]; coeffs++)
    {
        coeff = chimes_3b_params[tripidx][coeffs];
        
        powers[0] = chimes_3b_powers[tripidx][coeffs][mapped_pair_idx[0]];
        powers[1] = chimes_3b_powers[tripidx][coeffs][mapped_pair_idx[1]];
        powers[2] = chimes_3b_powers[tripidx][coeffs][mapped_pair_idx[2]];

        // For powers - the mapped pair will always be a size of 3,
        // so I can flatten by that dimension and then index in with coeffs * 3 + mapped_pair_index

        //GPU implementation notes - since this is small and all of the accesses are constant,
        //I think I can create a per thread array and the compiler will put them on registers
        //which will be nice and efficient - however, will need to use memory profiler to make
        //sure this is actually happening, becasue if it isn't I want to switch to a shared memory
        //scheme.
        
        energy += coeff * fcut_all * Tn_ij[ powers[0] ] * Tn_ik[ powers[1] ] * Tn_jk[ powers[2] ];    

        deriv[0] = fcut[0] * Tnd_ij[ powers[0] ] + fcutderiv[0] * Tn_ij[ powers[0] ];
        deriv[1] = fcut[1] * Tnd_ik[ powers[1] ] + fcutderiv[1] * Tn_ik[ powers[1] ];
        deriv[2] = fcut[2] * Tnd_jk[ powers[2] ] + fcutderiv[2] * Tn_jk[ powers[2] ];

        // deriv will need to be created on the GPU in the same way as powers, which ever way ends up being more efficient.


        force_scalar[0]  = coeff * deriv[0] * fcut_2[0] * Tn_ik[powers[1]]  * Tn_jk[powers[2]] ;
        force_scalar[1]  = coeff * deriv[1] * fcut_2[1] * Tn_ij[powers[0]]  * Tn_jk[powers[2]] ;
        force_scalar[2]  = coeff * deriv[2] * fcut_2[2] * Tn_ij[powers[0]]  * Tn_ik[powers[1]] ;

        // force scalar will need to be created on the GPU in the same way as powers and deriv, which ever way ends up being more efficient.
        
        // Accumulate forces/stresses on/from the ij pair
        
        force[0*CHDIM+0] += force_scalar[0] * dr[0*CHDIM+0];
        force[0*CHDIM+1] += force_scalar[0] * dr[0*CHDIM+1];
        force[0*CHDIM+2] += force_scalar[0] * dr[0*CHDIM+2];

        force[1*CHDIM+0] -= force_scalar[0] * dr[0*CHDIM+0];
        force[1*CHDIM+1] -= force_scalar[0] * dr[0*CHDIM+1];
        force[1*CHDIM+2] -= force_scalar[0] * dr[0*CHDIM+2];   

        // dr2_3B looks like a function call, but the optimizer should remove it entirely.
#ifdef USE_DISTANCE_TENSOR
        // New stress code.
        stress[0] -= force_scalar[0]  * dr2_3B(dr2,0,0,0,0); // xx tensor component
        stress[1] -= force_scalar[0]  * dr2_3B(dr2,0,0,0,1); // xy tensor component
        stress[2] -= force_scalar[0]  * dr2_3B(dr2,0,0,0,2); // xz tensor component
        stress[3] -= force_scalar[0]  * dr2_3B(dr2,0,1,0,1); // yy tensor component
        stress[4] -= force_scalar[0]  * dr2_3B(dr2,0,1,0,2); // yz tensor component
        stress[5] -= force_scalar[0]  * dr2_3B(dr2,0,2,0,2); // zz tensor component
        
#else
        stress[0] -= force_scalar[0]  * dr[0*CHDIM+0] * dr[0*CHDIM+0]; // xx tensor component
        stress[1] -= force_scalar[0]  * dr[0*CHDIM+0] * dr[0*CHDIM+1]; // xy tensor component
        stress[2] -= force_scalar[0]  * dr[0*CHDIM+0] * dr[0*CHDIM+2]; // xz tensor component
        stress[3] -= force_scalar[0]  * dr[0*CHDIM+1] * dr[0*CHDIM+1]; // yy tensor component
        stress[4] -= force_scalar[0]  * dr[0*CHDIM+1] * dr[0*CHDIM+2]; // yz tensor component
        stress[5] -= force_scalar[0]  * dr[0*CHDIM+2] * dr[0*CHDIM+2]; // zz tensor component
#endif        
        // Accumulate forces/stresses on/from the ik pair
        
        force[0*CHDIM+0] += force_scalar[1] * dr[1*CHDIM+0];
        force[0*CHDIM+1] += force_scalar[1] * dr[1*CHDIM+1];
        force[0*CHDIM+2] += force_scalar[1] * dr[1*CHDIM+2];

        force[2*CHDIM+0] -= force_scalar[1] * dr[1*CHDIM+0];
        force[2*CHDIM+1] -= force_scalar[1] * dr[1*CHDIM+1];
        force[2*CHDIM+2] -= force_scalar[1] * dr[1*CHDIM+2];   

#ifdef USE_DISTANCE_TENSOR
        stress[0] -= force_scalar[1]  * dr2_3B(dr2,1,0,1,0); // xx tensor component
        stress[1] -= force_scalar[1]  * dr2_3B(dr2,1,0,1,1); // xy tensor component
        stress[2] -= force_scalar[1]  * dr2_3B(dr2,1,0,1,2); // xz tensor component
        stress[3] -= force_scalar[1]  * dr2_3B(dr2,1,1,1,1); // yy tensor component
        stress[4] -= force_scalar[1]  * dr2_3B(dr2,1,1,1,2); // yz tensor component
        stress[5] -= force_scalar[1]  * dr2_3B(dr2,1,2,1,2); // zz tensor component
#else
        stress[0] -= force_scalar[1]  * dr[1*CHDIM+0] * dr[1*CHDIM+0]; // xx tensor component
        stress[1] -= force_scalar[1]  * dr[1*CHDIM+0] * dr[1*CHDIM+1]; // xy tensor component
        stress[2] -= force_scalar[1]  * dr[1*CHDIM+0] * dr[1*CHDIM+2]; // xz tensor component
        stress[3] -= force_scalar[1]  * dr[1*CHDIM+1] * dr[1*CHDIM+1]; // yy tensor component
        stress[4] -= force_scalar[1]  * dr[1*CHDIM+1] * dr[1*CHDIM+2]; // yz tensor component
        stress[5] -= force_scalar[1]  * dr[1*CHDIM+2] * dr[1*CHDIM+2]; // zz tensor component
#endif
        
        // Accumulate forces/stresses on/from the jk pair
        
        force[1*CHDIM+0] += force_scalar[2] * dr[2*CHDIM+0];
        force[1*CHDIM+1] += force_scalar[2] * dr[2*CHDIM+1];
        force[1*CHDIM+2] += force_scalar[2] * dr[2*CHDIM+2];

        force[2*CHDIM+0] -= force_scalar[2] * dr[2*CHDIM+0];
        force[2*CHDIM+1] -= force_scalar[2] * dr[2*CHDIM+1];
        force[2*CHDIM+2] -= force_scalar[2] * dr[2*CHDIM+2];   

#ifdef USE_DISTANCE_TENSOR
        stress[0] -= force_scalar[2]  * dr2_3B(dr2,2,0,2,0); // xx tensor component
        stress[1] -= force_scalar[2]  * dr2_3B(dr2,2,0,2,1); // xy tensor component
        stress[2] -= force_scalar[2]  * dr2_3B(dr2,2,0,2,2); // xz tensor component
        stress[3] -= force_scalar[2]  * dr2_3B(dr2,2,1,2,1); // yy tensor component
        stress[4] -= force_scalar[2]  * dr2_3B(dr2,2,1,2,2); // yz tensor component
        stress[5] -= force_scalar[2]  * dr2_3B(dr2,2,2,2,2); // zz tensor component
#else        
        stress[0] -= force_scalar[2]  * dr[2*CHDIM+0] * dr[2*CHDIM+0]; // xx tensor component
        stress[1] -= force_scalar[2]  * dr[2*CHDIM+0] * dr[2*CHDIM+1]; // xy tensor component
        stress[2] -= force_scalar[2]  * dr[2*CHDIM+0] * dr[2*CHDIM+2]; // xz tensor component
        stress[3] -= force_scalar[2]  * dr[2*CHDIM+1] * dr[2*CHDIM+1]; // yy tensor component
        stress[4] -= force_scalar[2]  * dr[2*CHDIM+1] * dr[2*CHDIM+2]; // yz tensor component
        stress[5] -= force_scalar[2]  * dr[2*CHDIM+2] * dr[2*CHDIM+2]; // zz tensor component
#endif        
    } */

    //force_scalar_in[0] = force_scalar[0];
    //force_scalar_in[1] = force_scalar[1];
    //force_scalar_in[2] = force_scalar[2];

    // the last iteration needs to be removed from
    // the gpu so force scalars can be returned.
    // unclear from meeting with professor if this 
    // actually works correctly or if anyone is using this,
    // so may end up getting removed.

    force_scalar_in[0] = 1;
    force_scalar_in[1] = 1;
    force_scalar_in[2] = 1;

    // what actually is the physical meaning behind returning the last force scalars to force scalar in?

    return;    
}

void chimesFF::compute_4B(const vector<double> & dx, const vector<double> & dr, const vector<int> & typ_idxs, vector<double> & force, vector<double> & stress, double & energy, chimes4BTmp &tmp)
{              
        vector<double> dummy_force_scalar(6);
        compute_4B(dx, dr, typ_idxs, force, stress, energy, tmp, dummy_force_scalar);                                                               
}
void chimesFF::compute_4B(const vector<double> & dx, const vector<double> & dr, const vector<int> & typ_idxs, vector<double> & force, vector<double> & stress, double & energy, chimes4BTmp &tmp, vector<double> & force_scalar_in)
{
    // Compute 4b (input: 4 atoms or distances, corresponding types... outputs (updates) force, acceleration, energy, stress
    //
    // Input parameters:
    //
    // dx_ij: Scalar (pair distance)
    // dr_ij: 1d-Array (pair distance: [x, y, and z-component])
    // Force: [natoms in interaction set][x,y, and z-component] *note
    // Stress [sxx, sxy, sxz, syy, syz, szz]
    // Energy: Scalar; energy for interaction set
    // Tmp: Structure containing temporary data.
    // Assumes atom indices start from zero
    // Assumes distances are atom_2 - atom_1
    //
    // *note: force and dr are packed vectors of coordinates.

    const int natoms = 4;                     // Number of atoms in an interaction set
    const int npairs = natoms*(natoms-1)/2;    // Number of pairs in an interaction set


    double fcut[npairs] ;
    double fcutderiv[npairs] ;
    //double deriv[npairs] ;
    

#if DEBUG == 1  
    if ( force.size() != CHDIM * natoms ) {
        cout << "Error: force vector had incorrect dimension of " << force.size() << endl ;
        exit(1) ;
    }
#endif      

    vector<double> &Tn_ij   = tmp.Tn_ij ;
    vector<double> &Tn_ik   = tmp.Tn_ik ;
    vector<double> &Tn_il   = tmp.Tn_il ;
    vector<double> &Tn_jk   = tmp.Tn_jk ;
    vector<double> &Tn_jl   = tmp.Tn_jl ;
    vector<double> &Tn_kl   = tmp.Tn_kl ;        
                                          
    vector<double> &Tnd_ij  = tmp.Tnd_ij ;
    vector<double> &Tnd_ik  = tmp.Tnd_ik ;
    vector<double> &Tnd_il  = tmp.Tnd_il ;  
    vector<double> &Tnd_jk  = tmp.Tnd_jk ;
    vector<double> &Tnd_jl  = tmp.Tnd_jl ;
    vector<double> &Tnd_kl  = tmp.Tnd_kl ;              

    int idx = typ_idxs[0]*natmtyps*natmtyps*natmtyps
        + typ_idxs[1]*natmtyps*natmtyps + typ_idxs[2]*natmtyps + typ_idxs[3] ;

    int quadidx = atom_int_quad_map[idx] ;

    if(quadidx < 0)    // Skipping an excluded interaction
        return;

    vector<int> & mapped_pair_idx = pair_int_quad_map[idx] ;

    // Check whether cutoffs are within allowed ranges

    for(int i=0; i<npairs; i++)
        if (dx[i] >= chimes_4b_cutoff[ quadidx ][1][mapped_pair_idx[i]])
            return;    

    // At this point, all distances are within allowed ranges. We can now proceed to the force/stress/energy calculation
    
    // Set up the polynomials
    
    set_cheby_polys(Tn_ij, Tnd_ij, dx[0], atom_int_pair_map[ typ_idxs[0]*natmtyps + typ_idxs[1] ], chimes_4b_cutoff[quadidx][0][mapped_pair_idx[0]], chimes_4b_cutoff[quadidx][1][mapped_pair_idx[0]], 2);
    set_cheby_polys(Tn_ik, Tnd_ik, dx[1], atom_int_pair_map[ typ_idxs[0]*natmtyps + typ_idxs[2] ], chimes_4b_cutoff[quadidx][0][mapped_pair_idx[1]], chimes_4b_cutoff[quadidx][1][mapped_pair_idx[1]], 2);
    set_cheby_polys(Tn_il, Tnd_il, dx[2], atom_int_pair_map[ typ_idxs[0]*natmtyps + typ_idxs[3] ], chimes_4b_cutoff[quadidx][0][mapped_pair_idx[2]], chimes_4b_cutoff[quadidx][1][mapped_pair_idx[2]], 2);
    set_cheby_polys(Tn_jk, Tnd_jk, dx[3], atom_int_pair_map[ typ_idxs[1]*natmtyps + typ_idxs[2] ], chimes_4b_cutoff[quadidx][0][mapped_pair_idx[3]], chimes_4b_cutoff[quadidx][1][mapped_pair_idx[3]], 2);
    set_cheby_polys(Tn_jl, Tnd_jl, dx[4], atom_int_pair_map[ typ_idxs[1]*natmtyps + typ_idxs[3] ], chimes_4b_cutoff[quadidx][0][mapped_pair_idx[4]], chimes_4b_cutoff[quadidx][1][mapped_pair_idx[4]], 2);
    set_cheby_polys(Tn_kl, Tnd_kl, dx[5], atom_int_pair_map[ typ_idxs[2]*natmtyps + typ_idxs[3] ], chimes_4b_cutoff[quadidx][0][mapped_pair_idx[5]], chimes_4b_cutoff[quadidx][1][mapped_pair_idx[5]], 2);     
    
#ifdef USE_DISTANCE_TENSOR  
    // Tensor product of displacement vectors.
    double dr2[CHDIM*CHDIM*npairs*npairs] ;
    init_distance_tensor(dr2, dr, npairs) ;
#endif
    
    
    // Set up the smoothing functions
    for (int i=0; i<npairs; i++)    
        get_fcut(dx[i], chimes_4b_cutoff[quadidx][1][mapped_pair_idx[i]], fcut[i], fcutderiv[i]);


    // Product of all 6 fcuts.
    double fcut_all = fcut[0] * fcut[1] * fcut[2] * fcut[3] * fcut[4] * fcut[5]  ;

    // Product of 5 fcuts divided by dx.
    double fcut_5[npairs] ;
    fcut_5[0] = fcut[1] * fcut[2] * fcut[3] * fcut[4] * fcut[5] / dx[0] ;
    fcut_5[1] = fcut[0] * fcut[2] * fcut[3] * fcut[4] * fcut[5] / dx[1] ;
    fcut_5[2] = fcut[0] * fcut[1] * fcut[3] * fcut[4] * fcut[5] / dx[2] ;
    fcut_5[3] = fcut[0] * fcut[1] * fcut[2] * fcut[4] * fcut[5] / dx[3] ;
    fcut_5[4] = fcut[0] * fcut[1] * fcut[2] * fcut[3] * fcut[5] / dx[4] ;
    fcut_5[5] = fcut[0] * fcut[1] * fcut[2] * fcut[3] * fcut[4] / dx[5] ;
    
    // Start the force/stress/energy calculation

    // Begin setting up the pointers for the GPU.

    double *device_chimes_params, *device_force, *device_stress;
    int *device_chimes_pows;

    struct poly_pointers_4b device_polys;

    // Various malloc calls
    cudaMalloc(&device_chimes_params, chimes_4b_params[quadidx].size() * sizeof(double));
    cudaMalloc(&device_force, force.size() * sizeof(double));
    cudaMalloc(&device_stress, stress.size() * sizeof(double));
    cudaMalloc(&device_chimes_pows, chimes_4b_powers_flat[quadidx].size() * sizeof(double));

    // Tn and Tnd mallocs
    cudaMalloc(&device_polys.Tn_ij, Tn_ij.size() * sizeof(double));
    cudaMalloc(&device_polys.Tn_ik, Tn_ik.size() * sizeof(double));
    cudaMalloc(&device_polys.Tn_il, Tn_il.size() * sizeof(double));
    cudaMalloc(&device_polys.Tn_jk, Tn_jk.size() * sizeof(double));
    cudaMalloc(&device_polys.Tn_jl, Tn_jl.size() * sizeof(double));
    cudaMalloc(&device_polys.Tn_kl, Tn_kl.size() * sizeof(double));

    cudaMalloc(&device_polys.Tnd_ij, Tnd_ij.size() * sizeof(double));
    cudaMalloc(&device_polys.Tnd_ik, Tnd_ik.size() * sizeof(double));
    cudaMalloc(&device_polys.Tnd_il, Tnd_il.size() * sizeof(double));
    cudaMalloc(&device_polys.Tnd_jk, Tnd_jk.size() * sizeof(double));
    cudaMalloc(&device_polys.Tnd_jl, Tnd_jl.size() * sizeof(double));
    cudaMalloc(&device_polys.Tnd_kl, Tnd_kl.size() * sizeof(double));

    // All of the mallocs should be done, can begin transferring memory
    // over to the gpu

    // Let's start off with constant memory first because
    // I just like it better for some reason.

    cudaMemcpyToSymbol(fcut_4b, fcut, npairs * sizeof(double));
    cudaMemcpyToSymbol(fcutderiv_4b, fcutderiv, npairs * sizeof(double));
    cudaMemcpyToSymbol(fcut5_4b, fcut_5, npairs * sizeof(double));
    cudaMemcpyToSymbol(dr_4b, dr.data(), dr.size() * sizeof(double));
    cudaMemcpyToSymbol(pair_idx_4b, mapped_pair_idx.data(), mapped_pair_idx.size() * sizeof(int));
    #ifdef USE_DISTANCE_TENSOR
        cudaMemcpyToSymbol(dr2_4b, dr2, sizeof(dr2_4b));
    #endif

    cudaMemcpyToSymbol(gpu_energy, &energy, sizeof(double));

    cudaMemcpy(device_chimes_params, chimes_4b_params[quadidx].data(), chimes_4b_params[quadidx].size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_force, force.data(), force.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_stress, stress.data(), stress.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_chimes_pows, chimes_4b_powers_flat[quadidx].data(), chimes_4b_powers_flat[quadidx].size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(device_polys.Tn_ij, Tn_ij.data(), Tn_ij.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_polys.Tn_ik, Tn_ik.data(), Tn_ik.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_polys.Tn_il, Tn_il.data(), Tn_il.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_polys.Tn_jk, Tn_jk.data(), Tn_jk.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_polys.Tn_jl, Tn_jl.data(), Tn_jl.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_polys.Tn_kl, Tn_kl.data(), Tn_kl.size() * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(device_polys.Tnd_ij, Tnd_ij.data(), Tnd_ij.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_polys.Tnd_ik, Tnd_ik.data(), Tnd_ik.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_polys.Tnd_il, Tnd_il.data(), Tnd_il.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_polys.Tnd_jk, Tnd_jk.data(), Tnd_jk.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_polys.Tnd_jl, Tnd_jl.data(), Tnd_jl.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_polys.Tnd_kl, Tnd_kl.data(), Tnd_kl.size() * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 512;
    // Can start with this
    dim3 dimBlock(blockSize, 1, 1);
    dim3 dimGrid(ceil((double)ncoeffs_4b[quadidx]/blockSize), 1 , 1);

    // Start the kernel
    compute4b_helper<<<dimGrid, dimBlock>>>(ncoeffs_4b[quadidx], fcut_all, device_chimes_params, device_chimes_pows, device_force, device_stress, device_polys);
    cudaDeviceSynchronize();

    // retrieve relevant results from the GPU

    cudaMemcpyFromSymbol(&energy, gpu_energy, sizeof(double));
    cudaMemcpy(stress.data(), device_stress, stress.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(force.data(), device_force, force.size() * sizeof(double), cudaMemcpyDeviceToHost);

    // Clean up time.

    cudaFree(device_chimes_params);
    cudaFree(device_chimes_pows);
    cudaFree(device_force);
    cudaFree(device_stress);

    cudaFree(device_polys.Tn_ij);
    cudaFree(device_polys.Tn_ik);
    cudaFree(device_polys.Tn_il);
    cudaFree(device_polys.Tn_jk);
    cudaFree(device_polys.Tn_jl);
    cudaFree(device_polys.Tn_kl);

    cudaFree(device_polys.Tnd_ij);
    cudaFree(device_polys.Tnd_ik);
    cudaFree(device_polys.Tnd_il);
    cudaFree(device_polys.Tnd_jk);
    cudaFree(device_polys.Tnd_jl);
    cudaFree(device_polys.Tnd_kl);

    // all clean I think.

    /*
    for(int coeffs=0; coeffs<ncoeffs_4b[quadidx]; coeffs++)
    {
        coeff = chimes_4b_params[quadidx][coeffs];
        
        for (int i=0; i<npairs; i++)
            powers[i] = chimes_4b_powers[quadidx][coeffs][mapped_pair_idx[i]];

        double Tn_ij_ik_il =  Tn_ij[ powers[0] ] * Tn_ik[ powers[1] ] * Tn_il[ powers[2] ] ;
        double Tn_jk_jl    =  Tn_jk[ powers[3] ] * Tn_jl[ powers[4] ] ;
        double Tn_kl_5     =  Tn_kl[ powers[5] ] ;

        energy += coeff * fcut_all * Tn_ij_ik_il * Tn_jk_jl * Tn_kl_5 ;        

        deriv[0] = fcut[0] * Tnd_ij[ powers[0] ] + fcutderiv[0] * Tn_ij[ powers[0] ];
        deriv[1] = fcut[1] * Tnd_ik[ powers[1] ] + fcutderiv[1] * Tn_ik[ powers[1] ];
        deriv[2] = fcut[2] * Tnd_il[ powers[2] ] + fcutderiv[2] * Tn_il[ powers[2] ];
        deriv[3] = fcut[3] * Tnd_jk[ powers[3] ] + fcutderiv[3] * Tn_jk[ powers[3] ];
        deriv[4] = fcut[4] * Tnd_jl[ powers[4] ] + fcutderiv[4] * Tn_jl[ powers[4] ];
        deriv[5] = fcut[5] * Tnd_kl[ powers[5] ] + fcutderiv[5] * Tn_kl[ powers[5] ];        

        force_scalar[0]  = coeff * deriv[0] * fcut_5[0] * Tn_ik[powers[1]]  * Tn_il[powers[2]] * Tn_jk_jl * Tn_kl_5 ;
        force_scalar[1]  = coeff * deriv[1] * fcut_5[1] * Tn_ij[powers[0]]  * Tn_il[powers[2]] * Tn_jk_jl * Tn_kl_5 ;
        force_scalar[2]  = coeff * deriv[2] * fcut_5[2] * Tn_ij[powers[0]]  * Tn_ik[powers[1]] * Tn_jk_jl * Tn_kl_5 ;
        force_scalar[3]  = coeff * deriv[3] * fcut_5[3] * Tn_ij_ik_il  * Tn_jl[powers[4]] * Tn_kl_5 ;
        force_scalar[4]  = coeff * deriv[4] * fcut_5[4] * Tn_ij_ik_il  * Tn_jk[powers[3]] * Tn_kl_5 ;
        force_scalar[5]  = coeff * deriv[5] * fcut_5[5] * Tn_ij_ik_il * Tn_jk_jl ;

        // Accumulate forces/stresses on/from the ij pair
        
        force[0*CHDIM+0] += force_scalar[0] * dr[0*CHDIM+0];
        force[0*CHDIM+1] += force_scalar[0] * dr[0*CHDIM+1];
        force[0*CHDIM+2] += force_scalar[0] * dr[0*CHDIM+2];

        force[1*CHDIM+0] -= force_scalar[0] * dr[0*CHDIM+0];
        force[1*CHDIM+1] -= force_scalar[0] * dr[0*CHDIM+1];
        force[1*CHDIM+2] -= force_scalar[0] * dr[0*CHDIM+2];   

#ifdef USE_DISTANCE_TENSOR      
        stress[0] -= force_scalar[0]  * dr2_4B(dr2,0,0,0,0); // xx tensor component
        stress[1] -= force_scalar[0]  * dr2_4B(dr2,0,0,0,1); // xy tensor component
        stress[2] -= force_scalar[0]  * dr2_4B(dr2,0,0,0,2); // xz tensor component
        stress[3] -= force_scalar[0]  * dr2_4B(dr2,0,1,0,1); // yy tensor component
        stress[4] -= force_scalar[0]  * dr2_4B(dr2,0,1,0,2); // yz tensor component
        stress[5] -= force_scalar[0]  * dr2_4B(dr2,0,2,0,2); // zz tensor component
#else
        stress[0] -= force_scalar[0]  * dr[0*CHDIM+0] * dr[0*CHDIM+0]; // xx tensor component
        stress[1] -= force_scalar[0]  * dr[0*CHDIM+0] * dr[0*CHDIM+1]; // xy tensor component
        stress[2] -= force_scalar[0]  * dr[0*CHDIM+0] * dr[0*CHDIM+2]; // xz tensor component
        stress[3] -= force_scalar[0]  * dr[0*CHDIM+1] * dr[0*CHDIM+1]; // yy tensor component
        stress[4] -= force_scalar[0]  * dr[0*CHDIM+1] * dr[0*CHDIM+2]; // yz tensor component
        stress[5] -= force_scalar[0]  * dr[0*CHDIM+2] * dr[0*CHDIM+2]; // zz tensor component
#endif      
        
        // Accumulate forces/stresses on/from the ik pair
        
        force[0*CHDIM+0] += force_scalar[1] * dr[1*CHDIM+0];
        force[0*CHDIM+1] += force_scalar[1] * dr[1*CHDIM+1];
        force[0*CHDIM+2] += force_scalar[1] * dr[1*CHDIM+2];

        force[2*CHDIM+0] -= force_scalar[1] * dr[1*CHDIM+0];
        force[2*CHDIM+1] -= force_scalar[1] * dr[1*CHDIM+1];
        force[2*CHDIM+2] -= force_scalar[1] * dr[1*CHDIM+2];   

#if USE_DISTANCE_TENSOR     
        stress[0] -= force_scalar[1]  * dr2_4B(dr2,1,0,1,0); // xx tensor component
        stress[1] -= force_scalar[1]  * dr2_4B(dr2,1,0,1,1); // xy tensor component
        stress[2] -= force_scalar[1]  * dr2_4B(dr2,1,0,1,2); // xz tensor component
        stress[3] -= force_scalar[1]  * dr2_4B(dr2,1,1,1,1); // yy tensor component
        stress[4] -= force_scalar[1]  * dr2_4B(dr2,1,1,1,2); // yz tensor component
        stress[5] -= force_scalar[1]  * dr2_4B(dr2,1,2,1,2); // zz tensor component
#else        
        stress[0] -= force_scalar[1]  * dr[1*CHDIM+0] * dr[1*CHDIM+0]; // xx tensor component
        stress[1] -= force_scalar[1]  * dr[1*CHDIM+0] * dr[1*CHDIM+1]; // xy tensor component
        stress[2] -= force_scalar[1]  * dr[1*CHDIM+0] * dr[1*CHDIM+2]; // xz tensor component
        stress[3] -= force_scalar[1]  * dr[1*CHDIM+1] * dr[1*CHDIM+1]; // yy tensor component
        stress[4] -= force_scalar[1]  * dr[1*CHDIM+1] * dr[1*CHDIM+2]; // yz tensor component
        stress[5] -= force_scalar[1]  * dr[1*CHDIM+2] * dr[1*CHDIM+2]; // zz tensor component
#endif      
        // Accumulate forces/stresses on/from the il pair
        
        force[0*CHDIM+0] += force_scalar[2] * dr[2*CHDIM+0];
        force[0*CHDIM+1] += force_scalar[2] * dr[2*CHDIM+1];
        force[0*CHDIM+2] += force_scalar[2] * dr[2*CHDIM+2];

        force[3*CHDIM+0] -= force_scalar[2] * dr[2*CHDIM+0];
        force[3*CHDIM+1] -= force_scalar[2] * dr[2*CHDIM+1];
        force[3*CHDIM+2] -= force_scalar[2] * dr[2*CHDIM+2];   

#ifdef USE_DISTANCE_TENSOR        
        stress[0] -= force_scalar[2]  * dr2_4B(dr2,2,0,2,0); // xx tensor component
        stress[1] -= force_scalar[2]  * dr2_4B(dr2,2,0,2,1); // xy tensor component
        stress[2] -= force_scalar[2]  * dr2_4B(dr2,2,0,2,2); // xz tensor component
        stress[3] -= force_scalar[2]  * dr2_4B(dr2,2,1,2,1); // yy tensor component
        stress[4] -= force_scalar[2]  * dr2_4B(dr2,2,1,2,2); // yz tensor component
        stress[5] -= force_scalar[2]  * dr2_4B(dr2,2,2,2,2); // zz tensor component           
#else       
        stress[0] -= force_scalar[2]  * dr[2*CHDIM+0] * dr[2*CHDIM+0]; // xx tensor component
        stress[1] -= force_scalar[2]  * dr[2*CHDIM+0] * dr[2*CHDIM+1]; // xy tensor component
        stress[2] -= force_scalar[2]  * dr[2*CHDIM+0] * dr[2*CHDIM+2]; // xz tensor component
        stress[3] -= force_scalar[2]  * dr[2*CHDIM+1] * dr[2*CHDIM+1]; // yy tensor component
        stress[4] -= force_scalar[2]  * dr[2*CHDIM+1] * dr[2*CHDIM+2]; // yz tensor component
        stress[5] -= force_scalar[2]  * dr[2*CHDIM+2] * dr[2*CHDIM+2]; // zz tensor component           
#endif
        
        // Accumulate forces/stresses on/from the jk pair
        
        force[1*CHDIM+0] += force_scalar[3] * dr[3*CHDIM+0];
        force[1*CHDIM+1] += force_scalar[3] * dr[3*CHDIM+1];
        force[1*CHDIM+2] += force_scalar[3] * dr[3*CHDIM+2];

        force[2*CHDIM+0] -= force_scalar[3] * dr[3*CHDIM+0];
        force[2*CHDIM+1] -= force_scalar[3] * dr[3*CHDIM+1];
        force[2*CHDIM+2] -= force_scalar[3] * dr[3*CHDIM+2];   

#ifdef USE_DISTANCE_TENSOR      
        stress[0] -= force_scalar[3]  * dr2_4B(dr2,3,0,3,0); // xx tensor component
        stress[1] -= force_scalar[3]  * dr2_4B(dr2,3,0,3,1); // xy tensor component
        stress[2] -= force_scalar[3]  * dr2_4B(dr2,3,0,3,2); // xz tensor component
        stress[3] -= force_scalar[3]  * dr2_4B(dr2,3,1,3,1); // yy tensor component
        stress[4] -= force_scalar[3]  * dr2_4B(dr2,3,1,3,2); // yz tensor component
        stress[5] -= force_scalar[3]  * dr2_4B(dr2,3,2,3,2); // zz tensor component
#else
        stress[0] -= force_scalar[3]  * dr[3*CHDIM+0] * dr[3*CHDIM+0]; // xx tensor component
        stress[1] -= force_scalar[3]  * dr[3*CHDIM+0] * dr[3*CHDIM+1]; // xy tensor component
        stress[2] -= force_scalar[3]  * dr[3*CHDIM+0] * dr[3*CHDIM+2]; // xz tensor component
        stress[3] -= force_scalar[3]  * dr[3*CHDIM+1] * dr[3*CHDIM+1]; // yy tensor component
        stress[4] -= force_scalar[3]  * dr[3*CHDIM+1] * dr[3*CHDIM+2]; // yz tensor component
        stress[5] -= force_scalar[3]  * dr[3*CHDIM+2] * dr[3*CHDIM+2]; // zz tensor component
#endif
        
        // Accumulate forces/stresses on/from the jl pair
        
        force[1*CHDIM+0] += force_scalar[4] * dr[4*CHDIM+0];
        force[1*CHDIM+1] += force_scalar[4] * dr[4*CHDIM+1];
        force[1*CHDIM+2] += force_scalar[4] * dr[4*CHDIM+2];

        force[3*CHDIM+0] -= force_scalar[4] * dr[4*CHDIM+0];
        force[3*CHDIM+1] -= force_scalar[4] * dr[4*CHDIM+1];
        force[3*CHDIM+2] -= force_scalar[4] * dr[4*CHDIM+2];     

#ifdef USE_DISTANCE_TENSOR      
        stress[0] -= force_scalar[4]  * dr2_4B(dr2,4,0,4,0); // xx tensor component
        stress[1] -= force_scalar[4]  * dr2_4B(dr2,4,0,4,1); // xy tensor component
        stress[2] -= force_scalar[4]  * dr2_4B(dr2,4,0,4,2); // xz tensor component
        stress[3] -= force_scalar[4]  * dr2_4B(dr2,4,1,4,1); // yy tensor component
        stress[4] -= force_scalar[4]  * dr2_4B(dr2,4,1,4,2); // yz tensor component
        stress[5] -= force_scalar[4]  * dr2_4B(dr2,4,2,4,2); // zz tensor component
#else       
        stress[0] -= force_scalar[4]  * dr[4*CHDIM+0] * dr[4*CHDIM+0]; // xx tensor component
        stress[1] -= force_scalar[4]  * dr[4*CHDIM+0] * dr[4*CHDIM+1]; // xy tensor component
        stress[2] -= force_scalar[4]  * dr[4*CHDIM+0] * dr[4*CHDIM+2]; // xz tensor component
        stress[3] -= force_scalar[4]  * dr[4*CHDIM+1] * dr[4*CHDIM+1]; // yy tensor component
        stress[4] -= force_scalar[4]  * dr[4*CHDIM+1] * dr[4*CHDIM+2]; // yz tensor component
        stress[5] -= force_scalar[4]  * dr[4*CHDIM+2] * dr[4*CHDIM+2]; // zz tensor component
#endif      
        // Accumulate forces/stresses on/from the kl pair
        
        force[2*CHDIM+0] += force_scalar[5] * dr[5*CHDIM+0];
        force[2*CHDIM+1] += force_scalar[5] * dr[5*CHDIM+1];
        force[2*CHDIM+2] += force_scalar[5] * dr[5*CHDIM+2];

        force[3*CHDIM+0] -= force_scalar[5] * dr[5*CHDIM+0];
        force[3*CHDIM+1] -= force_scalar[5] * dr[5*CHDIM+1];
        force[3*CHDIM+2] -= force_scalar[5] * dr[5*CHDIM+2];     

#ifdef USE_DISTANCE_TENSOR
        stress[0] -= force_scalar[5]  * dr2_4B(dr2,5,0,5,0); // xx tensor component
        stress[1] -= force_scalar[5]  * dr2_4B(dr2,5,0,5,1); // xy tensor component
        stress[2] -= force_scalar[5]  * dr2_4B(dr2,5,0,5,2); // xz tensor component
        stress[3] -= force_scalar[5]  * dr2_4B(dr2,5,1,5,1); // yy tensor component
        stress[4] -= force_scalar[5]  * dr2_4B(dr2,5,1,5,2); // yz tensor component
        stress[5] -= force_scalar[5]  * dr2_4B(dr2,5,2,5,2); // zz tensor component
#else       
        stress[0] -= force_scalar[5]  * dr[5*CHDIM+0] * dr[5*CHDIM+0]; // xx tensor component
        stress[1] -= force_scalar[5]  * dr[5*CHDIM+0] * dr[5*CHDIM+1]; // xy tensor component
        stress[2] -= force_scalar[5]  * dr[5*CHDIM+0] * dr[5*CHDIM+2]; // xz tensor component
        stress[3] -= force_scalar[5]  * dr[5*CHDIM+1] * dr[5*CHDIM+1]; // yy tensor component
        stress[4] -= force_scalar[5]  * dr[5*CHDIM+1] * dr[5*CHDIM+2]; // yz tensor component
        stress[5] -= force_scalar[5]  * dr[5*CHDIM+2] * dr[5*CHDIM+2]; // zz tensor component
#endif      
    }
    
	force_scalar_in[0] = force_scalar[0];
	force_scalar_in[1] = force_scalar[1];
	force_scalar_in[2] = force_scalar[2];
	force_scalar_in[3] = force_scalar[3];
	force_scalar_in[4] = force_scalar[4];
	force_scalar_in[5] = force_scalar[5];
    */

    force_scalar_in[0] = 1;
	force_scalar_in[1] = 1;
	force_scalar_in[2] = 1;
	force_scalar_in[3] = 1;
	force_scalar_in[4] = 1;
	force_scalar_in[5] = 1;
   
    return;
}

void chimesFF::get_cutoff_2B(vector<vector<double> >  & cutoff_2b)
{
    int dim = chimes_2b_cutoff.size();
    
    cutoff_2b.resize(dim);
    
    for (int i=0; i<dim; i++)
    {
        cutoff_2b[i].resize(0);
        
        for (int j=0; j<chimes_2b_cutoff[i].size(); j++)
        
            cutoff_2b[i].push_back(chimes_2b_cutoff[i][j]);
    }
}

double chimesFF::max_cutoff(int ntypes, vector<vector<vector<double> > > & cutoff_list)
{
    double max = cutoff_list[0][1][0]; 
    
    for (int i=0; i<ntypes; i++)
        for (int j=0; j<cutoff_list[i][1].size(); j++)
            if (cutoff_list[i][1][j] > max)
                max = cutoff_list[i][1][j];

    return max;

}

double chimesFF::max_cutoff_2B(bool silent)
{
    double max = chimes_2b_cutoff[0][1]; 
    
    for (int i=0; i<chimes_2b_cutoff.size(); i++)
        if (chimes_2b_cutoff[i][1] > max)
            max = chimes_2b_cutoff[i][1];
    
    if ((rank == 0)&&(!silent))        
        cout << "chimesFF: " << "\t" << "Setting 2-body max cutoff to: " << max << endl;
    
    return max;    
}

double chimesFF::max_cutoff_3B(bool silent)
{
    
    if (poly_orders[1] == 0)
        return 0.0;
    
    double max = max_cutoff(chimes_3b_cutoff.size(), chimes_3b_cutoff);
    
    if ((rank == 0)&&(!silent))    
        cout << "chimesFF: " << "\t" << "Setting 3-body max cutoff to: " << max << endl;
    
    return max;
    
}

double chimesFF::max_cutoff_4B(bool silent)
{
    if (poly_orders[2] == 0)
        return 0.0;
    
    double max =  max_cutoff(chimes_4b_cutoff.size(), chimes_4b_cutoff);
        
    if ((rank == 0)&&(!silent))    
        cout << "chimesFF: " << "\t" << "Setting 4-body max cutoff to: " << max << endl;
    
    return max;
}

void chimesFF::set_atomtypes(vector<string> & type_list)
{
    type_list.resize(natmtyps);
    
    for(int i=0;i<natmtyps;i++)
        type_list[i] = atmtyps[i];
}

int chimesFF::get_atom_pair_index(int pair_id)
{
    return atom_idx_pair_map[pair_id];
}

void chimesFF::build_pair_int_quad_map()
{
    // Build the pair maps for all possible quads.  Moved build_atom_and_pair_mappers out of the compute_XX routines
    // to support GPU environment without string operations.
    // This must be called prior to force evaluation.

    const int natoms = 4 ;
    const int npairs = natoms * (natoms-1) / 2 ;
    vector<int> pair_map(npairs) ;
    vector<int> typ_idxs(natoms) ;

    if ( atom_int_quad_map.size() == 0 ) return ; // No quads !
    
    pair_int_quad_map.resize(natmtyps*natmtyps*natmtyps*natmtyps) ;

    
    for ( int i = 0 ; i < natmtyps ; i++ )
    {
        typ_idxs[0] = i ;
        for ( int j = 0 ; j < natmtyps ; j++ )
        {
            typ_idxs[1] = j ;
            for ( int k = 0 ; k < natmtyps ; k++ )
            {
                typ_idxs[2] = k ;
                for ( int l = 0 ; l < natmtyps ; l++ )
                {
                    typ_idxs[3] = l ;
                    int idx = i*natmtyps*natmtyps*natmtyps + j*natmtyps*natmtyps + k*natmtyps + l ;
                    int quadidx = atom_int_quad_map[idx];

                    // Skip excluded interactions
                    if (quadidx < 0)
                        continue;

                    build_atom_and_pair_mappers(natoms, npairs, typ_idxs, quad_params_pair_typs[quadidx], pair_map);

                    // Save for re-use in force evaluators.
                    if ( quadidx >= natmtyps * natmtyps * natmtyps * natmtyps )
                    {
                        cout << "Error: quadidx out of range\n" ;
                        cout << "Quadidx = " << quadidx << endl ;
                        exit(1) ;
                    }

                    // Note: The entire vector<> is copied and stored.                  
                    pair_int_quad_map[idx] = pair_map ;
                }
            }
        }
    }
    for ( int i = 0 ; i < pair_int_quad_map.size() ; i++ )
    {
        if ( pair_int_quad_map[i].size() == 0 )
        {
		if (atom_int_quad_map[i] >= 0)
            		cout << "Error: Did not initialize pair_int_quad_map for entry " << i << endl ;
		else
			cout << "Warning: Did not initialize pair_int_quad_map for excluded entry " << i << endl ;
        }
    }   
}

void chimesFF::build_pair_int_trip_map()
// Build the pair maps for all possible triplets.  Moved build_atom_and_pair_mappers out of the compute_XX routines
// to support GPU environment without string operations.
// This must be called prior to force evaluation.
{
    const int natoms = 3 ;
    const int npairs = natoms * (natoms-1) / 2 ;
    vector<int> pair_map(npairs) ;
    vector<int> typ_idxs(natoms) ;

    if ( atom_int_trip_map.size() == 0 ) return ; // No trips !
    
    pair_int_trip_map.resize(natmtyps*natmtyps*natmtyps) ;
    
    for ( int i = 0 ; i < natmtyps ; i++ )
    {
        typ_idxs[0] = i ;
        for ( int j = 0 ; j < natmtyps ; j++ )
        {
            typ_idxs[1] = j ;
            for ( int k = 0 ; k < natmtyps ; k++ )
            {
                typ_idxs[2] = k ;
                int tripidx = atom_int_trip_map[i*natmtyps*natmtyps + j*natmtyps + k];
		
		// Skip excluded interactions
		if (tripidx < 0)
			continue;

                build_atom_and_pair_mappers(natoms, npairs, typ_idxs, trip_params_pair_typs[tripidx], pair_map);
                    
                // Save for re-use in force evaluators.
                if ( tripidx >= natmtyps * natmtyps * natmtyps * natmtyps )
                {
                    cout << "Error: tripidx out of range\n" ;
                    cout << "Tripidx = " << tripidx << endl ;
                    exit(1) ;
                }

                // Note: The entire vector<> is copied and stored.
                pair_int_trip_map[i*natmtyps*natmtyps + j*natmtyps + k] = pair_map ;
            }
        }
    }
    for ( int i = 0 ; i < pair_int_trip_map.size() ; i++ )
    {
        if ( pair_int_trip_map[i].size() == 0 )
        {
		if (atom_int_trip_map[i] >= 0)
            		cout << "Error: Did not initialize pair_int_trip_map for entry " << i << endl ;
		else
			cout << "Warning: Did not initialize pair_int_trip_map for excluded entry " << i << endl ;
        }
    }
    
}

