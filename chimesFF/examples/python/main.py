"""

	A simple usage example for computing system force, stress, and energy via 
	the chimesFF python wrapper.

	Expects "libchimescalc-direct_dl.so" in the same directory as this script. This file
	can be generated by typing "make all" in the directory containing this file

	Expects to be run with python version 3.X

	Run with: "python3 <this file> <parameter file> <coordinate file>

    ChIMES Calculator
    Copyright (C) 2020 Rebecca K. Lindsey, Nir Goldman, and Laurence E. Fried
	Contributing Author: Rebecca K. Lindsey (2020)

"""
# import standard python modules

import os
import sys
import math

# Import ChIMES modules

chimes_module_path = os.path.abspath( os.getcwd() + "/../../api/")
sys.path.append(chimes_module_path)

import chimescalc_py


# Define helper functions

def get_dist(lx,ly,lz,xcrd,ycrd,zcrd,i,j):  # does this work? xrd isnt used.

	r_ij = [None]*3
	
	r_ij[0] = xcrd[j] - xcrd[i]; r_ij[0] -= lx*round(r_ij[0]/lx)
	r_ij[1] = ycrd[j] - ycrd[i]; r_ij[1] -= ly*round(r_ij[1]/ly)
	r_ij[2] = zcrd[j] - zcrd[i]; r_ij[2] -= lz*round(r_ij[2]/lz)
	
	dist_ij = math.sqrt(r_ij[0]*r_ij[0] + r_ij[1]*r_ij[1] + r_ij[2]*r_ij[2])
	
	return dist_ij, r_ij
	


# Initialize the ChIMES calculator

chimescalc_py.chimes_wrapper = chimescalc_py.init_chimes_wrapper(os.getcwd() + "/libchimescalc-direct_dl.so")
chimescalc_py.set_chimes()
chimescalc_py.init_chimes()


# Read in the parameter and coordinate filename 

if len(sys.argv) != 3:
	
	print( "ERROR: Wrong number of commandline args")
	print( "       Run with: python <this file> <parameter file> <xyz file>")
	exit()

param_file = sys.argv[1] # parameter file
coord_file = sys.argv[2] # coordinate file


# Read the parameters

chimescalc_py.read_params(param_file)

# Read the coordinates, set up the force, stress, and energy vars

natoms  = None
lx      = None
ly      = None
lz      = None
atmtyps = []
xcrd    = []
ycrd    = []
zcrd    = []

ifstream = open(coord_file,'r')
natoms   = int(ifstream.readline())
lx       = ifstream.readline().split()
ly       = float(lx[4])
lz       = float(lx[8])
lx       = float(lx[0])

energy   = 0.0
stress   = [0.0]*9
forces   = [] 

for i in range(natoms):

	atmtyps.append(ifstream.readline().split())
	
	xcrd.append(float(atmtyps[-1][1]))
	ycrd.append(float(atmtyps[-1][2]))
	zcrd.append(float(atmtyps[-1][3]))
	
	atmtyps[-1] = atmtyps[-1][0]
	
	forces.append([0.0,0.0,0.0])


# Do the calculations using PBC, without neighbor lists

maxcut_2b = chimescalc_py.get_chimes_max_2b_cutoff()
maxcut_3b = chimescalc_py.get_chimes_max_3b_cutoff()
maxcut_4b = chimescalc_py.get_chimes_max_4b_cutoff()
maxcut    = max([maxcut_2b,maxcut_3b,maxcut_4b])

order_2b = chimescalc_py.get_chimes_2b_order()
order_3b = chimescalc_py.get_chimes_3b_order()
order_4b = chimescalc_py.get_chimes_4b_order()

tmp_force = None

for i in range(natoms):

	for j in range(i+1,natoms):
	
		dist_ij, r_ij = get_dist(lx,ly,lz,xcrd,ycrd,zcrd,i,j)

		if dist_ij >= maxcut:
			continue

		tmp_force, stress, energy = chimescalc_py.chimes_compute_2b_props(dist_ij, r_ij,[atmtyps[i],atmtyps[j]], [forces[i],forces[j]], stress, energy)

		forces[i][0] = tmp_force[0][0]; forces[i][1] = tmp_force[0][1]; forces[i][2] = tmp_force[0][2]
		forces[j][0] = tmp_force[1][0]; forces[j][1] = tmp_force[1][1]; forces[j][2] = tmp_force[1][2]
			
		if (order_3b > 0) or (order_4b> 0):

			for k in range(j+1,natoms):
			
				dist_ik, r_ik = get_dist(lx,ly,lz,xcrd,ycrd,zcrd,i,k)
				dist_jk, r_jk = get_dist(lx,ly,lz,xcrd,ycrd,zcrd,j,k)

				if dist_ik >= maxcut:
					continue
				if dist_jk >= maxcut:
					continue
				
				print([dist_ij,    dist_ik,    dist_jk])
				print([r_ij,       r_ik,       r_jk])
				print([atmtyps[i], atmtyps[j], atmtyps[k]])
				
				tmp_force, stress, energy = chimescalc_py.chimes_compute_3b_props(
					[dist_ij,    dist_ik,    dist_jk],
			 		[r_ij,       r_ik,       r_jk],
					[atmtyps[i], atmtyps[j], atmtyps[k]],
					[forces[i],  forces[j],  forces[k] ],
					stress, 
					energy)

				print(energy)
				sys.exit()

				forces[i][0] = tmp_force[0][0]; forces[i][1] = tmp_force[0][1]; forces[i][2] = tmp_force[0][2]
				forces[j][0] = tmp_force[1][0]; forces[j][1] = tmp_force[1][1]; forces[j][2] = tmp_force[1][2]
				forces[k][0] = tmp_force[2][0]; forces[k][1] = tmp_force[2][1]; forces[k][2] = tmp_force[2][2]
				
				if dist_ik >= maxcut_4b:
					continue
				if dist_jk >= maxcut_4b:
					continue
				
				if order_4b> 0:

					for l in range(k+1,natoms):
					
						dist_il, r_il = get_dist(lx,ly,lz,xcrd,ycrd,zcrd,i,l)
						dist_jl, r_jl = get_dist(lx,ly,lz,xcrd,ycrd,zcrd,j,l)
						dist_kl, r_kl = get_dist(lx,ly,lz,xcrd,ycrd,zcrd,k,l)

						if dist_il >= maxcut_4b:
							continue
						if dist_jl >= maxcut_4b:
							continue
						if dist_kl >= maxcut_4b:
							continue
						
						tmp_force, stress, energy = chimescalc_py.chimes_compute_4b_props(
							[dist_ij, dist_ik, dist_il, dist_jk, dist_jl, dist_kl],
					 		[r_ij,    r_ik,    r_il,   r_jk,    r_jl,    r_kl],
							[atmtyps[i], atmtyps[j], atmtyps[k], atmtyps[l]],
							[forces[i],  forces[j],  forces[k], forces[l] ],
							stress, 
							energy)

						forces[i][0] = tmp_force[0][0]; forces[i][1] = tmp_force[0][1]; forces[i][2] = tmp_force[0][2]
						forces[j][0] = tmp_force[1][0]; forces[j][1] = tmp_force[1][1]; forces[j][2] = tmp_force[1][2]
						forces[k][0] = tmp_force[2][0]; forces[k][1] = tmp_force[2][1]; forces[k][2] = tmp_force[2][2]
						forces[l][0] = tmp_force[3][0]; forces[l][1] = tmp_force[3][1]; forces[l][2] = tmp_force[3][2]



# Print results

print(energy)

for i in range(natoms):
	print(forces[i][0], forces[i][1], forces[i][2])
	
for i in range(9):
	print(stress[i]/lx/ly/lz*6.9479)



	
