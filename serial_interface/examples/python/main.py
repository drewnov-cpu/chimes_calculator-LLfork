"""

	A simple usage example for computing system force, stress, and energy via 
	the serial_chimes_calculator python wrapper.

	Expects "lib-C_wrapper-serial_interface.so" in the same directory as this script. This file
	can be generated by typing "make all" in the directory containing this file
	
	Depends on chimescalc_serial_py.py from the chimes_serial_interface's api files

	Expects to be run with python version 3.X

	Run with: "python3 <this file> <parameter file> <coordinate file>"  or 
               python3 Run with: python <this file> <parameter file> <xyz file> <allow_replicates(0/1)> <debug flag (0/1)> <path to wrapper_py.py>

    ChIMES Calculator
    Copyright (C) 2020 Rebecca K. Lindsey, Nir Goldman, and Laurence E. Fried
	Contributing Author: Rebecca K. Lindsey (2020)

"""

import os
import sys
import math

if (len(sys.argv) != 4) and (len(sys.argv) != 6):
	
	print( "ERROR: Wrong number of commandline args")
	print( "       Run with: python <this file> <parameter file> <xyz file> <allow_replicates(0/1)>")
	print( "       or")	
	print( "       Run with: python3 <this file> <parameter file> <xyz file> <allow_replicates(0/1)> <debug flag (0/1)> <path to wrapper_py.py>")
	exit()

# A small helper function

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

# Import ChIMES modules

curr_path = os.getcwd()

chimes_module_path = os.path.abspath( curr_path + "/../../api/")
if len(sys.argv) == 6:
	chimes_module_path = os.path.abspath(sys.argv[4])
sys.path.append(chimes_module_path)

import chimescalc_serial_py 

# Read in the parameter and coordinate filename 

small = False



param_file =          sys.argv[1]  # parameter file
coord_file =          sys.argv[2]  # coordinate file
small      = str2bool(sys.argv[3]) # allow replicates?
	
print("Read args:")

for i in range(len(sys.argv)-1):
	print (i+1,sys.argv[i+1])
	
# Initialize the ChIMES calculator curr_path should be /.../usr/WS2/rlindsey/chimes_calculator-fork/serial_interface/tests/

chimescalc_serial_py.chimes_wrapper = chimescalc_serial_py.init_chimes_wrapper("lib-C_wrapper-serial_interface.so")
chimescalc_serial_py.set_chimes(small)

rank = 0

chimescalc_serial_py.init_chimes(param_file, rank)

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
cell_a       = ifstream.readline().split()
cell_b       = [float(cell_a[3]), float(cell_a[4]), float(cell_a[5])]
cell_c       = [float(cell_a[6]), float(cell_a[7]), float(cell_a[8])]
cell_a       = [float(cell_a[0]), float(cell_a[1]), float(cell_a[2])]

energy = 0.0
stress = [0.0]*9
fx     = [] 
fy     = []
fz     = []

for i in range(natoms):

	atmtyps.append(ifstream.readline().split())
	
	xcrd.append(float(atmtyps[-1][1]))
	ycrd.append(float(atmtyps[-1][2]))
	zcrd.append(float(atmtyps[-1][3]))
	
	atmtyps[-1] = atmtyps[-1][0]
	
	fx.append(0.0)
	fy.append(0.0)
	fz.append(0.0)

# Do the calculations

fx, fy, fz, stress, energy = chimescalc_serial_py.calculate_chimes(
                           natoms, 
			   xcrd, 
			   ycrd, 
			   zcrd, 
			   atmtyps, 
			   cell_a,
			   cell_b,
			   cell_c,
			   energy,
			   fx,
			   fy,
			   fz,
			   stress)

print ("Success!")
print ("Energy (kcal/mol)",energy)
print("Stress tensors (GPa): ")
print(stress[0]*6.9479)
print(stress[4]*6.9479)
print(stress[8]*6.9479)
print(stress[1]*6.9479)
print(stress[2]*6.9479)
print(stress[5]*6.9479)
print("Forces (kcal/mol/A): ")
for i in range(natoms):
	print(fx[i])
	print(fy[i])
	print(fz[i])

debug = 0


if len(sys.argv) == 6:
	debug = int(sys.argv[-1])

if debug == 1:

	ofstream = open("debug.dat",'w')
	
	ofstream.write("{0:0.6f}\n".format(energy))
	ofstream.write("{0:0.6f}\n".format(stress[0]*6.9479))
	ofstream.write("{0:0.6f}\n".format(stress[4]*6.9479))
	ofstream.write("{0:0.6f}\n".format(stress[8]*6.9479))
	ofstream.write("{0:0.6f}\n".format(stress[1]*6.9479))
	ofstream.write("{0:0.6f}\n".format(stress[2]*6.9479))
	ofstream.write("{0:0.6f}\n".format(stress[5]*6.9479))
	for i in range(natoms):
		ofstream.write("{0:0.6e}\n{1:0.6e}\n{2:0.6e}\n".format(fx[i],fy[i],fz[i]))
	
	ofstream.close()
