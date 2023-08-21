#!/bin/bash

#SBATCH --job-name=cpu_benchmark
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=400m 
#SBATCH --time=1:00
#SBATCH --account=rklinds1
#SBATCH --partition=standard


./testing_interface drew_benchmarks/2b/published_params.HN3.2b-only.Tersoff.special.offsets.txt drew_benchmarks/2b/HN3.2.04gcc_20000K_#000.xyz 