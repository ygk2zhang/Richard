# templates.py

qeInputTemplate = """
&control
disk_io = 'none',
prefix = 'diffDFT',
calculation ='scf',
outdir = '{out}',
pseudo_dir = '{pseudopotentialDirectory}',
tstress = .true.
tprnfor = .true.
/
&system
ibrav=0,
nat={nat},
ntyp={ntyp},
ecutwfc={ecut},
ecutrho={erho}
occupations='smearing',
smearing = 'gaussian',
degauss = 0.01,
/
&electrons
mixing_mode='plain',
diagonalization='david',
/
&ions
ion_dynamics = 'bfgs'
/
CELL_PARAMETERS (angstrom)
{ccc}
ATOMIC_SPECIES
{atomic_species}
ATOMIC_POSITIONS (angstrom)
{aaa}
K_POINTS automatic
{k1} {k2} {k3} 0 0 0
"""

qeJobTemplate = """#!/bin/bash
#SBATCH --account=def-hpcg1725
#SBATCH --partition=reserved
#SBATCH --qos=privileged
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --job-name={jobName}
#SBATCH --output={runFile}
#SBATCH --mem-per-cpu={mem}
#SBATCH --time={time} # time (DD-HH:MM)
#SBATCH --wait

module load    cuda/11.6.1
module load    StdEnv/2020  gcc/9.3.0  openmpi/4.0.3
module load    quantumespresso/6.6

cd {folder}
export OMP_NUM_THREADS={cpus}
mpirun -np 1 pw.x < {inFile} > {outFile}
"""

mdJobTemplate = """#!/bin/bash
#SBATCH --account=def-hpcg1725
#SBATCH --partition=reserved
#SBATCH --qos=privileged
#SBATCH --nodes=1
#SBATCH --ntasks={cpus}
#SBATCH --cpus-per-task=1
#SBATCH --job-name={jobName}
#SBATCH --output={runFile}
#SBATCH --mem-per-cpu={mem}
#SBATCH --time={time} # time (DD-HH:MM)
#SBATCH --wait

module load    cuda/11.6.1
module load       StdEnv/2020  gcc/9.3.0
module load openmpi/4.0.3

cd {folder}
/usr/bin/time -o {timeFile} -f "%e" mpirun -np {cpus} /global/home/hpc5146/interface-lammps-mlip-3/lmp_mpi < {inFile} > {outFile}
"""

mdInputTemplate = """units            metal
dimension        3
boundary         p p p

atom_style       atomic
lattice          bcc {base}
region           whole block 0 {xdim} 0 {ydim} 0 {zdim} units lattice
create_box       {num_elem}  whole
{mass_block}
pair_style mlip      load_from={pot} extrapolation_control=true threshold_save=2.1 threshold_break=10  extrapolation_control:save_extrapolative_to=preselected.cfg
pair_coeff * *

neighbor	0.5 bin
neigh_modify    every 1 delay 5 check yes

timestep	0.001
thermo_style    custom step temp
thermo 1000

{create_atoms_block}
velocity all create  0.01 45454 rot yes dist gaussian 
fix  0 all recenter INIT INIT INIT 
fix     1 all npt temp {ttt} {ttt} 0.1 iso {ppp} {ppp} 1
dump 1 all xyz 1000 Meng_R.xyz
run             100000
"""

trainJobTemplate = """#!/bin/bash
#SBATCH --account=def-hpcg1725
#SBATCH --partition=reserved
#SBATCH --qos=privileged
#SBATCH --nodes=1
#SBATCH --ntasks={cpus}
#SBATCH --cpus-per-task=1
#SBATCH --job-name={jobName}
#SBATCH --output={runFile}
#SBATCH --mem-per-cpu={mem}
#SBATCH --time={time} # time (DD-HH:MM)
#SBATCH --wait

module load    cuda/11.6.1
module load       StdEnv/2020  gcc/9.3.0
module load openmpi/4.0.3

/usr/bin/time -o {timeFile} -f "%e" mpirun -np {cpus} --oversubscribe  /global/home/hpc5146/mlip-3/bin/mlp train {pot} {train} --init_random={init} --al_mode={mode}
"""
# Alternative with tight tolerance and more iters
# /usr/bin/time -o {timeFile} -f "%e" mpirun -np {cpus} --oversubscribe  /global/home/hpc5146/mlip-3/bin/mlp train {pot} {train} --iteration_limit=10000 --tolerance=0.000001 --init_random={init} --al_mode={mode}

selectJobTemplate = """#!/bin/bash
#SBATCH --account=def-hpcg1725
#SBATCH --partition=reserved
#SBATCH --qos=privileged
#SBATCH --nodes=1
#SBATCH --ntasks={cpus}
#SBATCH --cpus-per-task=1
#SBATCH --job-name={jobName}
#SBATCH --output={runFile}
#SBATCH --mem-per-cpu={mem}
#SBATCH --time={time} # time (DD-HH:MM)
#SBATCH --wait

module load    cuda/11.6.1
module load       StdEnv/2020  gcc/9.3.0
module load openmpi/4.0.3

/usr/bin/time -o {timeFile} -f "%e" mpirun -np {cpus} --oversubscribe  /global/home/hpc5146/mlip-3/bin/mlp select_add {pot} {train} {preselected} {diff}
"""

atomStrainTemplate = """&control
disk_io = 'none',
prefix = 'strainK',
calculation ='scf',
outdir = '{out}',
pseudo_dir = '{pseudo_dir}'
tstress = .true.
tprnfor = .true.
/
&system
ibrav=3,
celldm(1)={cell}
nat=1,
ntyp=1,
ecutwfc=150,
occupations='smearing',
smearing = 'gaussian',
degauss = 0.005,
/
&electrons
mixing_mode='plain',
diagonalization='david',
/
&ions
ion_dynamics = 'bfgs'
/

ATOMIC_SPECIES
K  39.0983 {pseudo}

ATOMIC_POSITIONS (angstrom)
K  0   0   0

K_POINTS automatic
20 20 20 0 0 0
"""

atomShearTemplate = """&control
disk_io = 'none',
prefix = 'shearK',
calculation ='scf',
outdir = '{out}',
pseudo_dir = '{pseudo_dir}'
tstress = .true.
tprnfor = .true.
/
&system
ibrav=0,
nat=1,
ntyp=1,
ecutwfc=60,
occupations='smearing',
smearing = 'gaussian',
degauss = 0.01,
/
&electrons
mixing_mode='plain',
diagonalization='david',
/
&ions
ion_dynamics = 'bfgs'
/

CELL_PARAMETERS (bohr)
{bbb1} {aaa2} {aaa3}
-{bbb4} {bbb5} {bbb6}
-{bbb7} -{bbb8} {bbb9}

ATOMIC_SPECIES
K  39.0983 {pseudo}

ATOMIC_POSITIONS (angstrom)
K  0   0   0

K_POINTS automatic
8 8 8 0 0 0"""
