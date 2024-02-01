qeInputTemplate = """
&control
   disk_io = 'none',
   prefix = 'diffDFT',
   calculation ='scf',
   outdir = '$out',
   pseudo_dir = '/global/home/hpc5146',
   tstress = .true.
   tprnfor = .true.
 /
 &system
   ibrav=0,
   nat=$nnn,
   ntyp=1,
   ecutwfc=$ecut,
   ecutrho=$erho
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
   $ccc
ATOMIC_SPECIES
K  39.0983 K.pbe-spn-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS (angstrom)
   $aaa
K_POINTS automatic
$k1 $k2 $k3 0 0 0
"""

qeJobTemplate = """#!/bin/bash
#SBATCH --account=def-hpcg1725
#SBATCH --partition=reserved
#SBATCH --qos=privileged
#SBATCH --nodes=1
#SBATCH --ntasks=$cpus
#SBATCH --cpus-per-task=1
#SBATCH --job-name=$jobName
#SBATCH --output=$runFile
#SBATCH --mem-per-cpu=$mem
#SBATCH --time=$time # time (DD-HH:MM)
#SBATCH --wait

module load    StdEnv/2020  gcc/9.3.0  openmpi/4.0.3
module load    quantumespresso/6.6

cd $folder

pw.x < $inFile > $outFile
"""

mdJobTemplate = """#!/bin/bash
#SBATCH --account=def-hpcg1725
#SBATCH --partition=reserved
#SBATCH --qos=privileged
#SBATCH --nodes=1
#SBATCH --ntasks=$cpus
#SBATCH --cpus-per-task=1
#SBATCH --job-name=$jobName
#SBATCH --output=$runFile
#SBATCH --mem-per-cpu=$mem
#SBATCH --time=$time # time (DD-HH:MM)
#SBATCH --wait

module load       StdEnv/2020  gcc/9.3.0  cuda/11.2.2
module load openmpi/4.0.3

cd $folder
mpirun -np $cpus /global/home/hpc5146/interface-lammps-mlip-3/lmp_mpi < $inFile > $outFile
"""

mdInputTemplate = """units            metal
dimension        3
boundary         p p p


atom_style       atomic
lattice          bcc $base
region           whole block 0 $111 0 $222 0 $333 units lattice
create_box       1  whole
create_atoms     1 region whole
mass             1 39.0983

pair_style mlip      load_from=$pot extrapolation_control=true threshold_save=2.1 threshold_break=10  extrapolation_control:save_extrapolative_to=preselected.cfg
pair_coeff * *

neighbor	0.5 bin
neigh_modify    every 1 delay 5 check yes

timestep	0.001

fix		1 all nve
fix		2 all langevin $ttt $ttt 0.1 826234 zero yes

thermo_style    custom step temp 
thermo 1000


run             100000
reset_timestep  0
"""

trainJobTemplate = """#!/bin/bash
#SBATCH --account=def-hpcg1725
#SBATCH --partition=reserved
#SBATCH --qos=privileged
#SBATCH --nodes=1
#SBATCH --ntasks=$cpus
#SBATCH --cpus-per-task=1
#SBATCH --job-name=$jobName
#SBATCH --output=$runFile
#SBATCH --mem-per-cpu=$mem
#SBATCH --time=$time # time (DD-HH:MM)
#SBATCH --wait

module load       StdEnv/2020  gcc/9.3.0  cuda/11.2.2
module load openmpi/4.0.3

mpirun -np $cpus --oversubscribe  /global/home/hpc5146/mlip-3/bin/mlp train $pot $train --iteration_limit=10000 --tolerance=0.000001 --init_random=$init
"""

selectJobTemplate = """#!/bin/bash
#SBATCH --account=def-hpcg1725
#SBATCH --partition=reserved
#SBATCH --qos=privileged
#SBATCH --nodes=1
#SBATCH --ntasks=$cpus
#SBATCH --cpus-per-task=1
#SBATCH --job-name=$jobName
#SBATCH --output=$runFile
#SBATCH --mem-per-cpu=$mem
#SBATCH --time=$time # time (DD-HH:MM)
#SBATCH --wait

module load       StdEnv/2020  gcc/9.3.0  cuda/11.2.2
module load openmpi/4.0.3

mpirun -np $cpus --oversubscribe  /global/home/hpc5146/mlip-3/bin/mlp select_add $pot $train $preselected $diff
"""

atomStrainTemplate = """&control
    disk_io = 'none',
    prefix = 'strainK',
    calculation ='scf',
    outdir = '$out',
    pseudo_dir = '$pseudo_dir'
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
$aaa1 $aaa2 $aaa3
-$aaa4 $aaa5 $aaa6
-$aaa7 -$aaa8 $aaa9

ATOMIC_SPECIES
K  39.0983 $pseudo

ATOMIC_POSITIONS (angstrom)
K  0   0   0

K_POINTS automatic
8 8 8 0 0 0
"""

atomShearTemplate = """&control
    disk_io = 'none',
    prefix = 'shearK',
    calculation ='scf',
    outdir = '$out',
    pseudo_dir = '$pseudo_dir'
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
    $bbb1 $aaa2 $aaa3
    -$bbb4 $bbb5 $bbb6
    -$bbb7 -$bbb8 $bbb9

ATOMIC_SPECIES
    K  39.0983 $pseudo

ATOMIC_POSITIONS (angstrom)
    K  0   0   0
    
K_POINTS automatic
    8 8 8 0 0 0"""
