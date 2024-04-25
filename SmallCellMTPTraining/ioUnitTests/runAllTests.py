import numpy as np
import pytest
import SmallCellMTPTraining.io.parsers as pa
import SmallCellMTPTraining.io.writers as wr


def testParseQEDir():
    out = pa.parseAllQEInDirectory("./", False)[0]
    out2 = pa.parseAllQEInDirectory("./", False)[1]
    out3 = pa.parseAllQEInDirectory("./", False)[2]

    assert out["cpuTimeSpent"] == pytest.approx(2 * 60 + 15.72)
    assert out2["cpuTimeSpent"] == pytest.approx(1 * 3600 + 16 * 60)
    assert out3["cpuTimeSpent"] == pytest.approx(1 * 24 * 3600 + 5 * 3600 + 10 * 60)

    assert len(out2["atomPositions"]) == 16
    assert len(out3["atomPositions"]) == 54

    # print(out)
    assert out["energy"] == -226.03075955
    assert out["pressure"] == 0.42
    assert out["cpuTimeSpent"] == 135.72
    for x, y in zip(out["atomIDs"], [1, 2]):
        assert x == y
    for x, y in zip(out["atomTypes"], [1, 1]):
        assert x == y
    for x, y in zip(
        out["superCell"],
        [
            [9.689763, 0, 0],
            [0, 9.689763, 0],
            [0, 0, 9.689763],
        ],
    ):
        for i, j in zip(x, y):
            assert i == j
    for x, y in zip(
        out["stressVectors"],
        [
            [2.35634691e-03, -1.54663697e-04, -8.18807805e-05],
            [-1.54663697e-04, 2.69296789e-03, 4.54893225e-05],
            [-8.18807805e-05, 4.54893225e-05, 2.76575081e-03],
        ],
    ):
        for i, j in zip(x, y):
            assert pytest.approx(i) == j
    for x, y in zip(
        out["atomPositions"],
        [
            [0.07627297, -0.03669513, -0.01933011],
            [4.76860853, 4.88157663, 4.86421161],
        ],
    ):
        for i, j in zip(x, y):
            assert pytest.approx(i) == j
    for x, y in zip(
        out["virialStresses"],
        [
            0.0023563469068098135,
            0.0026929678934969293,
            0.0027657508095373868,
            4.548932252528596e-05,
            -8.188078054551475e-05,
            -0.0001546636965859723,
        ],
    ):
        assert pytest.approx(x) == y
    for x, y in zip(
        out["atomForces"],
        [[-0.0012565, 0.00061458, 0.00033293], [0.0012565, -0.00061458, -0.00033293]],
    ):
        for i, j in zip(x, y):
            assert pytest.approx(i) == j


def testParseMTPFIle():
    out = pa.parseMTPConfigsFile("mtpTest.cfg", convertFromAngRy=False)[0]
    # print(out)
    assert out["energy"] == pytest.approx(-1.2795996451918654)
    assert out["numAtoms"] == 2
    for x, y in zip(out["atomIDs"], [1, 2]):
        assert x == y
    for x, y in zip(out["atomTypes"], [0, 0]):
        assert x == y
    for x, y in zip(
        out["superCell"],
        [[5.12760176, 0.0, 0.0], [0.0, 5.12760176, 0.0], [0.0, 0.0, 5.12760176]],
    ):
        for i, j in zip(x, y):
            assert i == pytest.approx(j)
    for x, y in zip(
        out["atomPositions"],
        [[5.02192957, 0.82414507, 4.86074854], [2.66947306, 1.7396558, 2.83065409]],
    ):
        for i, j in zip(x, y):
            assert pytest.approx(i) == j

    for x, y in zip(
        out["virialStresses"],
        [2.06061146, -0.52421995, 1.52562241, -0.36293103, 0.20919285, -0.17379098],
    ):
        assert pytest.approx(x) == y
    for x, y in zip(
        out["atomForces"],
        [[0.22891356, -0.43058988, 0.47981519], [-0.22891356, 0.43058988, -0.47981519]],
    ):
        for i, j in zip(x, y):
            assert pytest.approx(i) == j


def testWriteQEInput():
    testQE = pa.parseQEOutput("dftTest.out")
    testQE["kPoints"] = [3, 3, 3]
    testQE["ecutwfc"] = 60
    testQE["ecutrho"] = 300
    testQE["qeOutDir"] = "./"
    out = wr.writeQEInput("test.test", testQE)
    assert (
        out
        == """
&control
   disk_io = 'none',
   prefix = 'diffDFT',
   calculation ='scf',
   outdir = './',
   pseudo_dir = '/global/home/hpc5146',
   tstress = .true.
   tprnfor = .true.
 /
 &system
   ibrav=0,
   nat=2,
   ntyp=1,
   ecutwfc=60,
   ecutrho=300
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
   5.12760176 0.         0.        
  0.         5.12760176 0.        
  0.         0.         5.12760176
ATOMIC_SPECIES
K  39.0983 K.pbe-spn-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS (angstrom)
    K 0.040362 -0.019418 -0.010229 
  K 2.523439 2.583219 2.574030 

K_POINTS automatic
3 3 3 0 0 0
"""
    )


def testWriteQEJob():
    jobProperties = {
        "jobName": "test",
        "ncpus": 3,
        "runFile": "./test.run",
        "maxDuration": "0-0:03",
        "memPerCpu": "4G",
        "inFile": "./test.test",
        "outFile": "./test.testes",
    }
    out = wr.writeQEJob("test.qsub", jobProperties)
    assert (
        out
        == """#!/bin/bash
#SBATCH --account=def-hpcg1725
#SBATCH --partition=reserved
#SBATCH --qos=privileged
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --job-name=test
#SBATCH --output=./test.run
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0-0:03 # time (DD-HH:MM)
#SBATCH --wait

module load    cuda/11.6.1
module load    StdEnv/2020  gcc/9.3.0  openmpi/4.0.3
module load    quantumespresso/6.6

cd .

pw.x < test.test > ./test.testes
"""
    )


def testWriteMDJob():
    jobProperties = {
        "jobName": "test",
        "ncpus": 3,
        "runFile": "./test.run",
        "maxDuration": "0-0:03",
        "memPerCpu": "4G",
        "inFile": "./test.mdin",
        "outFile": "./test.testes",
    }
    out = wr.writeMDJob("test.qsub", jobProperties)
    assert (
        out
        == """#!/bin/bash
#SBATCH --account=def-hpcg1725
#SBATCH --partition=reserved
#SBATCH --qos=privileged
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --job-name=test
#SBATCH --output=./test.run
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0-0:03 # time (DD-HH:MM)
#SBATCH --wait

module load    cuda/11.6.1
module load       StdEnv/2020  gcc/9.3.0
module load openmpi/4.0.3

cd .
mpirun -np 3 /global/home/hpc5146/interface-lammps-mlip-3/lmp_mpi < test.mdin > ./test.testes
"""
    )


def testWriteMDInput():
    jobProperties = {
        "latticeParameter": 5,
        "boxDimensions": [1, 2, 3],
        "temperature": 600,
        "potFile": "./test.pot",
    }
    out = wr.writeMDInput("test.mdin", jobProperties)
    assert (
        out
        == """units            metal
dimension        3
boundary         p p p


atom_style       atomic
lattice          bcc 5
region           whole block 0 1 0 2 0 3 units lattice
create_box       1  whole
create_atoms     1 region whole
mass             1 39.0983

pair_style mlip      load_from=./test.pot extrapolation_control=true threshold_save=2.1 threshold_break=10  extrapolation_control:save_extrapolative_to=preselected.cfg
pair_coeff * *

neighbor	0.5 bin
neigh_modify    every 1 delay 5 check yes

timestep	0.001

fix		1 all nve
fix		2 all langevin 600 600 0.1 826234 zero yes

thermo_style    custom step temp 
thermo 1000


run             100000
reset_timestep  0
"""
    )


def testWriteTrainJob():
    jobProperties = {
        "jobName": "test",
        "ncpus": 3,
        "runFile": "./test.run",
        "timeFile": "./test.time",
        "maxDuration": "0-0:03",
        "memPerCpu": "4G",
        "potFile": "./test.pot",
        "trainFile": "./mtpTest.cfg",
        "initRandom": "true",
    }
    out = wr.writeTrainJob("test.qsub", jobProperties)
    assert (
        out
        == """#!/bin/bash
#SBATCH --account=def-hpcg1725
#SBATCH --partition=reserved
#SBATCH --qos=privileged
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --job-name=test
#SBATCH --output=./test.run
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0-0:03 # time (DD-HH:MM)
#SBATCH --wait

module load    cuda/11.6.1
module load       StdEnv/2020  gcc/9.3.0
module load openmpi/4.0.3

/usr/bin/time -o ./test.time -f "%e" mpirun -np 3 --oversubscribe  /global/home/hpc5146/mlip-3/bin/mlp train ./test.pot ./mtpTest.cfg --iteration_limit=10000 --tolerance=0.000001 --init_random=true --al_mode=nbh
"""
    )


def testWriteSelectJob():
    jobProperties = {
        "jobName": "test",
        "ncpus": 3,
        "runFile": "./test.run",
        "timeFile": "./test.time",
        "maxDuration": "0-0:03",
        "memPerCpu": "4G",
        "potFile": "./test.pot",
        "trainFile": "./mtpTest.cfg",
        "preselectedFile": "./preselected.cfg",
        "diffFile": "./diff.cfg",
    }
    out = wr.writeSelectJob("test.qsub", jobProperties)
    assert (
        out
        == """#!/bin/bash
#SBATCH --account=def-hpcg1725
#SBATCH --partition=reserved
#SBATCH --qos=privileged
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --job-name=test
#SBATCH --output=./test.run
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0-0:03 # time (DD-HH:MM)
#SBATCH --wait

module load    cuda/11.6.1
module load       StdEnv/2020  gcc/9.3.0
module load openmpi/4.0.3

/usr/bin/time -o ./test.time -f "%e" mpirun -np 3 --oversubscribe  /global/home/hpc5146/mlip-3/bin/mlp select_add ./test.pot ./mtpTest.cfg ./preselected.cfg ./diff.cfg
"""
    )


def testParsePartialMTPFile():
    out = pa.parsePartialMTPConfigsFile("./partial.cfg")[0]
    assert out["numAtoms"] == 2
    for x, y in zip(out["atomIDs"], [1, 2]):
        assert x == y
    for x, y in zip(out["atomTypes"], [0, 0]):
        assert x == y
    for x, y in zip(
        out["atomPositions"],
        [[0, 0, 0], [1.510940, 2.510940, 3.510940]],
    ):
        for i, j in zip(x, y):
            assert pytest.approx(i) == j
    for x, y in zip(
        out["superCell"],
        [
            [5.021881, 0, 0],
            [0, 5.021881, 0],
            [0, 0, 5.021881],
        ],
    ):
        for i, j in zip(x, y):
            assert i == j


def testParseTime():
    assert pa.parseTimeFile("timeReport.txt") == pytest.approx(0.54)
    assert (pa.parseMDTime("sampleMD.txt")) == pytest.approx(56 * 60 + 16)


if __name__ == "__main__":
    # testParseQEDir()
    # testParseMTPFIle()
    # testWriteQEInput()
    # testWriteQEJob()
    # testWriteMDJob()
    # testWriteMDInput()
    # testWriteTrainJob()
    # testWriteSelectJob()
    # testParsePartialMTPFile()
    # testParseTime()

    pass
