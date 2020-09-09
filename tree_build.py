import h5py
import readtreeHDF5
import numpy as np

"""
Class to build and utilise SubLink tree of IllustrisTNG

Arguments:
  -mpi     : Instance of the MPI environment class
  -path    : Path to the simulation of interest [STRING]
  -sub_tab : Instance of the build_table class
  -snap    : Snapshot to trace back from [INT]
  -scut    : Snapshot to trace back to [INT]
"""


class trees:
    def __init__(self, mpi, path, sub_tab, snap=99, scut=25):
        """
        Set up some basics, like path to tree files
        """

        # Set up - number of snapshots, number of objects, tree path
        self.Isnap = snap
        self.Nsnaps = snap - (scut - 1)
        self.snaps = np.arange(self.Nsnaps)[::-1] + scut

        self.halos = sub_tab.tags

        self.tree_path = "{0}/postprocessing/trees/SubLink/".format(path[:-7])

        # Store Subfind table
        self.sub_tab = sub_tab

        # Get redshift list and save if required
        self.get_redshifts(mpi)

        # Initiate merger tree class instance
        self.tree = readtreeHDF5.TreeDB(self.tree_path)
        return

    def get_redshifts(self, mpi):
        """
        Compute the redshift and expansion factor list of required snaps

        Arguments:
          -mpi : Instance of the MPI environment class
        """

        try:
            # Try loading from local hdf5 save
            if not mpi.Rank:
                print("  -Reading redshifts")
            f = h5py.File("store/TNGredshifts.hdf5", "r")
            self.zred = f["Redshifts"][:]
            self.aexp = f["Aexpansion"][:]
            f.close()
        except:
            # Otherwise read TNG file, compute and store locally
            if not mpi.Rank:
                print("  -Computing redshifts")
            data = np.loadtxt(
                "/n/hernquistfs3/IllustrisTNG/Runs/L205n2500TNG/arepo/L75TNG_OutputList100.dat"
            )
            self.aexp = data[:, 0]
            self.zred = 1.0 / self.aexp - 1.0

            f = h5py.File("store/TNGredshifts.hdf5", "w")
            f.create_dataset("Redshifts", data=self.zred)
            f.create_dataset("Aexpansion", data=self.aexp)
            f.close()

        # Select those for relevant snaps
        self.zred = self.zred[self.snaps]
        self.aexp = self.aexp[self.snaps]
        return

    def build_branches(self, mpi):
        """
        Loop through tree extracting required data

        Arguments:
          -mpi : Instance of the MPI environment class
        """

        if not mpi.Rank:
            print("  -Reading tree properties")
        self.index = np.zeros((len(self.halos), self.Nsnaps), dtype=np.int) - 1
        self.M500c = (
            np.zeros((len(self.halos), self.Nsnaps), dtype=np.float) + 6.774e-11
        )
        self.M200c = (
            np.zeros((len(self.halos), self.Nsnaps), dtype=np.float) + 6.774e-11
        )
        self.M200m = (
            np.zeros((len(self.halos), self.Nsnaps), dtype=np.float) + 6.774e-11
        )
        self.Mvir = np.zeros((len(self.halos), self.Nsnaps), dtype=np.float) + 6.774e-11
        for j in range(0, len(self.halos), 1):
            branch = self.tree.get_main_branch(
                self.Isnap,
                self.sub_tab.FirstSub[j],
                keysel=[
                    "Group_M_Crit500",
                    "Group_M_Crit200",
                    "Group_M_Mean200",
                    "Group_M_TopHat200",
                    "SubhaloGrNr",
                ],
            )

            if len(branch.SubhaloGrNr) >= self.Nsnaps:
                tmp = self.Nsnaps
            else:
                tmp = len(branch.SubhaloGrNr)
            self.index[j, :tmp] = branch.SubhaloGrNr[:tmp]
            self.M500c[j, :tmp] = branch.Group_M_Crit500[:tmp]
            self.M200c[j, :tmp] = branch.Group_M_Crit200[:tmp]
            self.M200m[j, :tmp] = branch.Group_M_Mean200[:tmp]
            self.Mvir[j, :tmp] = branch.Group_M_TopHat200[:tmp]

        # Convert code units to astro
        self.M500c *= 1.0e10 / self.sub_tab.hub  # [Msun]
        self.M200c *= 1.0e10 / self.sub_tab.hub  # [Msun]
        self.M200m *= 1.0e10 / self.sub_tab.hub  # [Msun]
        self.Mvir *= 1.0e10 / self.sub_tab.hub  # [Msun]
        return
