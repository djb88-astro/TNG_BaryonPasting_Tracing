import h5py
import mpi_init
import subfind_data
import cosmology
import tree_build
import numpy as np
from scipy.interpolate import interp1d


def compute_accretion_rates(mpi, path, datasets, data, snap=99, mcut=1.0e12):
    """
    Compute the accretion rate based on different mass estimates
    """

    if not mpi.Rank:
        print(" > Examining: {0}\n > Snap: {1:03d}".format(path, snap), flush=True)

    # First we need to find the halos of interest
    subfind_table = subfind_data.build_table(mpi, sim=path, snap=snap)
    subfind_table.select_halos(mpi, cut=mcut)
    if not mpi.Rank:
        print(" > Found {0:d} halo(s)".format(len(subfind_table.tags)), flush=True)

    # Now rebuild the trees for those halos
    if not mpi.Rank:
        print(" > Building merger tree for halos...", flush=True)
    Mtrees = tree_build.trees(mpi, path, subfind_table, snap)

    Mtrees.build_branches(mpi)

    # Initialize cosmology class instance
    cosmo = cosmology.cosmology(
        subfind_table.hub, subfind_table.omega_m, subfind_table.omega_L
    )

    # Age of Universe at snapshots
    if not mpi.Rank:
        print(" > Computing mass accretion rates", flush=True)
    age_Gyr = cosmo.age(Mtrees.zred)

    # Now compute accretion rates for halos
    tdyn_500c_Gyr = cosmo.t_dynamic_Gyr(Mtrees.zred, delta=500.0, mode="CRIT")
    tdyn_200c_Gyr = cosmo.t_dynamic_Gyr(Mtrees.zred, delta=200.0, mode="CRIT")
    tdyn_200m_Gyr = cosmo.t_dynamic_Gyr(Mtrees.zred, delta=200.0, mode="MEAN")
    tdyn_vir_Gyr = cosmo.t_dynamic_Gyr(Mtrees.zred, mode="VIR")

    # Compute age of Universe one dynamical time ago for each snapshot
    dt_500c_Gyr = age_Gyr - tdyn_500c_Gyr
    dt_200c_Gyr = age_Gyr - tdyn_200c_Gyr
    dt_200m_Gyr = age_Gyr - tdyn_200m_Gyr
    dt_vir_Gyr = age_Gyr - tdyn_vir_Gyr

    # Delta log(a) of a dynamical time for all snapshots
    aexp_int = interp1d(age_Gyr, Mtrees.aexp, fill_value="extrapolate")
    Delta_lgAexp_500c = np.log(Mtrees.aexp) - np.log(aexp_int(dt_500c_Gyr))
    Delta_lgAexp_200c = np.log(Mtrees.aexp) - np.log(aexp_int(dt_200c_Gyr))
    Delta_lgAexp_200m = np.log(Mtrees.aexp) - np.log(aexp_int(dt_200m_Gyr))
    Delta_lgAexp_vir = np.log(Mtrees.aexp) - np.log(aexp_int(dt_vir_Gyr))

    # Now loop over haloes computing Delta log(M) -- with appropriate mass definition
    Delta_lgM500c = np.zeros(Mtrees.M500c.shape, dtype=np.float)
    Delta_lgM200c = np.zeros(Mtrees.M200c.shape, dtype=np.float)
    Delta_lgM200m = np.zeros(Mtrees.M200m.shape, dtype=np.float)
    Delta_lgMvir = np.zeros(Mtrees.Mvir.shape, dtype=np.float)
    for j in range(0, len(Mtrees.index), 1):
        lgM500c_int = interp1d(
            age_Gyr, np.log(Mtrees.M500c[j]), fill_value="extrapolate"
        )
        lgM200c_int = interp1d(
            age_Gyr, np.log(Mtrees.M200c[j]), fill_value="extrapolate"
        )
        lgM200m_int = interp1d(
            age_Gyr, np.log(Mtrees.M200m[j]), fill_value="extrapolate"
        )
        lgMvir_int = interp1d(age_Gyr, np.log(Mtrees.Mvir[j]), fill_value="extrapolate")

        Delta_lgM500c[j] = np.log(Mtrees.M500c[j]) - lgM500c_int(dt_500c_Gyr)
        Delta_lgM200c[j] = np.log(Mtrees.M200c[j]) - lgM200c_int(dt_200c_Gyr)
        Delta_lgM200m[j] = np.log(Mtrees.M200m[j]) - lgM200m_int(dt_200m_Gyr)
        Delta_lgMvir[j] = np.log(Mtrees.Mvir[j]) - lgMvir_int(dt_vir_Gyr)

    # Now compute mass accretion rates
    Macc_500c = Delta_lgM500c / Delta_lgAexp_500c
    Macc_200c = Delta_lgM200c / Delta_lgAexp_200c
    Macc_200m = Delta_lgM200m / Delta_lgAexp_200m
    Macc_vir = Delta_lgMvir / Delta_lgAexp_vir

    # Store and return
    data["Halos"] = Mtrees.halos
    data["Indices"] = Mtrees.index
    data["Redshifts"] = Mtrees.zred
    data["Snapshots"] = Mtrees.snaps
    data["M200mean"] = Mtrees.M200m
    data["M200crit"] = Mtrees.M200c
    data["M500crit"] = Mtrees.M500c
    data["Mvir"] = Mtrees.Mvir
    data["Macc_500c"] = Macc_500c
    data["Macc_200c"] = Macc_200c
    data["Macc_200m"] = Macc_200m
    data["Macc_vir"] = Macc_vir
    del Mtrees
    return


def select_massive_subhalos_with_redshift(mpi, path, data):
    """
    Loop through snapshots reading subfind tables and extracting most massive subhalos
    """

    if not mpi.Rank:
        print(" > Extracting subhalo masses", flush=True)
    masses = data["M200crit"]
    snaps = data["Snapshots"]
    indices = data["Indices"]
    Msubs = np.zeros((len(data["Halos"]), len(snaps), 5), dtype=np.float)
    for j in range(0, len(snaps), 1):
        if not mpi.Rank:
            print("  -{0:2d}".format(snaps[j]), flush=True)
        # First read subfind table to minimum mass
        subfind_table = subfind_data.build_table(mpi, sim=path, snap=snaps[j])
        subfind_table.select_halos(mpi, cut="IDX", max_idx=indices[:, j].max())

        # Now loop over halo indices from the tree trace
        for k in range(0, len(indices[:, j]), 1):
            # Skip halos that are no longer in the tree -- ID == -1
            if indices[k, j] == -1:
                continue
            kdx = np.where(subfind_table.idx == indices[k, j])[0][0]
            mdx = np.argsort(subfind_table.SubMass[kdx])[-5:]
            Msubs[k, j, 0 : len(mdx)] = ((subfind_table.SubMass[kdx])[mdx])[
                ::-1
            ] / 1.989e33
        del subfind_table

    # Store and return
    data["Msubs"] = Msubs
    del masses, snaps, indices, Msubs
    return


def save_data(mpi, path, data):
    """
    Save data to hdf5 file
    """

    tag = {
        "L75n455TNG": "TNG100_L3",
        "L75n910TNG": "TNG100_L2",
        "L75n1820TNG": "TNG100_L1",
        "L205n625TNG": "TNG300_L3",
        "L205n1250TNG": "TNG300_L2",
        "L205n2500TNG": "TNG300_L1",
    }

    # Build output file name
    fname = "output/{0}_tracing.hdf5".format(tag[path.split("/")[-2]])

    # Only Rank zero outputs
    if not mpi.Rank:
        print(" > Saving data to {0}".format(fname), flush=True)
        # Open file
        f = h5py.File(fname, "w")

        # Save some basics
        f.create_dataset("Redshifts", data=data["Redshifts"])
        f.create_dataset("Snapshot_list", data=data["Snapshots"])

        # Loop over halos storing relevant quanities
        for j in range(0, len(data["Halos"]), 1):
            grp = f.create_group(data["Halos"][j])
            f.create_dataset(
                "{0}/FoF_IDs".format(data["Halos"][j]), data=data["Indices"][j]
            )
            f.create_dataset(
                "{0}/M200mean_Msun".format(data["Halos"][j]), data=data["M200mean"][j]
            )
            f.create_dataset(
                "{0}/M200crit_Msun".format(data["Halos"][j]), data=data["M200crit"][j]
            )
            f.create_dataset(
                "{0}/M500crit_Msun".format(data["Halos"][j]), data=data["M500crit"][j]
            )
            f.create_dataset(
                "{0}/Mvir_Msun".format(data["Halos"][j]), data=data["Mvir"][j]
            )
            f.create_dataset(
                "{0}/Macc_500c".format(data["Halos"][j]), data=data["Macc_500c"][j]
            )
            f.create_dataset(
                "{0}/Macc_200c".format(data["Halos"][j]), data=data["Macc_200c"][j]
            )
            f.create_dataset(
                "{0}/Macc_200m".format(data["Halos"][j]), data=data["Macc_200m"][j]
            )
            f.create_dataset(
                "{0}/Macc_vir".format(data["Halos"][j]), data=data["Macc_vir"][j]
            )
            f.create_dataset(
                "{0}/Msubs_Top5_Msun".format(data["Halos"][j]), data=data["Msubs"][j]
            )
        f.close()
    # Wait for save to finish and return
    mpi.comm.Barrier()
    return


if __name__ == "__main__":

    # Simulation of interest
    paths = [
        "/n/hernquistfs3/IllustrisTNG/Runs/L75n455TNG/output",
        "/n/hernquistfs3/IllustrisTNG/Runs/L75n910TNG/output",
        "/n/hernquistfs3/IllustrisTNG/Runs/L75n1820TNG/output",
        "/n/hernquistfs3/IllustrisTNG/Runs/L205n625TNG/output",
        "/n/hernquistfs3/IllustrisTNG/Runs/L205n1250TNG/output",
        "/n/hernquistfs3/IllustrisTNG/Runs/L205n2500TNG/output",
    ]

    # Initialize MPI environment
    mpi = mpi_init.mpi()

    # Datasets required
    datasets = [
        "Group/Group_M_Crit500",
        "Group/Group_M_Crit200",
        "Group/Group_M_Mean200",
        "Group/Group_M_TopHat200",
        "Group/GroupFirstSub",
    ]

    # Loop over simulations
    for x in paths:
        if "L205n" in x:
            cut = 1.0e13
        elif "L75n" in x:
            cut = 1.0e12
        data = {}

        compute_accretion_rates(mpi, x, datasets, data, mcut=cut)

        select_massive_subhalos_with_redshift(mpi, x, data)

        save_data(mpi, x, data)
        quit()
