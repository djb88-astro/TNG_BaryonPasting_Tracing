import numpy as np
from scipy.integrate import quad

"""
A class to compute various cosmological quantities
"""


class cosmology:
    def __init__(self, hubble_parameter, OmegaM, OmegaL):
        """
        Set up by storing various critical cosmological parameters and constants

        Arguments:
          -hubble_parameter : Hubble parameter in units of 100 km/s/Mpc
          -OmegaM           : Matter density in units of the critical density
          -OmegaL           : Dark energy density in units of the critical density
        """

        # Inputs
        self.hubp = hubble_parameter
        self.OmgM = OmegaM
        self.OmgL = OmegaL

        # Constants
        self.Gcgs = 6.67408e-8
        self.GYRcgs = 3.15576e16
        return

    def Ez(self, redshift):
        return np.sqrt(self.OmgM * (1.0 + redshift) ** 3.0 + self.OmgL)

    def matter_density_at_z(self, redshift):
        return self.OmgM * (1.0 + redshift) ** 3.0 / self.Ez(redshift) ** 2.0

    def critcal_density(self, redshift):
        return (
            self.Ez(redshift)
            * self.Ez(redshift)
            * 3.0
            * (100.0 * self.hubp / 3.0856776e19) ** 2.0
            / (8.0 * np.pi * self.Gcgs)
        )

    def mean_density(self, redshift):
        return self.critcal_density(0.0) * self.OmgM * (1.0 + redshift) ** 3.0

    def delta_vir(self, redshift):
        BN98x = self.matter_density_at_z(redshift) - 1.0
        return 18.0 * np.pi ** 2.0 + 82.0 * BN98x - 39.0 * BN98x ** 2.0

    def density_threshold(self, redshift, delta, mode):
        if not isinstance(delta, int) and not isinstance(delta, float):
            print("\nERROR:\n--> DELTA must be INT or FLOAT!\nEXITING!")
            quit()

        if mode == "CRIT":
            return delta * self.critcal_density(redshift)
        elif mode == "MEAN":
            return delta * self.mean_density(redshift)
        elif mode == "VIR":
            return self.delta_vir(redshift) * self.critcal_density(redshift)
        else:
            print("\nERROR:\n--> Threshold unrecognised: {0}\nEXITING!".format(mode))
            quit()
        return

    def t_hubble(self, redshift):
        return np.sqrt(3.0 / (8.0 * np.pi * self.Gcgs * self.critcal_density(redshift)))

    def t_hubble_Gyr(self, redshift):
        return self.t_hubble(redshift) / self.GYRcgs

    def t_dynamic(self, redshift, delta=200.0, mode="CRIT"):
        return (
            2.0 ** 1.5
            * self.t_hubble(redshift)
            * (
                self.density_threshold(redshift, delta, mode)
                / self.critcal_density(redshift)
            )
            ** -0.5
        )

    def t_dynamic_Gyr(self, redshift, delta=200.0, mode="CRIT"):
        return self.t_dynamic(redshift, delta=delta, mode=mode) / self.GYRcgs

    def age(self, redshift):
        return self._exact_age(redshift)

    def _exact_age(self, redshift):
        return self.t_hubble_Gyr(0.0) * self._integral_OneOverEz1pz(redshift, np.inf)

    def _integral_OneOverEz1pz(self, z_min, z_max=np.inf):
        def integrand(z):
            return 1.0 / self.Ez(z) / (1.0 + z)

        return self._integral(integrand, z_min, z_max)

    def _integral(self, integrand, z_min, z_max):
        min_is_array = self.isArray(z_min)
        max_is_array = self.isArray(z_max)
        use_array = min_is_array or max_is_array

        if use_array and not min_is_array:
            z_min_use = np.array([z_min] * len(z_max))
        else:
            z_min_use = z_min

        if use_array and not max_is_array:
            z_max_use = np.array([z_max] * len(z_min))
        else:
            z_max_use = z_max

        if use_array:
            if min_is_array and max_is_array and len(z_min) != len(z_max):
                raise Exception(
                    "If both z_min and z_max are arrays, they need to have the same size."
                )
            integ = np.zeros_like(z_min_use)
            for i in range(len(z_min_use)):
                integ[i] = quad(integrand, z_min_use[i], z_max_use[i])[0]
        else:
            integ = quad(integrand, z_min, z_max)[0]
        return integ

    def isArray(self, var):
        try:
            dummy = iter(var)
        except TypeError:
            is_array = False
        else:
            is_array = not isinstance(var, dict)
        return is_array
