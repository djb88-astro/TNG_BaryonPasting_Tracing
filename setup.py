import setuptools

setuptools.setup(
    name="tng_baryonpasting_tracing",
    version="3.0.1",
    description="Trace TNG halo properties for Baryon Pasters collaboration",
    url="https://github.com/djb88-astro/TNG_BaryonPasting_Tracing",
    author="David J Barnes",
    author_email="djbarnes@mit.edu",
    packages=setuptools.find_packages(),
    license="MIT",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy", "scipy", "h5py", "mpi4py"],
)
