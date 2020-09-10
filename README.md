# TNG_BaryonPasting_Tracing

Code that traces IllustrisTNG halos from a selected snapshot back through time using the SubLink merger trees that are publically available.
It allows you to trace back to a given snapshot, set a requirement that the halos are sufficiently resolved (i.e. above a certain mass), and can handle when a halo's main progenitor goes missing.
Note, the properties recovered are only as good as the trees, which are only as good as Subfind allows (See Muldrew+ 2011 and papers that cite it).
