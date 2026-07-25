import numpy as np
import msm_system_construction
import deeptime as deeptime



def sigma_and_timescales_from_unbiased_simulation(trj, trj_frame_interval, n_bins):
    """
    Calculate frame distribution width and diffusion timescales from an unbiased simulation trajectory
    These are usually used to set MTD tunable parameters
    
    Parameters
    ----------
    trj: 1d numpy array or list
        an unbiased simulation trajectory
        specifically the discrete state index of an unbiased simulation at evenly spaced time intervals
    trj_frame_interval: float
        the time interval between frames in the trajectory, assumed to be uniform
    n_bins: int
        the number of discrete states in the system 

    Returns
    -------
    sigma: float
        the standard deviation of the distribution of frames in the unbiased simulation, in units of bin width
        these frames are assumed to be trapped in the harmonic bottom of a well such that they are normally distributed
        in which case the standard deviation captures basically all the interesting information about the shape of the distribution
    well_diffusion_timescale: float
        the time to diffuse across the starting well, in units of trj_frame_interval
    system_diffusion_timescale: float
        the time to diffuse across the whole system, in units of trj_frame_interval

    """

    #----------------------------
    #width of starting well
    sigma = np.std(trj)

    #----------------------------
    #diffusion rate
    bins = np.arange(-0.5, n_bins+0.5, 1)

    #equilibrium populations in each state, used for reweighting below
    eq_pops = np.histogram(trj, bins=bins)[0]

    #transition counts matrix
    #TODO implement bin-independent version for continuous trajectories if/when we try brownian dynamics
    c = np.zeros((n_bins, n_bins))
    for i in range(len(trj)-1):
        c[trj[i+1], trj[i]] += 1

    #transition probability matrix
    p = c/np.sum(c, axis=0, keepdims=True)

    #reweight to get diffusion coefficient by canceling out 
    # the effect of equilibrium free energy differences on transition probabilities
    thermodynamic_reweight_matrix = np.outer(1/np.sqrt(eq_pops), np.sqrt(eq_pops))
    reweighted_tpm = np.multiply(p, thermodynamic_reweight_matrix)
    if False:
        plt.imshow(reweighted_tpm)
        plt.show()

    #calculate the average probability of transitioning to a neighboring state, 
    # assuming a uniform diffusion coefficient and bin width
    prefactors = []
    counts = [] #for weighting
    for i in range(n_bins-1):
        if c[i, i+1] > 0: #avoid adding nans
            prefactors.append(reweighted_tpm[i, i+1])
            counts.append(c[i, i+1])
        if c[i+1, i] > 0: #avoid adding nans
            prefactors.append(reweighted_tpm[i+1, i])
            counts.append(c[i+1, i])

    #in units of 1/trj_frame_interval
    prefactor = np.average(prefactors, weights=counts)
    print(f"Estimated discrete time rate prefactor between adjacent states: {prefactor} per frame save interval")


    #TODO we're missing a factor of something here; work out analytically what this time 
    # should be using the formulas used to spatially discretize the system to begin with and the MSD relation
    #----------------------------
    #timescale of diffusing the width of the starting well
    # in units of 1/trj_frame_interval
    well_diffusion_timescale = sigma**2/prefactor

    #----------------------------
    #timescale of diffusing across the entire CV range
    # in units of trj_frame_interval
    system_diffusion_timescale = (n_bins/2)**2/prefactor
    print(f"Estimated whole-system diffusive timescale: {system_diffusion_timescale} frame save intervals")


    return sigma, well_diffusion_timescale, system_diffusion_timescale



################################################################################################
#                     SET METADYNAMICS PARAMETERS FROM UNBIASED SIMULATION
################################################################################################


def unbiased_simulation_to_mtd_params(tpm, n_bins, init_state, lag_time, n_steps=-1):
    """
    Run an unbiased simulation,
    calculate the first implied timescale of the MSM in effigy of experimental MFPT data, 
    and use the results to set the four tunable parameters for well-tempered metadynamics.

    Parameters
    ----------
    tpm: n x n matrix of float
        The transition probability matrix for the system described by the parameters, discretized at the lag time.
        TODO: should we use a deeptime MSM object instead of passing around a numpy TPM matrix?
    n_bins: int
        the number of discrete states in the system 
    init_state: int
        the MSM state in which the system starts, normally the lowest energy one
    lag_time: float
        the lag time at which the MSM was constructed
    n_steps: int
        how many steps to run the unbiased simulation for, should be much smaller than the MSM's MFPT
        the default value of -1 means to use 1% of the MFPT

    Returns
    -------
    sigma: float
        the standard deviation of the MTD gaussian
    omega_g: float
        the area (in units of energy * number of bins) of each MTD potential, 
        not including the tempering factor from well-tempered metadynamics
    tau_g: float
        the gaussian deposition interval, in units of the MSM lag time
    delta_T: float
        the metadynamics temperature factor, in units of the physical temperature T

    """

    #-------------------------------------------------------
    #calculate implied timescale from MSM

    implied_timescale = msm_system_construction.first_implied_timescale(tpm)

    if n_steps == -1:
        n_steps = int(implied_timescale//100)
        print(f"running for {n_steps} steps")

    #-------------------------------------------------------
    #SIMULATE

    msm = deeptime.markov.msm.MarkovStateModel(tpm.transpose())
    #much faster than even my clean looking numpy implementation
    #dt = 1 means that trajectory frames are saved every MSM lag time
    #note that kT does not explicitly appear here because it is already baked into the TPM
    trj = msm.simulate(n_steps=n_steps, start = init_state, dt=1) 

    #-------------------------------------------------------
    #calculate accessible well width and timescales
    well_frame_std, well_diffusion_timescale, system_diffusion_timescale = sigma_and_timescales_from_unbiased_simulation(trj=trj, trj_frame_interval=lag_time, n_bins=n_bins)

    #-------------------------------------------------------
    #estimate free energy barrier based on how much longer the MFPT is than the diffusion time
    est_barrier = np.log(implied_timescale/system_diffusion_timescale) #in units of kT
    print("Estimated barrier from MFPT and diffusion in initial state: ", est_barrier, " kT")

    #-------------------------------------------------------
    #CALCULATE MTD PARAMETERS
    # The gaussian width, potential deposition rate, and potential deposition interval are set according to:
    # Reviews in Computational Chemistry Volume 28
    # Editor(s):Abby L. Parrill, Kenny B. Lipkowitz
    # First published:10 April 2015
    # Print ISBN:9781118407776 |Online ISBN:9781118889886 |DOI:10.1002/9781118889886
    # Chapter 1
    # Free-Energy Calculations with Metadynamics: Theory and Practice
    # by Giovanni Bussi and Davide Branduardi
    # Section: Metadynamics How-To

    sigma = well_frame_std

    omega_g = 0.1

    tau_g = well_diffusion_timescale

    delta_T = 2*np.log(implied_timescale/system_diffusion_timescale) - 1


    return sigma, omega_g, tau_g, delta_T