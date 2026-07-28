import numpy as np
import msm_system_construction
import deeptime as deeptime
import matplotlib.pyplot as plt


#Written by Gemini on 7/27/26 with prompt: 
# "write me a function to keep only the nan-free rows and columns of a numpy matrix"
# and then edited after I realized that that prompt was wrong
def remove_nan_rows_cols_strict(matrix):
    """Drops any row and any column that contains at least one NaN."""
    # Mask for rows that are not all NaN
    row_mask = ~np.isnan(matrix).all(axis=1)
    
    # Mask for columns that are not all NaN
    col_mask = ~np.isnan(matrix).all(axis=0)

    #keep only states with both incoming and outgoing transitions
    #since the output TPM must be square
    combined_mask = row_mask & col_mask
    
    # Filter rows first, then filter columns
    return matrix[combined_mask][:, combined_mask]


def sigma_and_timescales_from_unbiased_simulation(trj, n_bins):
    """
    Calculate frame distribution width and diffusion timescales from an unbiased simulation trajectory
    These are usually used to set MTD tunable parameters
    
    Parameters
    ----------
    trj: 1d numpy array or list
        an unbiased simulation trajectory
        specifically the discrete state index of an unbiased simulation at evenly spaced time intervals
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

    #1. ------------SIGMA----------------
    #width of starting well
    sigma = np.std(trj)


    #2. ------------BUILD MSM FROM INPUT TRAJECTORY---------------
    
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


    #3. ------------WELL DIFFUSION TIMESCALE----------------
    #timescale of equilibrating in the starting well

    #collect the portion of the empirical TPM which has ergodic* sampling
    #*this function removes states with no in or no out transitions but is not a rigorous test of connectedness or detailed balance
    starting_well_tpm = remove_nan_rows_cols_strict(p)
    if False:
        plt.imshow(starting_well_tpm)
        plt.colorbar()
        plt.show()

    #TODO there are other ways to calculate this using autocorrelation decay. Do they agree?
    # in units of frame intervals
    well_diffusion_timescale = msm_system_construction.first_implied_timescale(starting_well_tpm)


    #4. ------------SYSTEM DIFFUSION TIMESCALE----------------
    #Estimate what the system MFPT would be if motion were purely diffusive

    #Extract the diffusive component of the TPM by cancelling out
    # the effect of equilibrium free energy differences on transition probabilities
    # however this is only reliable for transitions close to the main diagonal
    thermodynamic_reweight_matrix = np.outer(1/np.sqrt(eq_pops), np.sqrt(eq_pops))
    reweighted_tpm = np.multiply(p, thermodynamic_reweight_matrix)
    if False:
        plt.imshow(p, interpolation='nearest')
        plt.colorbar()
        plt.show()
        plt.imshow(reweighted_tpm, interpolation='nearest')
        plt.colorbar()
        plt.show()
        plt.imshow(starting_well_tpm, interpolation='nearest')
        plt.colorbar()
        plt.show()

    #collect the prefactors at each distance from the main diagonal and average them
    prefactors_by_diagonal_distance = [[] for _ in range(n_bins)]
    counts_by_diagonal_distance = [[] for _ in range(n_bins)]

    for i in range(n_bins):
        for j in range(n_bins):
            if not np.isnan(reweighted_tpm[i,j]):
                prefactors_by_diagonal_distance[int(abs(i-j))].append(reweighted_tpm[i,j])
                counts_by_diagonal_distance[int(abs(i-j))].append(c[i,j])

    mean_prefactors_by_diagonal_distance = np.zeros(n_bins)

    for i in range(n_bins):
        if np.sum(counts_by_diagonal_distance[i])>0:
            mean_prefactors_by_diagonal_distance[i] = np.average(prefactors_by_diagonal_distance[i], weights=counts_by_diagonal_distance[i])

    #we only actually use the prefactor from the transitions adjacent to the diagonal
    #because the reweighting of the others is unreliable
    implied_prefactors_from_neighbors = np.zeros(n_bins)
    implied_prefactors_from_neighbors[1] = mean_prefactors_by_diagonal_distance[1]

    for i in range(2, n_bins):
        implied_prefactors_from_neighbors[i] = implied_prefactors_from_neighbors[1]**i

    # plt.plot(mean_prefactors_by_diagonal_distance)
    # plt.plot(implied_prefactors_from_neighbors)
    # plt.show()

    #print(f"replacing diagonal mean prefactor {mean_prefactors_by_diagonal_distance[0]}, which can't be effectively reweighted")


    #construct a synthetic TPM using the averages from the real one
    #to describe a purely diffusive system with the same size and diffusion coefficient as the real one
    
    #the width of the synthetic landscape is reduced to avoid including high-energy regions at the edges which 
    #have little effect on the real MFPT but would affect the diffusive one
    n_synth_bins = int(max((n_bins-2*np.mean(trj), n_bins-2*(n_bins-np.mean(trj)))))

    flat_landscape_tpm = np.zeros((n_synth_bins, n_synth_bins))

    for k in range(-n_synth_bins+1, n_synth_bins):
        flat_landscape_tpm += np.diag([implied_prefactors_from_neighbors[int(abs(k))]] * (n_synth_bins - int(abs(k))), k=k)


    # plt.plot(np.sum(flat_landscape_tpm, axis=0))
    # plt.show()

    # plt.imshow(flat_landscape_tpm, interpolation='nearest')
    # plt.colorbar()
    # plt.show()

    #normalize 
    # because of how the transition probabilities were extracted from real data, 
    # the system should be close to normalized to start with, 
    # but nothing in the construction procedure guarantees it will be exactly normalized
    flat_landscape_tpm += np.diag(1 - np.sum(flat_landscape_tpm, axis=0))

    # plt.plot(np.sum(flat_landscape_tpm, axis=0))
    # plt.show()

    # plt.imshow(flat_landscape_tpm, interpolation='nearest')
    # plt.colorbar()
    # plt.show()

    print(np.trace(flat_landscape_tpm)/n_synth_bins)

    if False:
        plt.imshow(flat_landscape_tpm)
        plt.colorbar()
        plt.show()

    #timescale of diffusing across the CV range between the estimated main well positions
    # in units of frame intervals
    system_diffusion_timescale = msm_system_construction.first_implied_timescale(flat_landscape_tpm)
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
    #plot neighbor transition rates for debugging
    if False:
        plt.plot([tpm[i,i+1] for i in range(len(tpm)-1)])
        plt.title("actual uncorrected rates")
        plt.show()

    #-------------------------------------------------------
    #calculate implied timescale from MSM

    #in units of MSM lag time
    implied_timescale = msm_system_construction.first_implied_timescale(tpm)

    if n_steps == -1:
        n_steps = int(implied_timescale//1000)
        print(f"running for {n_steps} steps")

    #-------------------------------------------------------
    #SIMULATE

    #note that kT does not explicitly appear here because it is already baked into the TPM
    msm = deeptime.markov.msm.MarkovStateModel(tpm.transpose())

    #dt = 1 means that trajectory frames are saved every MSM lag time
    dt=1

    #much faster than even my clean looking numpy implementation
    trj = msm.simulate(n_steps = n_steps, start = init_state, dt = dt) 

    #-------------------------------------------------------
    #calculate accessible well width and timescales
    #timescales are returned in units of the frame interval, which is lag_time*dt
    well_frame_std, well_diffusion_timescale, system_diffusion_timescale = sigma_and_timescales_from_unbiased_simulation(trj=trj, n_bins=n_bins)

    #-------------------------------------------------------
    #estimate free energy barrier based on how much longer the MFPT is than the diffusion time
    #this is just for user information and is not actually used to calculate the function return data
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

    sigma = well_frame_std/3

    omega_g = 0.1

    #in real time units
    tau_g = well_diffusion_timescale*lag_time*dt

    #dt is included to convert the system diffusion timescale to units of lag time to match the numerator.
    delta_T = 2*np.log(implied_timescale/(system_diffusion_timescale*dt)) - 1 


    return (sigma, omega_g, tau_g, delta_T)