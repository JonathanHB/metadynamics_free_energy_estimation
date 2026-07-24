import numpy as np
from scipy.linalg import expm
import deeptime as deeptime
import matplotlib.pyplot as plt



def energies_and_prefactors_to_tpm(g_eq, prefactors, kB, T, lag_time):
    """
    Generate a markov state model discrete-time transition probability matrix (TPM) 
    from equilibrium free energies and a continuous-time prefactor matrix. 
    This is done by constructing an instantaneous rate constant matrix 
    and then calculating the corresponding TPM at the specified lag time by matrix exponentiation.
    
    Parameters
    ----------
    g_eq: 1d array of float, of length n
        Equilibrium free energies of n states. 
        This method respects gauge symmetry so a constant energy offset does not affect the output.
    prefactors: n x n matrix of float
        The instantaneous rate prefactors for the transition between each pair of states. 
        This should be symmetric if detailed balance is to be obeyed.
        The diagonal entries are not used (see comments in function), and all others should be nonnegative.
    kB: float
        Boltzmann's constant
    T: float
        Temperature
    lag_time: float
        Lag time used to generate a discrete time transition probability matrix

    Returns
    -------
    tpm: n x n matrix of float
        The transition probability matrix for the system described by the parameters, discretized at the lag time.
        All entries should be nonnegative.
    """

    #matrix of equilibrium delta G values for each pair of states
    dg_eq = g_eq - g_eq[:,None] 

    #matrix of instantaneous rate constants for each pair of states
    rates = np.multiply(prefactors, np.exp(dg_eq/(2*kB*T))) 

    #Each row of an instantaneous rate constant matrix holds the 
    # coefficients of the differential equation for the rate of change in the population of one state.
    # The off-diagonal elements are positive coefficients describing flux into the state from each other state.
    # The diagonal element is a negative coefficient describing total flux out of the state into all other states.
    #Each column of an instantaneous rate matrix describes what happens to probability starting in one state.
    # the off-diagonal elements are positive coefficients describing flux out of the state into each other state.
    # the diagonal element describes the resulting loss of probability from the starting state
    # To satisfy conservation of matter, the entries in each column must sum to zero,
    # so the diagonal element is just the negative of the sum of all the other elements in the column.
    
    #The previous lines of code do not generate correct diagonal elements
    # so we set them to 0 to enable summing of the off-diagonal rate constants to get the total rate out of each state
    np.fill_diagonal(rates, 0)
    #total rate out of each state
    diag_rates = np.sum(rates, axis=0) 
    #set the self-transition rates to the total rate out of each state
    np.fill_diagonal(rates, -diag_rates) 
    
    #calculate the discrete time transition probability matrix for the given lag time by exponentiating the rate matrix
    tpm = expm(rates*lag_time)

    return tpm



def spatially_discretize_linear_system(system, n):
    """
    Spatially discretize a 1D system

    Parameters
    ----------
    system: system object as defined in msm_systems.py
        It must have the following methods and variable:
        
        bounds(): tuple of two floats
            The left and right limits of the CV. The latter must be large than the former.
        G(): float : float
            free energy as a function of the coordinate (spatially continuous)
        D: float
            diffusion coefficient, assumed to be uniform

    n: int
        the number of discrete bins to make

    Returns
    -------
    g_eq: 1d array of float, of length n
        Equilibrium free energies of n states. 
        Not normalized to make the discrete partition function equal to any particular value.
    prefactors: n x n matrix of float
        The instantaneous rate prefactors for the transition between each pair of states. 
        This is symmetric to obey detailed balance.
        The diagonal entries are zero (see comments in function), and all others are nonnegative.
    """

    #construct bins and calculate their equilibrium free energies
    x_bins = np.linspace(system.bounds()[0], system.bounds()[1], n)
    g_eq = system.G(x_bins)

    #calculate spatially discrete, temporally continuous rate prefactors
    # for the above bin spacing assuming a uniform diffusion coefficient
    
    #the spacing between neighboring bins
    a = (system.bounds()[1] - system.bounds()[0])/n   

    #for transitions between neighboring states
    #The spatially discrete prefactor shrinks quadratically with bin spacing 
    # because the mean squared deviation of a diffusive system is linear in time.
    neighbor_prefactor = system.D/a**2

    #fill in the rate prefactor matrix
    prefactors = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            #All transitions in a temporally continuous 1d system must be between adjacent bins 
            # because that is the only way the system can get to more distant ones, 
            # so all entries not adjacent to the main diagonal should be zero
            if abs(i-j)==1:
                prefactors[i,j] = neighbor_prefactor

            #The diagonal entries of a prefactor matrix do not mean anything,
            # and the formula above is undefined along the diagonal, so the diagonal is left as zero.
    
    return g_eq, prefactors, x_bins



def spatially_temporally_discretize_linear_system(system, n, kB, T, lag_time):
    """
    A wrapper function for spatially_discretize_linear_system() and energies_and_prefactors_to_tpm()
    """
    g_eq, prefactors, xb = spatially_discretize_linear_system(system, n)
    tpm = energies_and_prefactors_to_tpm(g_eq, prefactors, kB, T, lag_time)

    return tpm



def timescale_vs_spatial_discretization(system, kB, T, lag_time, n_range, tolerance=0.05):
    """
    Calculate first implied timescale as a function of the number of bins
    to determine how many bins we need to represent the system's dynamics

    Parameters
    ----------
    system: system object as defined in msm_systems.py
        It must have the following methods and variable:
        
        bounds(): tuple of two floats
            The left and right limits of the CV. The latter must be large than the former.
        G(): float : float
            free energy as a function of the coordinate (spatially continuous)
        D: float
            diffusion coefficient, assumed to be uniform

    kB: float
        Boltzmann's constant
    T: float
        Temperature
    lag_time: float
        Lag time used to generate a discrete time transition probability matrix
    n_range: tuple of two ints
        minimum and maximum number of bins to test respectively
    tolerance: float
        maximum allowed deviation between implied timescale at n_i bins
        and implied timescale at the maximum number of bins tested
        as a fraction of the latter. 
        This is used to determine the number of bins at which the system's
        first implied timescale is approximately bin independent.

    Returns
    -------
    n_min: int
        Minimum number of bins n_i at which the implied timescale is roughly the asymptotic values, estimated as:
        tolerance > (implied timescale at n_i bins - implied timescale at the maximum number of bins tested)/implied timescale at the maximum number of bins tested
    """

    #-------------------------------------------------
    #calculate implied timescale vs number of bins

    n_all = []
    first_implied_timescales = []

    for n in range(n_range[0], n_range[1]):
        tpm = spatially_temporally_discretize_linear_system(system, n, kB, T, lag_time)

        eigenvalues = deeptime.markov.tools.analysis.eigenvalues(tpm)
        if np.imag(eigenvalues[1]) != 0:
            print(f"error: complex eigenvalue {eigenvalues[1]}")
            return 0

        n_all.append(n)
        first_implied_timescales.append(-lag_time/np.log(np.real(eigenvalues[1])))

    #-------------------------------------------------
    #calculate minimum number of bins required for implied timescale to be near its asymptote 
    # (how near being determined by tolerance)

    n_min = -1
    for ni, ti in zip(n_all, first_implied_timescales):
        if abs(first_implied_timescales[-1]-ti)/first_implied_timescales[-1] < tolerance:
            n_min = ni
            print(f"implied timescale converged to within {100*tolerance}% of the value calculated for {n_range[1]} bins at {ni} bins")
            break

    #-------------------------------------------------
    #plot implied timescales vs number of bins

    plt.clf()
    plt.plot(n_all, first_implied_timescales)
    plt.xlim(0,n_range[1])
    plt.xlabel("number of bins")
    plt.ylabel("first implied timescale")

    plt.axvline(n_min, linestyle="dashed", color = "black")
    plt.axhline(first_implied_timescales[-1], linestyle="dotted", color = "grey")
    plt.show()

    return n_min
