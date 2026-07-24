import numpy as np
from scipy.linalg import expm
import deeptime as deeptime
import matplotlib.pyplot as plt



def free_energies_and_prefactors_to_tpm(g_eq, prefactors, kB, T, lag_time):
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


#SPATIALLY DISCRETIZE A 1D SYSTEM
# n = number of bins
# D = diffusion coefficient
# x0,x1 = the ends of the system
# f = the free energy function
def linear_system_energies_prefactors(n, D, x0,x1, f):

    #G_EQ
    x_bins = np.linspace(x0,x1,n)
    g_eq = f(x_bins)

    #G_TS
    a = (x1-x0)/n   #the spacing between neighboring bins
    discrete_rate_prefactor = D/a**2
    prefactors = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if abs(i-j) < 2:
                prefactors[i,j] = discrete_rate_prefactor
            else:
                prefactors[i,j] = 0

    return g_eq, prefactors, x_bins


def unit_double_well(x, A):
    return A*(x**4 - x**2)


def build_tpm_doublewell(n, A, xl, D, lag_time, kB, T, plot=False):
    #bins = np.linspace(-xl,xl,n)
    # plt.plot(bins, system_functions.unit_double_well(bins, A))
    # plt.show()

    def f(x):
        return unit_double_well(x, A)
    
    g_eq, prefactors = linear_system_energies_prefactors(n, D, -xl,xl, f)
    if plot:
        plt.plot(g_eq)
        plt.show()
        # plt.imshow(prefactors)
        # plt.show()

    P = tpm_from_geqpf(g_eq, prefactors, kB, T, lag_time)
    if plot:
        plt.imshow(P)
        plt.xlabel("start state")
        plt.ylabel("end state")
    #msm = deeptime.markov.msm.MarkovStateModel(P.transpose())

    return P #msm, 


#THIS FUNCTION CHECKS THAT THE RELAXATION TIMES ARE INDEPENDENT OF HOW FINELY THE SYSTEM IS DISCRETIZED
def timescale_vs_discretization():

    A = 20
    xl = 1

    D = 0.05

    lag_time = 1
    T = 1
    kB = 1

    n_all = []
    eig_all = []

    for n in range(2, 100):
        P = build_tpm_doublewell(n, A, xl, D, lag_time, kB, T)
        eigenvalues = deeptime.markov.tools.analysis.eigenvalues(P)
        if np.imag(eigenvalues[1]) != 0:
            print(f"error: complex eigenvalue {eigenvalues[1]}")
            return 0

        n_all.append(n)
        eig_all.append(-lag_time/np.log(np.real(eigenvalues[1])))

    plt.clf()
    plt.plot(n_all, eig_all)
    plt.xlim(0,100)
    plt.ylim(0,200)
    plt.xlabel("number of bins")
    plt.ylabel("first implied timescale")
