import numpy as np
import deeptime as deeptime
import matplotlib.pyplot as plt

import msm_system_construction
import utility


def mtd_simulation(n_parallel, t_mol, t_save, kT, mtd_params, system_data, macrostate_classifier_continuous, seed_states):

    #------------------------------------------------------------------------
    #INPUT HANDLING

    (n_bins, x_bins, g_eq, prefactors) = system_data

    (sigma, omega, t_mtd, deltaT) = mtd_params

    #convert ratios of times to numbers of steps
    if abs(t_mol/t_mtd % 1) != 0: 
        raise ValueError(f"molecular time ({t_mol}) must be an integer multiple of mtd gaussian addition interval ({t_mtd})")
    else:
        n_mtd_rounds = int(round(t_mol/t_mtd))

    if abs(t_mtd/t_save % 1) != 0: 
        raise ValueError(f"mtd gaussian addition interval ({t_mtd}) must be an integer multiple of frame save interval ({t_save})")
    else:
        n_frames_per_mtd = int(round(t_mtd/t_save))
        n_frames = int(round(t_mol/t_save))


    print("-----------------MTD params-----------------")
    print(f"sigma = {sigma}")
    print(f"deltaT = {deltaT}")
    print(f"omega = {omega}")

    #------------------------------------------------------------------------
    #DISCRETE GAUSSIAN KERNEL FOR MTD BIAS UPDATE

    #x coordinates for gaussian kernel, centered at 0, 
    # must be no longer than the number of states in the system for np.convolve to produce output of the correct length
    gkx = np.arange(1-np.floor(len(g_eq)/2), np.floor(len(g_eq)/2))
    gaussian_kernel = np.exp(-0.5 * (gkx / sigma) ** 2) #gaussian kernel
    gaussian_kernel /= np.sum(gaussian_kernel) #normalize

    plot_kernel = False
    if plot_kernel:
        wm, wstd = utility.weighted_avg_and_std(gkx, gaussian_kernel)

        plt.plot(gkx, gaussian_kernel)
        plt.xlabel("x")
        plt.ylabel("kernel")
        plt.show()
        print(f"Gaussian kernel mean: {wm}, std: {wstd}")

    #------------------------------------------------------------------------
    #MACROSTATE FREE ENERGY ESTIMATES

    def macrostate_classifier(x):
        return macrostate_classifier_continuous(x_bins[x])

    macrostate_1hot = macrostate_classifier([b for b in range(n_bins)])
    Z_0 = np.sum(np.exp(-g_eq[np.where(macrostate_1hot==0)]/(kT)))
    Z_1 = np.sum(np.exp(-g_eq[np.where(macrostate_1hot==1)]/(kT)))

    true_dg = -kT*np.log(Z_1/Z_0)

    #------------------------------------------------------------------------
    #RUN MTD

    mtd_potential = np.zeros((n_mtd_rounds+1, n_bins))
    #TODO what became of the start argument?
    trjs = np.zeros((n_mtd_rounds+1, n_frames_per_mtd, n_parallel), dtype=int) #+1 because the first frame is just the value of the 'start' argument

    for i_mtd in range(n_mtd_rounds):

        P = msm_system_construction.energies_and_prefactors_to_tpm(g_eq + mtd_potential[i_mtd], prefactors, kT, t_save) 
        #TODO: can the mtd potential be updated without taking a matrix exponential again 
        # (i.e. by multiplying by an outer product), or does that fail because the diagonal 
        # of the instantaneous rate matrix has to be constructed specially?
        msm = deeptime.markov.msm.MarkovStateModel(P.transpose())

        mtd_potential[i_mtd+1] = mtd_potential[i_mtd]

        for i in range(n_parallel):
            #propagate and record trajectory
            trjseg = msm.simulate(n_steps=n_frames_per_mtd+1, start = trjs[i_mtd, -1, i], dt=1)[1:] #the first frame is just the value of the 'start' argument
            
            trjs[i_mtd+1, :, i] = trjseg

            #update and record MTD bias

            #the heights of new gaussians centered at each location
            new_gaussian_heights = omega * np.multiply(np.sum(np.eye(n_bins)[trjseg], axis=0), np.exp(-mtd_potential[i_mtd]/(kB*deltaT)))
            delta_mtd_bias = np.convolve(new_gaussian_heights, gaussian_kernel, mode='same')
            mtd_potential[i_mtd+1] += delta_mtd_bias


    return trjs, mtd_potential, g_eq, true_dg, deltaT, n_mtd_rounds, n_frames_per_mtd
