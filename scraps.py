
    #------------------------------------------------------------------------
    #SPATIALLY DISCRETIZE SYSTEM
    g_eq, prefactors, x_bins = system_functions.linear_system_energies_prefactors(n_bins, *sys_params)

    P0 = system_functions.tpm_from_geqpf(g_eq, prefactors, kB, T, t_save) 
    if False:
        plt.imshow(P0)
        plt.xlabel("start state")
        plt.ylabel("end state")
        plt.show()

    seed_states = [np.argmin(g_eq)]
    print(f"starting in state {seed_states}")

    n_steps_unbiased = int(t_mol//t_save)
    min_to_ts = n_bins/2 - seed_states[0]-1/2
    sigma, t_sigma, t_s, dg_est = unbiased_simulation_set_mtd_params_2(P0, n_bins, min_to_ts, seed_states[0], n_steps_unbiased)
    #print(sigma, t_s, dg_est)
    #print(t_sigma)
    print("orig. sigma: ", sigma)
    #sigma = 6

    asymptotic_feature_height = 1 #kT
    deltaT = T*(dg_est/asymptotic_feature_height-1) # set omega to make the barrier flat to within half a kT
    omega = (1/n_parallel)*(dg_est/t_sigma) #energy per unit time

    #deltaT = deltaT/4