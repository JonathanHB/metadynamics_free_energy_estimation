
# def build_tpm_doublewell(n, A, xl, D, lag_time, kB, T, plot=False):
#     #bins = np.linspace(-xl,xl,n)
#     # plt.plot(bins, system_functions.unit_double_well(bins, A))
#     # plt.show()

#     def f(x):
#         return unit_double_well(x, A)
    
#     g_eq, prefactors = linear_system_energies_prefactors(n, D, -xl,xl, f)
#     if plot:
#         plt.plot(g_eq)
#         plt.show()
#         # plt.imshow(prefactors)
#         # plt.show()

#     P = tpm_from_geqpf(g_eq, prefactors, kB, T, lag_time)
#     if plot:
#         plt.imshow(P)
#         plt.xlabel("start state")
#         plt.ylabel("end state")
#     #msm = deeptime.markov.msm.MarkovStateModel(P.transpose())

#     return P #msm, 


#THIS FUNCTION CHECKS THAT THE RELAXATION TIMES ARE INDEPENDENT OF HOW FINELY THE SYSTEM IS DISCRETIZED