import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

#from https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    The weights must sum to 1.

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))



#see the following stackoverfow posts: 
# https://stackoverflow.com/questions/22548813/python-color-map-but-with-all-zero-values-mapped-to-black
# https://stackoverflow.com/questions/56062299/how-to-add-axis-labels-to-imshow-plots-in-python
# https://stackoverflow.com/questions/13384653/imshow-extent-and-aspect

#TODO move to another file and import

def plot_masked_energies(data, xlims, ylims, plot_shape, aspect_ratio, vmax, labels):

    # mask 'bad' regions with no sampling
    masked_rfe = np.ma.masked_where(data == 0, data)

    #set color mapping for regions with sampling
    cmap = mpl.colormaps.get_cmap("viridis").copy()

    #set color for 'bad' regions with no sampling
    cmap.set_bad(color='grey')

    plt.figure(figsize=plot_shape)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    im = plt.imshow(masked_rfe, interpolation='none', cmap=cmap, extent = [xlims[0], xlims[1], ylims[0], ylims[1]], aspect = aspect_ratio, vmax=vmax, origin="lower")
    plt.show()