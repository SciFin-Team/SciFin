# Standard library imports
# /

# Third party imports
import itertools
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import cluster

# Local application imports
from scifin import timeseries as ts
from scifin import montecarlo as mc
from scifin import classifier as cl

# Generate two kinds of processes
N = 50
mc_MA = mc.generate_series(n=N, names_base="MA-", series_model=ts.moving_average,
                         start_date="2000-01-01", end_date="2020-01-01", frequency='M',
                         cst=1., order=2, coeffs=[0.5, 0.2], sigma=0.4)

mc_Garch = mc.generate_series(n=N, names_base="Garch-", series_model=ts.garch,
                         start_date="2000-01-01", end_date="2020-01-01", frequency='M',
                         cst=0.2, order_a=1, coeffs_a=[0.2], order_sig=1, coeffs_sig=[0.1])

# Combining them together
mc1 = [*mc_MA, *mc_Garch]

# Plot
ts.multi_plot(mc1)

# Shuffling elements
list_idx = [i for i in range(len(mc1))]
np.random.shuffle(list_idx)
mc1 = [mc1[i] for i in list_idx]

# Build Normalized DTW distance matrix based on 'abs'
obs1 = cl.dtw_distance_matrix_from_ts(mc1, window=10, mode='abs', normalize=True)

# Plot this DTW distance matrix
sns.heatmap(obs1, cmap='viridis')
plt.title("Initial DTW distance matrix")
plt.show()

# Fit the model
n_clust_range = range(2,4)
save_labels1, save_quality1 = cl.cluster_observation_matrix(X=obs1,
                                                            n_clust_range=n_clust_range,
                                                            model=cluster.KMeans,
                                                            verbose=False)

# Select the 2 clusters case
labels1 = save_labels1[2]
quality1 = save_quality1[2]

# Plot the clustered DTW distance matrix
clustered_obs1 = cl.reorganize_observation_matrix(X=obs1, labels=labels1)
sns.heatmap(clustered_obs1, cmap='viridis')
plt.title("Clustered DTW distance matrix")
plt.show()

# Find the most likely list of names ("expected truth")
name_perms = list(itertools.permutations(['MA', 'Ga']))
max_diag_sum = -1E10
for perm in name_perms:
    resu_matrix = cl.show_clustering_results(mc1, labels1, list(perm))
    diag_sum = np.trace(resu_matrix.values)
    if diag_sum > max_diag_sum:
        max_diag_sum = diag_sum
        expected_truth = list(perm)

# Show results
print()
clustering_results1 = cl.show_clustering_results(mc1, labels1, expected_truth)
print("Quality: ", quality1)
print("Trace: ", np.trace(clustering_results1.values))
print(clustering_results1)
print()

# Eventually plot the clusters content
cl.print_clusters_content(list_ts=mc1, labels=labels1)
