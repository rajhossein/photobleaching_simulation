import numpy as np
import pandas as pd
from random import choices, expovariate, seed as random_seed
from analysis import run_gillespie, save_csv, combine_pixels, calculate_weights, calculate_weighted_photons, calculate_intensity_drop, calculate_correlations, plot_results

# Simulation parameters
params = {"kDA": 10**15, "kAD": 2*10**8, "kAT": 5*10**8, "kTD": 10., "kTB": 0.0005}
sigma = 1.0

propensities = (
    lambda u, p, t: p["kDA"] * u[0],  # D -> A
    lambda u, p, t: p["kAD"] * u[1],  # A -> D
    lambda u, p, t: p["kAT"] * u[1],  # A -> T
    lambda u, p, t: p["kTD"] * u[2],  # T -> D
    lambda u, p, t: p["kTB"] * u[2]   # T -> B
)

reactions = (
    lambda u, p, t: [u[0] - 1, u[1] + 1, u[2], u[3], u[4]],      # D -> A
    lambda u, p, t: [u[0] + 1, u[1] - 1, u[2], u[3], u[4] + 1],  # A -> D
    lambda u, p, t: [u[0], u[1] - 1, u[2] + 1, u[3], u[4]],      # A -> T
    lambda u, p, t: [u[0] + 1, u[1], u[2] - 1, u[3], u[4]],      # T -> D
    lambda u, p, t: [u[0], u[1], u[2] - 1, u[3] + 1, u[4]]       # T -> B
)

# Run simulations
num_pixels = 5
delta_t = 0.5
t_end = 2000
N = 100

for pixel in range(num_pixels):
    seed_value = 43 + pixel
    random_seed(seed_value)
    output_file = f'data/N{N}_kTD{params["kTD"]}_kTB{params["kTB"]}_pixel{pixel}.csv'
    ts, us = run_gillespie(propensities, reactions, params, [N, 0, 0, 0, 0], t_end)
    save_csv(ts, us, delta_t, output_file)

# Analyze data
combined_df = combine_pixels(num_pixels, N, params)
central_weight, left_neighbor_weight, right_neighbor_weight = calculate_weights(sigma)
weighted_photons_emitted_df = calculate_weighted_photons(combined_df, num_pixels, central_weight, left_neighbor_weight, right_neighbor_weight)
intensity_drop_df = calculate_intensity_drop(weighted_photons_emitted_df, num_pixels)
correlation_df = calculate_correlations(intensity_drop_df, num_pixels)

# Plot and save results
plot_results(combined_df, weighted_photons_emitted_df, intensity_drop_df, correlation_df, 'data/Fig1.png')

