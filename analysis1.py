import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import choices, expovariate
from scipy.stats import norm

def run_gillespie(propensities, reactions, params, u_0, t_end, tstart=0):
    t, u = tstart, np.asarray(u_0)
    us, ts = [u.copy()], [t]
    while t < t_end:
        ps = [f(u, params, t) for f in propensities]
        p_total = sum(ps)

        if p_total > 0:
            dt = expovariate(p_total)
            u = np.asarray(choices(reactions, weights=ps)[0](u, params, t))
            t += dt
        else:
            dt = 0.001
            t += dt
        us.append(u.copy())
        ts.append(t)
    return np.array(ts), np.array(us)

def save_csv(ts, us, delta_t, output_file):
    num_intervals = len(us[:, 4])
    num_photons_emitted_store = []
    time_points = []
    dark_molecules = []
    active_molecules = []
    triplet_molecules = []
    bleached_molecules = []
    current_time = 0
    current_index = 0
    next_time = current_time + delta_t

    while next_time <= ts[-1] - delta_t:
        for j in range(current_index, num_intervals):
            if ts[j] >= next_time:
                next_index = j
                break
        
        num_photons_emitted = us[next_index, 4] - us[current_index, 4]
        num_photons_emitted_store.append(num_photons_emitted)
        time_points.append(next_time)

        dark_molecules.append(us[current_index, 0])
        active_molecules.append(us[current_index, 1])
        triplet_molecules.append(us[current_index, 2])
        bleached_molecules.append(us[current_index, 3])

        current_time = next_time
        current_index = next_index
        next_time = current_time + delta_t

    photon_df = pd.DataFrame({
        'Time': time_points,
        'Photons Emitted': num_photons_emitted_store,
        'Dark Molecule': dark_molecules,
        'Active Molecule': active_molecules,
        'Triplet Molecule': triplet_molecules,
        'Bleached Molecule': bleached_molecules
    })

    photon_df.to_csv(output_file, header=True, index=False)

def combine_pixels(num_pixels, N, params):
    combined_df = pd.DataFrame()

    for pixel in range(num_pixels):
        input_file = f'data/N{N}_kTD{params["kTD"]}_kTB{params["kTB"]}_pixel{pixel}.csv'
        df = pd.read_csv(input_file)
        df[f'Photons Emitted pixel {pixel}'] = df['Photons Emitted']
        df = df[['Time', f'Photons Emitted pixel {pixel}']]
        if combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, on='Time')

    combined_df.to_csv('data/combined_photon_data.csv', index=False)
    return combined_df


def calculate_weights(sigma):
    central_weight = norm.cdf(1, loc=0, scale=sigma) - norm.cdf(-1, loc=0, scale=sigma)  # Center pixel
    remaining_weight = 1 - central_weight
    left_neighbor_weight = right_neighbor_weight = remaining_weight / 2

    return central_weight, left_neighbor_weight, right_neighbor_weight

def calculate_weighted_photons(combined_df, num_pixels, central_weight, left_neighbor_weight, right_neighbor_weight):
    weighted_photons_emitted = []

    for i, row in combined_df.iterrows():
        weighted_row = []
        for pixel in range(num_pixels):
            left_pixel = (pixel - 1) % num_pixels  # Handle periodic boundary
            right_pixel = (pixel + 1) % num_pixels  # Handle periodic boundary
            central_value = row[f'Photons Emitted pixel {pixel}']
            left_value = row[f'Photons Emitted pixel {left_pixel}']
            right_value = row[f'Photons Emitted pixel {right_pixel}']
            weighted_value = (
                central_weight * central_value +
                left_neighbor_weight * left_value +
                right_neighbor_weight * right_value
            )
            weighted_row.append(weighted_value)
        weighted_photons_emitted.append(weighted_row)

    weighted_photons_emitted_df = pd.DataFrame(weighted_photons_emitted, columns=[f'Weighted Pixel {i}' for i in range(num_pixels)])
    weighted_photons_emitted_df['Time'] = combined_df['Time']
    weighted_photons_emitted_df.to_csv('data/weighted_photons_emitted.csv', header=True, index=False)
    return weighted_photons_emitted_df

