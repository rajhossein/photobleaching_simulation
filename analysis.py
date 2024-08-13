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

def calculate_intensity_drop(weighted_photons_emitted_df, num_pixels):
    intensity_drop_df = pd.DataFrame()
    intensity_drop_df['Time'] = weighted_photons_emitted_df['Time']

    for pixel in range(num_pixels):
        intensity_drop = weighted_photons_emitted_df[f'Weighted Pixel {pixel}'].shift(1) - weighted_photons_emitted_df[f'Weighted Pixel {pixel}']
        intensity_drop_df[f'Pixel{pixel}'] = intensity_drop

    intensity_drop_df = intensity_drop_df.dropna().reset_index(drop=True)
    intensity_drop_df.to_csv('data/intensity_drop.csv', header=True, index=False)
    return intensity_drop_df

def calculate_correlations(intensity_drop_df, num_pixels):
    correlation_df = pd.DataFrame({'Time': intensity_drop_df['Time']})
    pixel_pairs = [(2, 0), (2, 1), (2, 3), (2, 4)]

    for pair in pixel_pairs:
        correlation_over_time = []
        pixel_a, pixel_b = pair
        for i in range(len(correlation_df['Time'])):
            intensity_drop_a = intensity_drop_df[f'Pixel{pixel_a}'][:i+1]
            intensity_drop_b = intensity_drop_df[f'Pixel{pixel_b}'][:i+1]
            if len(intensity_drop_a) > 1 and len(intensity_drop_b) > 1:  # Ensure there are enough points to calculate correlation
                correlation = intensity_drop_a.corr(intensity_drop_b)
            else:
                correlation = np.nan
            correlation_over_time.append(correlation)
        
        correlation_df[f'Correlation {pixel_a}-{pixel_b}'] = correlation_over_time

    correlation_df.to_csv('data/correlations.csv', header=True, index=False)
    return correlation_df

def plot_results(combined_df, weighted_photons_emitted_df, intensity_drop_df, correlation_df, output_file):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Plot the original data
    axes[0].plot(combined_df['Time'], combined_df['Photons Emitted pixel 0'], label='Photons Emitted', alpha=0.5)
    axes[0].set_ylabel('Number of photons emitted')
    axes[0].set_xlabel('Time')

    # Plot for Intensity Drop
    axes[1].plot(intensity_drop_df['Time'], intensity_drop_df['Pixel0'], label='', alpha=0.5)
    axes[1].set_ylabel('Intensity Drop')
    axes[1].set_xlabel('Time')

    # Plot the correlations
    for pair in [(2, 0), (2, 1), (2, 3), (2, 4)]:
        Pixel_a, Pixel_b = pair
        axes[2].plot(correlation_df['Time'], correlation_df[f'Correlation {Pixel_a}-{Pixel_b}'], label=f'Correlation {Pixel_a}-{Pixel_b}', alpha=0.5)
    axes[2].set_ylabel('Correlation')
    axes[2].set_xlabel('Time')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

