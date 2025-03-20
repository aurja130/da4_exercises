import scipy as sp
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

# import data and clean up NaN values

experimentally_measured_data = np.load('I_q_IPA_exp.npy')
theoretically_predicted_data = np.load('I_q_IPA_model.npy')

experimentally_measured_data = experimentally_measured_data[~np.isnan(experimentally_measured_data).any(axis=1)]
theoretically_predicted_data = theoretically_predicted_data[~np.isnan(theoretically_predicted_data).any(axis=1)]

# Define the scaling function
def scaling_function(x, a):
    return a * x

# Extract points and values
experimental_points = experimentally_measured_data[:, 0]
experimental_values = experimentally_measured_data[:, 1]
theoretical_points = theoretically_predicted_data[:, 0]
theoretical_values = theoretically_predicted_data[:, 1]

# Find the overlapping range
common_min = max(min(experimental_points), min(theoretical_points))
common_max = min(max(experimental_points), max(theoretical_points))

# Trim both datasets to the common range
experimental_mask = (experimental_points >= common_min) & (experimental_points <= common_max)
theoretical_mask = (theoretical_points >= common_min) & (theoretical_points <= common_max)

trimmed_experimental_points = experimental_points[experimental_mask]
trimmed_experimental_values = experimental_values[experimental_mask]
trimmed_theoretical_points = theoretical_points[theoretical_mask]
trimmed_theoretical_values = theoretical_values[theoretical_mask]

# Interpolate the theoretical data to get values at the same points as the experimental dataset
interp_function = interp1d(trimmed_theoretical_points, trimmed_theoretical_values, kind='linear', fill_value="extrapolate")
interpolated_theoretical_values = interp_function(trimmed_experimental_points)

# Use scipy.optimize.curve_fit to find the best scaling factor
popt, _ = curve_fit(scaling_function, interpolated_theoretical_values, trimmed_experimental_values)
scaling_factor = popt[0]

fig, (left, right) = plt.subplots(figsize=(12, 6), ncols=2)

left.plot(trimmed_experimental_points, trimmed_experimental_values, label='Experimental measurement', color='red')
left.legend(loc='upper left')
left.set_xlabel('Scattering vector')
left.set_ylabel('Scattering strength- experimental measurement')

left2 = left.twinx()
left2.plot(trimmed_experimental_points, interpolated_theoretical_values * scaling_factor, label='Scaled theoretical prediction', color='green')
left2.legend(loc='lower right')
left2.set_ylabel('Scattering strength- scaled theoretical prediction')

right.plot(trimmed_experimental_points, trimmed_experimental_values, label='Experimental measurement', color='red')
right.legend(loc='upper left')
right.set_xlabel('Scattering vector')
right.set_ylabel('Scattering strength- experimental measuremenright')

right2 = right.twinx()
right2.plot(trimmed_experimental_points, interpolated_theoretical_values, label='Unscaled theoretical prediction', color='blue')
right2.legend(loc='lower right')
right2.set_ylabel('Scattering strength- unscaled theoretical prediction')

fig.subplots_adjust(top=0.95, bottom=0.1, left=0.07, right=0.9, wspace=0.35)
plt.savefig('scaling_results.png')