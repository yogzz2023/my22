import numpy as np
from scipy.stats import chi2

# Generate random measurements and track predictions
np.random.seed(42)  # Set seed for reproducibility
num_measurements = 5
num_tracks = 3

measurements = np.random.rand(num_measurements, 3)  # Random measurements for range, azimuth, elevation
tracks = np.random.rand(num_tracks, 3)  # Random track predictions for range, azimuth, elevation

# Measurement covariance matrix (assuming identity for simplicity)
S = np.eye(3)
chi_squared_threshold = chi2.ppf(0.95, df=3)

# Calculate residuals and perform chi-squared gating
association_list = []
for i, track in enumerate(tracks):
    for j, measurement in enumerate(measurements):
        residual = measurement - track
        nis = residual.T @ np.linalg.inv(S) @ residual
        if nis <= chi_squared_threshold:
            association_list.append((i, j))

print("Association List:", association_list)

# Cluster determination (simplified)
clusters = [{track: [report for t, report in association_list if t == track]} for track, _ in association_list]
print("Clusters:", clusters)

# Hypothesis generation (simplified)
hypotheses = []
for cluster in clusters:
    for track in cluster.keys():
        for report in cluster[track]:
            hypotheses.append({track: report})

print("Hypotheses:", hypotheses)

# Calculate probabilities (simplified example)
hypothesis_probs = {i: 1/len(hypotheses) for i in range(len(hypotheses))}
print("Hypothesis Probabilities:", hypothesis_probs)