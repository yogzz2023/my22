import numpy as np
from scipy.stats import chi2

# Generate 20 random measurements and tracks
np.random.seed(42)  # for reproducibility
measurements = np.random.rand(20, 2)  # 20 measurements, each with 2 dimensions
tracks = np.random.rand(20, 2)  # 20 tracks, each with 2 dimensions

# Measurement covariance matrix (assuming identity for simplicity)
S = np.eye(2)
chi_squared_threshold = chi2.ppf(0.95, df=2)

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

print()
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
print()
print("Hypothesis Probabilities:", hypothesis_probs)

