import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from solution import get_city_area_data


# ============================================================
# Load dataset
# ============================================================

train_x = np.loadtxt(
    "train_x.csv",
    delimiter=",",
    skiprows=1
)

train_y = np.loadtxt(
    "train_y.csv",
    delimiter=",",
    skiprows=1
)

test_x = np.loadtxt(
    "test_x.csv",
    delimiter=",",
    skiprows=1
)


# ============================================================
# Extract coordinates
# ============================================================

train_coordinates, train_residential_flags, \
test_coordinates, test_residential_flags = get_city_area_data(
    train_x,
    test_x
)


# ============================================================
# Input and target
#
# We use only latitude as the input feature.
#
# train_coordinates[:, 0] -> latitude
# train_coordinates[:, 1] -> longitude
# train_y                 -> pollution
# ============================================================

X = train_x
y = train_y


# ============================================================
# Select training points
# ============================================================

rng = np.random.RandomState(1)

training_indices = rng.choice(
    np.arange(y.size),
    size=6,
    replace=False
)

X_train = X#[training_indices]
y_train = y#[training_indices]


# ============================================================
# Observation noise
# ============================================================

noise_std = 0.75

y_train_noisy = y_train + rng.normal(
    loc=0.0,
    scale=noise_std,
    size=y_train.shape
)


# ============================================================
# Gaussian Process
# ============================================================

kernel = 1 * RBF(
    length_scale=1.0,
    length_scale_bounds=(1e-2, 1e2)
)

gaussian_process = GaussianProcessRegressor(
    kernel=kernel,
    alpha=noise_std**2,
    n_restarts_optimizer=0
)

gaussian_process.fit(
    X_train,
    y_train_noisy
)


# ============================================================
# Points used for plotting
# ============================================================

X_plot = np.linspace(
    X.min(),
    X.max(),
    500
).reshape(-1, 1)


# ============================================================
# GP prediction
# ============================================================

mean_prediction, epistemic_std = gaussian_process.predict(
    X_plot,
    return_std=True
)


# ============================================================
# Uncertainties
# ============================================================

# Aleatoric uncertainty
aleatoric_std = noise_std

# Total predictive uncertainty
predictive_std = np.sqrt(
    epistemic_std**2 + aleatoric_std**2
)


# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(11, 6))


# Total uncertainty: epistemic + aleatoric
plt.fill_between(
    X_plot.ravel(),
    mean_prediction - 1.96 * predictive_std,
    mean_prediction + 1.96 * predictive_std,
    color="tab:orange",
    alpha=0.20,
    label="Total uncertainty"
)


# Epistemic uncertainty
plt.fill_between(
    X_plot.ravel(),
    mean_prediction - 1.96 * epistemic_std,
    mean_prediction + 1.96 * epistemic_std,
    color="tab:blue",
    alpha=0.30,
    label="Epistemic uncertainty"
)


# GP mean
plt.plot(
    X_plot.ravel(),
    mean_prediction,
    color="tab:blue",
    linewidth=2,
    label="GP mean"
)


# Training observations


# ============================================================
# Labels
# ============================================================

plt.xlabel("Latitude")
plt.ylabel("Pollution")

plt.title(
    "Gaussian Process Regression: "
    "Epistemic vs Aleatoric Uncertainty"
)

plt.legend()
plt.grid(alpha=0.2)

plt.tight_layout()
plt.show()