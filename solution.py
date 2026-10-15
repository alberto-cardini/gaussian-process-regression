import os
import typing
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RationalQuadratic, WhiteKernel
from scipy.linalg import cho_solve


# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation

# Cost function constants
COST_W_UNDERPREDICT = 50.0
COST_W_NORMAL = 1.0

TILE_GRID = 16
TILE_MARGIN = 0.03
HYPER_MARGIN = None
TILE_OFFSET = 0.0
TILE_NU = 1.5
KERNEL_SET = "m15_m05"
TILE_HYPER_POINTS = 3500

TEST_NOISE_FRACTION = 0.35
K = 0.9
BORDER_WIDTH = 0.0118
BORDER_SHIFT = 0.0
BORDER_PROFILE = "step"
BORDER_K = 0.85
K_DEEP = None
DEEP_DEPTH = 0.05

HOLDOUT_SIZE = 1000
PREDICT_BATCH_SIZE = 5000

LENGTH_SCALE_BOUNDS = (5e-3, 1.0)
AMPLITUDE_BOUNDS = (1e-2, 1e3)
NOISE_BOUNDS = (1e-4, 1.0)

RESIDENTIAL_CIRCLES = np.array([[0.5488135, 0.71518937, 0.17167342],
                                [0.79915856, 0.46147936, 0.1567626 ],
                                [0.26455561, 0.77423369, 0.10298338],
                                [0.6976312,  0.06022547, 0.04015634],
                                [0.31542835, 0.36371077, 0.17985623],
                                [0.15896958, 0.11037514, 0.07244247],
                                [0.82099323, 0.09710128, 0.08136552],
                                [0.41426299, 0.0641475,  0.04442035],
                                [0.09394051, 0.5759465,  0.08729856],
                                [0.84640867, 0.69947928, 0.04568374],
                                [0.23789282, 0.934214,   0.04039037],
                                [0.82076712, 0.90884372, 0.07434012],
                                [0.09961493, 0.94530153, 0.04755969],
                                [0.88172021, 0.2724369,  0.04483477],
                                [0.9425836,  0.6339977,  0.04979664]])


def print_progress_bar(current, total, kernel_name, elapsed=None, bar_length=30):
    """
    Prints a progress bar for the training of the kernels.
    """
    progress = current / total
    filled = int(bar_length * progress)
    bar = "█" * filled + "░" * (bar_length - filled)

    elapsed_str = ""
    if elapsed is not None:
        elapsed_str = f" | {elapsed:.2f}s"

    print(
        f"\rTraining kernels | {bar} {progress * 100:6.2f}% "
        f"| {kernel_name}{elapsed_str}",
        end="",
        flush=True
    )

    if current == total:
        print()


def distance_inside_zone(coordinates: np.ndarray) -> np.ndarray:
    centers, radii = RESIDENTIAL_CIRCLES[:, :2], RESIDENTIAL_CIRCLES[:, 2]
    dist = np.linalg.norm(coordinates[:, None, :2] - centers[None, :, :], axis=2)
    depth = radii[None, :] - dist
    return np.max(np.where(depth > 0, depth, -np.inf), axis=1)


KERNEL_SETS = {
    "m15_m25": ("matern15", "matern25"),
    "m15_rq": ("matern15", "rq"),
    "m15_m05": ("matern15", "matern05"),
    "all": ("matern05", "matern15", "matern25", "rq"),
}


def candidate_kernel(name: str):
    if name == "rq":
        base = RationalQuadratic(length_scale=0.05, alpha=1.0, length_scale_bounds=LENGTH_SCALE_BOUNDS,
                                 alpha_bounds=(1e-2, 1e3))
    else:
        base = Matern(0.05, LENGTH_SCALE_BOUNDS, nu={"matern05": 0.5, "matern15": 1.5, "matern25": 2.5}[name])
    return ConstantKernel(1.0, AMPLITUDE_BOUNDS) * base + WhiteKernel(0.005, NOISE_BOUNDS)


def loo_error(gp: GaussianProcessRegressor) -> float:
    inverse = cho_solve((gp.L_, True), np.eye(len(gp.alpha_)))
    residuals = gp.alpha_.ravel() / np.diag(inverse)
    return float(np.mean(residuals ** 2))


class LocalGP:

    def __init__(self, nu: float):
        self.nu = nu
        self.kind = "matern"
        self.rq_alpha = None
        self.hyperparams = None

    def optimize(self, X: np.ndarray, y: np.ndarray, n_points: int):
        if len(X) > n_points:
            keep = np.random.default_rng(0).permutation(len(X))[:n_points]
            X, y = X[keep], y[keep]
        variance = np.var(y)
        if KERNEL_SET is None:
            kernel = (ConstantKernel(1.0, AMPLITUDE_BOUNDS) * Matern(0.05, LENGTH_SCALE_BOUNDS, nu=self.nu)
                      + WhiteKernel(0.005, NOISE_BOUNDS))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                fitted = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0).fit(X, y).kernel_
            self.at_bound = any(issubclass(w.category, ConvergenceWarning) for w in caught)
            self.hyperparams = (fitted.k1.k1.constant_value * variance,
                                float(fitted.k1.k2.length_scale),
                                fitted.k2.noise_level * variance)
            return self
        best = None
        for name in KERNEL_SETS[KERNEL_SET]:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                gp = GaussianProcessRegressor(kernel=candidate_kernel(name), normalize_y=True, random_state=0).fit(X, y)
            error = loo_error(gp)
            if best is None or error < best[0]:
                best = (error, name, gp.kernel_, any(issubclass(w.category, ConvergenceWarning) for w in caught))
        _, self.name, fitted, self.at_bound = best
        base = fitted.k1.k2
        if self.name == "rq":
            self.kind = "rq"
            self.rq_alpha = float(base.alpha)
        else:
            self.kind = "matern"
            self.nu = float(base.nu)
        self.hyperparams = (fitted.k1.k1.constant_value * variance,
                            float(base.length_scale),
                            fitted.k2.noise_level * variance)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray):
        amplitude, length_scale, noise = self.hyperparams
        if self.kind == "rq":
            base = RationalQuadratic(length_scale=length_scale, alpha=self.rq_alpha,
                                     length_scale_bounds="fixed", alpha_bounds="fixed")
        else:
            base = Matern(length_scale, "fixed", nu=self.nu)
        kernel = ConstantKernel(amplitude, "fixed") * base + WhiteKernel(noise, "fixed")
        self.offset = y.mean()
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False, optimizer=None).fit(X, y - self.offset)
        return self

    def predict(self, X: np.ndarray):
        mean = np.empty(len(X))
        std = np.empty(len(X))
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            for start in range(0, len(X), PREDICT_BATCH_SIZE):
                stop = start + PREDICT_BATCH_SIZE
                mean[start:stop], std[start:stop] = self.gp.predict(X[start:stop], return_std=True)
        return mean + self.offset, std, np.full(len(X), self.hyperparams[2])


class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict_pollution methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        self.rng = np.random.default_rng(seed=0)
        self.tiles = []

    @staticmethod
    def features(coordinates: np.ndarray) -> np.ndarray:
        return np.asarray(coordinates, dtype=float)[:, :2]

    @staticmethod
    def columns() -> int:
        return TILE_GRID + 1 if TILE_OFFSET > 0 else TILE_GRID

    @staticmethod
    def tile_of(X: np.ndarray) -> np.ndarray:
        n = Model.columns()
        shifted = X + TILE_OFFSET / TILE_GRID
        column = np.minimum((shifted[:, 0] * TILE_GRID).astype(int), n - 1)
        row = np.minimum((shifted[:, 1] * TILE_GRID).astype(int), n - 1)
        return column + n * row

    @staticmethod
    def region(X: np.ndarray, tile: int, margin: float = None) -> np.ndarray:
        margin = TILE_MARGIN if margin is None else margin
        n = Model.columns()
        column, row = tile % n, tile // n
        shifted = X + TILE_OFFSET / TILE_GRID
        low = np.array([column, row]) / TILE_GRID - margin
        high = np.array([column + 1, row + 1]) / TILE_GRID + margin
        return np.all((shifted >= low) & (shifted < high), axis=1)

    def fit_tiles(self, X: np.ndarray, y: np.ndarray):
        for tile, local in enumerate(self.tiles):
            mask = self.region(X, tile)
            local.fit(X[mask], y[mask])

    def predict_tiles(self, X: np.ndarray):
        mean = np.empty(len(X))
        std = np.empty(len(X))
        noise = np.empty(len(X))
        tiles = self.tile_of(X)
        for tile, local in enumerate(self.tiles):
            idx = np.where(tiles == tile)[0]
            if len(idx):
                mean[idx], std[idx], noise[idx] = local.predict(X[idx])
        return mean, std, noise

    @staticmethod
    def decide(mean, std, noise_var, residential_flags, coordinates) -> np.ndarray:
        predictions = mean.copy()
        residential = np.asarray(residential_flags) == 1
        noise = noise_var[residential]
        latent_var = np.maximum(std[residential] ** 2 - noise, 0.0)
        variance = latent_var + TEST_NOISE_FRACTION * noise
        depth = distance_inside_zone(np.asarray(coordinates)[residential])
        k_point = np.full(len(variance), K)
        if K_DEEP is not None:
            k_point[depth >= DEEP_DEPTH] = K_DEEP
        shift = k_point * np.sqrt(variance)
        if BORDER_WIDTH > 0:
            extra = BORDER_SHIFT + BORDER_K * np.sqrt(variance)
            if BORDER_PROFILE == "linear":
                shift = shift + extra * np.clip(1.0 - depth / BORDER_WIDTH, 0.0, 1.0)
            else:
                shift = shift + np.where(depth < BORDER_WIDTH, extra, 0.0)
        predictions[residential] += shift
        return predictions

    # Don't change the name or the signature of this function
    def fit_model(self, train_coordinates: np.ndarray, train_pollution_targets: np.ndarray, train_residential_flags: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_coordinates: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_pollution_targets: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        :param train_residential_flags: Binary variable denoting whether the 2D training point is in the residential area (1) or not (0)
        """
        X = self.features(train_coordinates)
        y = np.asarray(train_pollution_targets, dtype=float)
        flags = np.asarray(train_residential_flags)
        n_total = len(X)
        n_tiles = self.columns() ** 2

        print("\n" + "=" * 80)
        print("                         GP TRAINING")
        print("=" * 80)
        print(f"Total dataset          : {n_total}")
        print(f"Tiles                  : {TILE_GRID}x{TILE_GRID}, margin {TILE_MARGIN}, offset {TILE_OFFSET}, hyperparameter margin {HYPER_MARGIN if HYPER_MARGIN is not None else TILE_MARGIN}")
        print(f"Kernel                 : Matern nu {TILE_NU} + white noise, hyperparameters per tile" if KERNEL_SET is None else f"Kernel                 : chosen per tile by leave-one-out among {KERNEL_SETS[KERNEL_SET]}")
        print(f"Test noise fraction    : {TEST_NOISE_FRACTION}")
        print(f"k                      : {K}" + (f" (deep points, depth >= {DEEP_DEPTH}: {K_DEEP})" if K_DEEP is not None else ""))
        print(f"Border width / shift   : {BORDER_WIDTH} / {BORDER_SHIFT} ({BORDER_PROFILE}), extra k {BORDER_K}")
        print("=" * 80 + "\n")

        start_all = time.perf_counter()
        self.tiles = []
        for tile in range(n_tiles):
            start = time.perf_counter()
            mask = self.region(X, tile, HYPER_MARGIN)
            if mask.sum() < 10:
                mask = self.region(X, tile, TILE_MARGIN + 0.05)
            self.tiles.append(LocalGP(TILE_NU).optimize(X[mask], y[mask], TILE_HYPER_POINTS))
            print_progress_bar(tile + 1, n_tiles, f"hyperparameters tile {tile} ({mask.sum()} points)",
                               time.perf_counter() - start)
        table = np.array([local.hyperparams for local in self.tiles])
        print(f"Hyperparameters ({time.perf_counter() - start_all:.1f}s):")
        for name, column in zip(["amplitude", "length scale", "noise variance"], table.T):
            print(f"  {name:15s}: min {column.min():.4f} | median {np.median(column):.4f} | max {column.max():.4f}")
        print(f"  tiles with a hyperparameter at its bound: {sum(local.at_bound for local in self.tiles)} / {n_tiles}")
        if KERNEL_SET is not None:
            for name in KERNEL_SETS[KERNEL_SET]:
                print(f"  tiles choosing {name}: {sum(getattr(local, 'name', '') == name for local in self.tiles)}")

        n_holdout = min(HOLDOUT_SIZE, int(0.1 * n_total))
        permutation = self.rng.permutation(n_total)
        idx_holdout, idx_pool = permutation[:n_holdout], permutation[n_holdout:]
        self.fit_tiles(X[idx_pool], y[idx_pool])
        mean, std, noise = self.predict_tiles(X[idx_holdout])
        y_hold, flags_hold = y[idx_holdout], flags[idx_holdout]
        print("=" * 80)
        print("HOLDOUT (noisy labels, hyperparameters from all data)")
        print(f"Noise sd estimated     : {np.sqrt(noise.mean()):.3f}")
        print(f"MSE of the mean        : {np.mean((y_hold - mean) ** 2):.4f}")
        print(f"Chosen rule            : {calculate_cost(y_hold, self.decide(mean, std, noise, flags_hold, X[idx_holdout]), flags_hold):.4f}")
        print(f"No margin              : {calculate_cost(y_hold, mean, flags_hold):.4f}")
        print("=" * 80)

        start = time.perf_counter()
        self.fit_tiles(X, y)
        print(f"Final model on {n_total} points ({time.perf_counter() - start:.1f}s)")
        mean, std, noise = self.predict_tiles(X)
        print("=" * 80)
        print("FINAL MODEL ON ALL DATA (training points, noisy labels)")
        print(f"MSE of the mean        : {np.mean((y - mean) ** 2):.4f}")
        print(f"Chosen rule            : {calculate_cost(y, self.decide(mean, std, noise, flags, X), flags):.4f}")
        print(f"No margin              : {calculate_cost(y, mean, flags):.4f}")
        print("=" * 80)

    # Don't change the name or the signature of this function
    def predict_pollution(self, test_coordinates: np.ndarray, test_residential_flags: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of city_areas.
        :param test_coordinates: city_areas as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param test_residential_flags: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """
        X_test = self.features(test_coordinates)
        gp_mean, gp_std, noise = self.predict_tiles(X_test)
        predictions = self.decide(gp_mean, gp_std, noise, test_residential_flags, X_test)
        return predictions, gp_mean, gp_std


# You don't have to change this function
def calculate_cost(ground_truth: np.ndarray, predictions: np.ndarray, residential_flags: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param ground_truth: Ground truth pollution levels as a 1d NumPy float array
    :param predictions: Predicted pollution levels as a 1d NumPy float array
    :param residential_flags: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
    :return: Total cost of all predictions as a single float
    """
    assert ground_truth.ndim == 1 and predictions.ndim == 1 and ground_truth.shape == predictions.shape

    # Unweighted cost
    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL

    # Case i): underprediction
    mask = (predictions < ground_truth) & [bool(residential_flag) for residential_flag in residential_flags]
    weights[mask] = COST_W_UNDERPREDICT

    # Weigh the cost and return the average
    return np.mean(cost * weights)


# You don't have to change this function
def is_inside_circle(coordinate, circle_parameters):
    """
    Checks if a coordinate is inside a circle.
    :param coordinate: 2D coordinate
    :param circle_parameters: 3D coordinate of the circle center and its radius
    :return: True if the coordinate is inside the circle, False otherwise
    """
    return (coordinate[0] - circle_parameters[0])**2 + (coordinate[1] - circle_parameters[1])**2 < circle_parameters[2]**2


# You don't have to change this function
def determine_residential_flags(grid_coordinates):
    """
    Determines the city_area index for each coordinate in the visualization grid.
    :param grid_coordinates: 2D coordinates of the visualization grid
    :return: 1D array of city_area indexes
    """
    # Circles coordinates
    circles = np.array([[0.5488135, 0.71518937, 0.17167342],
                    [0.79915856, 0.46147936, 0.1567626 ],
                    [0.26455561, 0.77423369, 0.10298338],
                    [0.6976312,  0.06022547, 0.04015634],
                    [0.31542835, 0.36371077, 0.17985623],
                    [0.15896958, 0.11037514, 0.07244247],
                    [0.82099323, 0.09710128, 0.08136552],
                    [0.41426299, 0.0641475,  0.04442035],
                    [0.09394051, 0.5759465,  0.08729856],
                    [0.84640867, 0.69947928, 0.04568374],
                    [0.23789282, 0.934214,   0.04039037],
                    [0.82076712, 0.90884372, 0.07434012],
                    [0.09961493, 0.94530153, 0.04755969],
                    [0.88172021, 0.2724369,  0.04483477],
                    [0.9425836,  0.6339977,  0.04979664]])

    residential_flags = np.zeros((grid_coordinates.shape[0],))

    for i,coordinate in enumerate(grid_coordinates):
        residential_flags[i] = any([is_inside_circle(coordinate, circ) for circ in circles])

    return residential_flags


# Don't change the name or the signature of this function
def perform_extended_model_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_grid = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)
    grid_residential_flags = determine_residential_flags(visualization_grid)

    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.predict_pollution(visualization_grid, grid_residential_flags)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0

    # Plot the actual predictions
    fig, ax = plt.subplots()
    ax.set_title('Extended visualization of task 1')
    im = ax.imshow(predictions, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()


# You do not need to change the name or the signature of this function; convenience helper used by main()
def get_city_area_data(train_x: np.ndarray, test_x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the city_area information from the training and test features.
    :param train_x: Training features
    :param test_x: Test features
    :return: Tuple of (training features' 2D coordinates, training features' city_area information,
        test features' 2D coordinates, test features' city_area information)
    """
    train_coordinates = train_x[:, :2]
    train_residential_flags = train_x[:, 2]
    test_coordinates = test_x[:, :2]
    test_residential_flags = test_x[:, 2]

    assert train_coordinates.shape[0] == train_residential_flags.shape[0] and test_coordinates.shape[0] == test_residential_flags.shape[0]
    assert train_coordinates.shape[1] == 2 and test_coordinates.shape[1] == 2
    assert train_residential_flags.ndim == 1 and test_residential_flags.ndim == 1

    return train_coordinates, train_residential_flags, test_coordinates, test_residential_flags


# you don't have to change this function
def main():
    # Load the training dataset and test features
    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Extract the city_area information
    train_coordinates, train_residential_flags, test_coordinates, test_residential_flags = get_city_area_data(train_x, test_x)

    # Fit the model
    print('Training model')
    model = Model()
    model.fit_model(train_coordinates, train_y, train_residential_flags)

    # Predict on the test features
    print('Predicting on test features')
    predictions = model.predict_pollution(test_coordinates, test_residential_flags)

    if EXTENDED_EVALUATION:
        perform_extended_model_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()