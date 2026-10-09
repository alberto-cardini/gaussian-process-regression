import os
import typing
from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as gp
import matplotlib.pyplot as plt
from matplotlib import cm


# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = True
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation

# Cost function constants
COST_W_UNDERPREDICT = 50.0
COST_W_NORMAL = 1.0

import time


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


def print_prediction_table(
    coordinates,
    residential_flags,
    predictions,
    gp_mean,
    gp_std
):
    """
    Prints GP predictions in a readable table.
    """

    print("\n" + "=" * 95)
    print("                           GP PREDICTIONS")
    print("=" * 95)

    print(
        f"{'ID':>4} | "
        f"{'X':>8} {'Y':>8} | "
        f"{'Residential':>11} | "
        f"{'GP Mean':>10} | "
        f"{'GP Std':>10} | "
        f"{'Prediction':>11}"
    )

    print("-" * 95)

    for i in range(len(predictions)):
        residential = "YES" if residential_flags[i] else "NO"

        print(
            f"{i:4d} | "
            f"{coordinates[i, 0]:8.4f} "
            f"{coordinates[i, 1]:8.4f} | "
            f"{residential:>11} | "
            f"{gp_mean[i]:10.4f} | "
            f"{gp_std[i]:10.4f} | "
            f"{predictions[i]:11.4f}"
        )

    print("=" * 95)


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

        # TODO: Add custom initialization for your model here if necessary

        self.kernels = [RBF() + WhiteKernel(noise_level=1.0),
                   RationalQuadratic() + WhiteKernel(noise_level=1.0),
                   ExpSineSquared(periodicity=10.0) + WhiteKernel(noise_level=1.0),
                   DotProduct(sigma_0=1.0)**2 + WhiteKernel(noise_level=1.0),
                   Matern() + WhiteKernel(noise_level=1.0)
                   ]
        self.k = 0.0
        self.modello = None

    # Don't change the name or the signature of this function
    def fit_model(
        self,
        train_coordinates: np.ndarray,
        train_pollution_targets: np.ndarray,
        train_residential_flags: np.ndarray
    ):
        X_tr = np.column_stack(
            (train_coordinates, train_residential_flags)
        )

        indici_mescolati = np.random.permutation(len(X_tr))
        taglio = int(len(X_tr) * 0.8)

        indici_train = indici_mescolati[int(taglio * 0.5):taglio]
        indici_val = indici_mescolati[taglio:]

        X_train = X_tr[indici_train]
        X_val = X_tr[indici_val]

        Y_train = train_pollution_targets[indici_train]
        Y_val = train_pollution_targets[indici_val]

        models = []

        print("\n" + "=" * 80)
        print("                         GP TRAINING")
        print("=" * 80)
        print(f"Training samples   : {len(X_train)}")
        print(f"Validation samples : {len(X_val)}")
        print(f"Kernels to test    : {len(self.kernels)}")
        print("=" * 80 + "\n")

        total_kernels = len(self.kernels)

        for i, kernel in enumerate(self.kernels, start=1):

            kernel_name = str(kernel)

            start_time = time.perf_counter()

            modello = gp(
                kernel=kernel,
                normalize_y=True
            )

            print_progress_bar(
                i - 1,
                total_kernels,
                kernel_name
            )

            modello.fit(X_train, Y_train)

            elapsed = time.perf_counter() - start_time

            models.append(modello)

            print_progress_bar(
                i,
                total_kernels,
                kernel_name,
                elapsed
            )

        print("\nTraining completed.\n")

        valori_k = np.arange(0, 3.01, 0.1)
        residenziale = X_val[:, 2] == 1

        best_model = None
        best_cost = float('inf')
        best_k = None

        results = []

        for model in models:

            mu, sd = model.predict(
                X_val,
                return_std=True
            )

            model_best_cost = float('inf')
            model_best_k = None

            for k in valori_k:

                predizioni = mu.copy()

                predizioni[residenziale] += (
                    k * sd[residenziale]
                )

                cost = calculate_cost(
                    Y_val,
                    predizioni,
                    X_val[:, 2]
                )

                if cost < model_best_cost:
                    model_best_cost = cost
                    model_best_k = k

                if cost < best_cost:
                    best_cost = cost
                    best_model = model
                    best_k = k

            results.append(
                (
                    model.kernel_,
                    model_best_k,
                    model_best_cost
                )
            )

        # ---------------------------------------------------------
        # Print dei risultati
        # ---------------------------------------------------------

        print("=" * 90)
        print("                         MODEL RESULTS")
        print("=" * 90)

        print(
            f"{'#':>3} | "
            f"{'Kernel':<42} | "
            f"{'Best k':>8} | "
            f"{'Validation cost':>17}"
        )

        print("-" * 90)

        for i, (kernel, k, cost) in enumerate(results, start=1):

            kernel_string = str(kernel)

            # Evita che kernel troppo lunghi distruggano la tabella
            if len(kernel_string) > 42:
                kernel_string = kernel_string[:39] + "..."

            print(
                f"{i:3d} | "
                f"{kernel_string:<42} | "
                f"{k:8.2f} | "
                f"{cost:17.4f}"
            )

        print("-" * 90)

        print(f"BEST MODEL")
        print(f"Kernel : {best_model.kernel_}")
        print(f"k      : {best_k:.2f}")
        print(f"Cost   : {best_cost:.4f}")

        print("=" * 90)

        self.k = best_k
        self.modello = best_model

    # Don't change the name or the signature of this function
    def predict_pollution(
        self,
        test_coordinates: np.ndarray,
        test_residential_flags: np.ndarray
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        X_test = np.column_stack(
            (test_coordinates, test_residential_flags)
        )

        gp_mean, gp_std = self.modello.predict(
            X_test,
            return_std=True
        )

        predictions = gp_mean.copy()

        residenziale = test_residential_flags == 1

        predictions[residenziale] += (
            self.k * gp_std[residenziale]
        )

        print_prediction_table(
            test_coordinates,
            test_residential_flags,
            predictions,
            gp_mean,
            gp_std
        )

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
    train_coordinates = np.zeros((train_x.shape[0], 2), dtype=float)
    train_residential_flags = np.zeros((train_x.shape[0],), dtype=bool)
    test_coordinates = np.zeros((test_x.shape[0], 2), dtype=float)
    test_residential_flags = np.zeros((test_x.shape[0],), dtype=bool)

    #TODO: Extract the city_area information from the training and test features

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
    print(predictions)

    if EXTENDED_EVALUATION:
        perform_extended_model_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()
