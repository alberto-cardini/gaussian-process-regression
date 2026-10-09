import os
import typing
import pickle
import time
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

        self.kernels = [RBF() + WhiteKernel(noise_level=1.0),
                   RationalQuadratic() + WhiteKernel(noise_level=1.0),
                   ExpSineSquared(periodicity=10.0) + WhiteKernel(noise_level=1.0),
                   DotProduct(sigma_0=1.0)**2 + WhiteKernel(noise_level=1.0),
                   Matern() + WhiteKernel(noise_level=1.0)
                   ]
        self.k = 0.0
        self.modello = None

    def save_model(self, filepath: str = 'trained_model.pkl'):
        """
        Salva il modello addestrato e il parametro k su file.
        """
        with open(filepath, 'wb') as f:
            pickle.dump({'modello': self.modello, 'k': self.k}, f)
        print(f"\n[INFO] Modello salvato con successo in '{filepath}'")

    def load_model(self, filepath: str = 'trained_model.pkl'):
        """
        Carica un modello addestrato da file.
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.modello = data['modello']
            self.k = data['k']
        print(f"\n[INFO] Modello caricato con successo da '{filepath}'")

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

        # Salvataggio automatico del modello al termine dell'addestramento
        self.save_model('trained_model.pkl')

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


def calculate_cost(ground_truth: np.ndarray, predictions: np.ndarray, residential_flags: np.ndarray) -> float:
    assert ground_truth.ndim == 1 and predictions.ndim == 1 and ground_truth.shape == predictions.shape

    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL

    mask = (predictions < ground_truth) & [bool(residential_flag) for residential_flag in residential_flags]
    weights[mask] = COST_W_UNDERPREDICT

    return np.mean(cost * weights)


def is_inside_circle(coordinate, circle_parameters):
    return (coordinate[0] - circle_parameters[0])**2 + (coordinate[1] - circle_parameters[1])**2 < circle_parameters[2]**2


def determine_residential_flags(grid_coordinates):
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


def perform_extended_model_evaluation(model: Model, output_dir: str = '.'):
    print('Performing extended evaluation')

    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_grid = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)
    grid_residential_flags = determine_residential_flags(visualization_grid)

    predictions, gp_mean, gp_stddev = model.predict_pollution(visualization_grid, grid_residential_flags)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0

    fig, ax = plt.subplots()
    ax.set_title('Extended visualization of task 1')
    im = ax.imshow(predictions, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)

    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()


def get_city_area_data(train_x: np.ndarray, test_x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_coordinates = train_x[:, :2]
    train_residential_flags = train_x[:, 2]
    test_coordinates = test_x[:, :2]
    test_residential_flags = test_x[:, 2]

    assert train_coordinates.shape[0] == train_residential_flags.shape[0] and test_coordinates.shape[0] == test_residential_flags.shape[0]
    assert train_coordinates.shape[1] == 2 and test_coordinates.shape[1] == 2
    assert train_residential_flags.ndim == 1 and test_residential_flags.ndim == 1

    return train_coordinates, train_residential_flags, test_coordinates, test_residential_flags


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