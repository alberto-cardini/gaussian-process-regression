import os
import typing
from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from matplotlib import cm


# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation

# Cost function constants
COST_W_UNDERPREDICT = 50.0
COST_W_NORMAL = 1.0


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

        # --- GP locali: la mappa [0,1]x[0,1] viene divisa in una griglia 16x16 = 256 zone.
        # Ogni zona ha il suo GP con i suoi iperparametri (il campo non e' stazionario:
        # l'ampiezza cambia molto da una parte all'altra della mappa).
        self.tile_grid = 16
        # Ogni GP viene fittato sui punti della sua zona allargata di 0.03 per lato,
        # cosi' vicino al confine tra due zone ha comunque i vicini dell'altra parte.
        self.tile_margin = 0.03

        # --- Scelta del kernel per zona: ogni GP prova Matern 1.5 e Matern 0.5 e tiene
        # quello con l'errore leave-one-out piu' basso sui propri punti.
        self.kernel_nus = (1.5, 0.5)
        # Limiti degli iperparametri (relativi ai target normalizzati della zona).
        self.length_scale_bounds = (5e-3, 1.0)
        self.amplitude_bounds = (1e-2, 1e3)
        self.noise_bounds = (1e-4, 1.0)

        # --- Regola di decisione (solo per i punti residenziali, dove sottostimare costa 50x).
        # Il test non ha rumore di misura: nel margine entra l'incertezza sul campo latente
        # piu' solo una frazione del rumore stimato.
        self.test_noise_fraction = 0.35
        # Margine di base: k * sqrt(varianza latente + frazione * rumore).
        self.k = 0.925
        # Bordo delle zone residenziali: entro 0.0118 dal confine del cerchio il margine
        # aumenta di border_k * sqrt(varianza), cioe' in proporzione all'incertezza del punto.
        self.border_width = 0.0118
        self.border_k = 0.85

        # Cerchi delle zone residenziali (gli stessi di determine_residential_flags),
        # servono per calcolare la distanza di un punto dal confine della sua zona.
        self.circles = np.array([[0.5488135, 0.71518937, 0.17167342],
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

        # Per ogni zona: GP fittato, media sottratta e varianza del rumore (scala assoluta).
        self.tiles = []

    # Don't change the name or the signature of this function
    def fit_model(self, train_coordinates: np.ndarray, train_pollution_targets: np.ndarray, train_residential_flags: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_coordinates: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_pollution_targets: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        :param train_residential_flags: Binary variable denoting whether the 2D training point is in the residential area (1) or not (0)
        """

        # TODO: Fit your model here
        import warnings
        from scipy.linalg import cho_solve
        from sklearn.exceptions import ConvergenceWarning

        # Il GP vede solo le coordinate: il flag residenziale non entra nel kernel
        # (romperebbe la correlazione tra punti vicini ai due lati di un confine),
        # serve solo nella regola di decisione.
        X = np.asarray(train_coordinates, dtype=float)[:, :2]
        y = np.asarray(train_pollution_targets, dtype=float)
        n = self.tile_grid

        def region(points, tile, margin):
            # Punti dentro la zona `tile` allargata di `margin` per lato.
            column, row = tile % n, tile // n
            low = np.array([column, row]) / n - margin
            high = np.array([column + 1, row + 1]) / n + margin
            return np.all((points >= low) & (points < high), axis=1)

        def loo_error(gp):
            # Errore leave-one-out in forma chiusa per un GP: residuo_i = alpha_i / (K^-1)_ii.
            inverse = cho_solve((gp.L_, True), np.eye(len(gp.alpha_)))
            residuals = gp.alpha_.ravel() / np.diag(inverse)
            return float(np.mean(residuals ** 2))

        self.tiles = []
        for tile in range(n * n):
            mask = region(X, tile, self.tile_margin)
            if mask.sum() < 10:
                # Zona quasi vuota: si allarga la finestra per avere abbastanza punti.
                mask = region(X, tile, self.tile_margin + 0.05)
            X_tile, y_tile = X[mask], y[mask]
            variance = np.var(y_tile)

            # 1) Stima degli iperparametri per zona con la marginal likelihood
            #    (target normalizzati), una volta per ciascun kernel candidato.
            # 2) Scelta del kernel: quello con l'errore leave-one-out piu' basso.
            best = None
            for nu in self.kernel_nus:
                kernel = (ConstantKernel(1.0, self.amplitude_bounds)
                          * Matern(0.05, self.length_scale_bounds, nu=nu)
                          + WhiteKernel(0.005, self.noise_bounds))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    candidate = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                                         random_state=0).fit(X_tile, y_tile)
                error = loo_error(candidate)
                if best is None or error < best[0]:
                    best = (error, candidate.kernel_)
            fitted = best[1]

            # Iperparametri riportati in scala assoluta (moltiplicando per la varianza della zona).
            amplitude = fitted.k1.k1.constant_value * variance
            length_scale = float(fitted.k1.k2.length_scale)
            nu = float(fitted.k1.k2.nu)
            noise = fitted.k2.noise_level * variance

            # 3) Fit finale della zona con iperparametri fissati, su tutti i suoi punti
            #    (si sottrae la media della zona invece di normalizzare).
            fixed_kernel = (ConstantKernel(amplitude, "fixed") * Matern(length_scale, "fixed", nu=nu)
                            + WhiteKernel(noise, "fixed"))
            offset = y_tile.mean()
            gp = GaussianProcessRegressor(kernel=fixed_kernel, normalize_y=False,
                                          optimizer=None).fit(X_tile, y_tile - offset)
            self.tiles.append((gp, offset, noise))

        print(f"Fitted {len(self.tiles)} local GPs ({n}x{n} tiles)")

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

        # TODO: Use your GP to estimate the posterior mean and stddev for each city_area here
        gp_mean = np.zeros(test_coordinates.shape[0], dtype=float)
        gp_std = np.zeros(test_coordinates.shape[0], dtype=float)

        X = np.asarray(test_coordinates, dtype=float)[:, :2]
        n = self.tile_grid
        noise_var = np.zeros(X.shape[0], dtype=float)

        # Ogni punto viene previsto dal GP della zona in cui cade.
        column = np.minimum((X[:, 0] * n).astype(int), n - 1)
        row = np.minimum((X[:, 1] * n).astype(int), n - 1)
        tile_index = column + n * row
        for tile, (gp, offset, noise) in enumerate(self.tiles):
            idx = np.where(tile_index == tile)[0]
            if len(idx) == 0:
                continue
            with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                for start in range(0, len(idx), 5000):
                    batch = idx[start:start + 5000]
                    mean, std = gp.predict(X[batch], return_std=True)
                    gp_mean[batch] = mean + offset
                    gp_std[batch] = std
            noise_var[idx] = noise

        # TODO: Use the GP posterior to form your predictions here
        predictions = gp_mean.copy()

        residential = np.asarray(test_residential_flags) == 1
        noise = noise_var[residential]
        # gp_std include il rumore di misura (WhiteKernel): la varianza del campo latente
        # e' la varianza totale meno il rumore. Nel margine entra solo una frazione del rumore.
        latent_var = np.maximum(gp_std[residential] ** 2 - noise, 0.0)
        variance = latent_var + self.test_noise_fraction * noise

        # Margine di base per i punti residenziali: sottostimare costa 50 volte di piu'.
        shift = self.k * np.sqrt(variance)

        # Profondita' del punto dentro la sua zona residenziale (raggio - distanza dal centro).
        dist = np.linalg.norm(X[residential][:, None, :] - self.circles[None, :, :2], axis=2)
        depth = self.circles[None, :, 2] - dist
        depth = np.max(np.where(depth > 0, depth, -np.inf), axis=1)

        # Vicino al confine della zona residenziale si aggiunge un margine extra
        # proporzionale all'incertezza del punto.
        shift = shift + np.where(depth < self.border_width, self.border_k * np.sqrt(variance), 0.0)

        predictions[residential] += shift

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
    # Le prime due colonne sono le coordinate, la terza il flag residenziale.
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