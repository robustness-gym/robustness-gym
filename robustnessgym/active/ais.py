import math

import numpy as np
from sklearn.isotonic import IsotonicRegression


# Alternative to tree: isotonic regression
def fit_isotonic(y_test, prob_pos):
    """Given 0/1 labels 'y_test' and predicted model scores 'prob_pos', Train
    an isotonic regression model to calibrate model scores."""
    try:
        iso_reg = IsotonicRegression(out_of_bounds="clip").fit(prob_pos, y_test)
    except RuntimeError:
        iso_reg = None
    return iso_reg


def sample_iso(
    y_pred,
    prob_pos,
    iso_reg,
    samples,
    g,
    alpha,
    d_set,
):
    """Weighted sampling of rows to label.

    Args:
        y_pred: predicted labels (k,)
        prob_pos: model scores (k,)
        iso_reg: isotonic regression model (or None)
        samples: number of datapoints to sample (int)
        g: current best-prediction of the F-Score (float in [0,1])
        alpha: defines the F_{\alpha} score estimated (float in [0,1])
            alpha = 0   -> recall
            alpha = 1   -> precision
            alpha = 0.5 -> F1
        d_set: indicators of whether each datapoint has already been labeled (k,)
            set this to np.zeros((k)) if no datapoints have been labeled

    Returns:
        sampled rows (samples,)
        weights of sample d rows (samples,)
    """
    num_rows = len(y_pred)

    # Deterministically sample the rows that have already been labeled
    # (since they require no additional budget)
    d_rows = np.where(d_set)[0]
    d_weights = np.ones(len(d_rows))
    if len(d_rows) > samples:
        return d_rows, d_weights
    samples -= len(d_rows)

    # Sample randomly if there's no isotonic regression model available for calibration
    # Otherwise, use the calibration model to compute calibrated probabilities 'p_1'
    if iso_reg is None:
        rand_rows = np.random.choice(num_rows, size=samples, replace=True)
        return np.concatenate([rand_rows, d_rows]), np.concatenate(
            [np.full((samples), 1.0 / num_rows), d_weights]
        )
    else:
        p_1 = iso_reg.predict(prob_pos) * 0.98 + 0.01  # Just to smooth 0 and 1

    # Compute Sawade et al.'s sampling weights from the calibrated probabilities 'p_1'
    weight_1 = np.sqrt(p_1 * ((1.0 - g) ** 2) + ((alpha * g) ** 2) * (1.0 - p_1))
    weight_0 = (1 - alpha) * np.sqrt(p_1 * (g ** 2))
    weights = weight_1 * y_pred + weight_0 * (1 - y_pred)

    # Sample according to 'weights' and return the rows sampled along with their
    # associated weights
    weight_sum = np.sum(weights)
    if weight_sum > 0:
        weights = weights[d_set is not True].squeeze()
        weights /= np.sum(weights)
        options = np.arange(num_rows)
        options = options[d_set is not True].squeeze()

        rand_row_indices = np.random.choice(
            len(options), size=samples, p=weights, replace=True
        )
        rand_rows = options[rand_row_indices]
        return np.concatenate([rand_rows, d_rows]), np.concatenate(
            [1.0 / (weights[rand_row_indices] * samples), d_weights]
        )
    else:
        # If all weights are zero, sample randomly
        rand_rows = np.random.choice(num_rows, size=samples, replace=True)
        return np.concatenate([rand_rows, d_rows]), np.concatenate(
            [np.full(samples, 1.0 / num_rows), d_weights]
        )


def get_fscore(y_pred, y_test, rows, weights):
    """
    Compute F-Score from sampled rows and weights
    Args:
        y_pred: model predictions (k,)
        y_test: ground-truth labels (k,)
        rows: sampled rows (samples,)
        weights: sampled weights (samples,)
    Returns:
        score: estimated F-Score
        trialstd: estimated standard deviation of F-Score estimate
        den:
    """
    unique_rows = np.unique(rows)
    unique_weights = np.array([weights[rows == elem].sum() for elem in unique_rows])
    rows = unique_rows
    weights = unique_weights

    # Compute F-Score
    num = ((y_test[rows] == y_pred[rows]) * weights).sum()
    den = weights.sum()
    score = num / den

    # Compute standard deviation of estimate
    var = ((((y_test[rows] == y_pred[rows]) - score) ** 2) * (weights ** 2)).mean() / (
        (weights.mean() ** 2) * (1.0 - ((weights ** 2).sum() / (weights.sum() ** 2)))
    )
    n = len(rows)
    trialstd = np.sqrt(var) / np.sqrt(n)

    return score, trialstd, den


def ais_singleiter(
    y_pred, y_test, prob_pos, sample_budget, g, alpha, known_rows, filter_rows
):
    """Perform a single AIS iteration of calibration + sampling.

    Args:
        y_pred: model predictions (k,)
        y_test: ground-truth labels for sampled rows (samples,)
        prob_pos: model scores (k,)
        sample_budget: labeling budget for the iteration (int)
        g: current best-prediction of the F-Score (float in [0,1])
        alpha: defines the F_{\alpha} score estimated (float in [0,1])
            alpha = 0   -> recall
            alpha = 1   -> precision
            alpha = 0.5 -> F1
        known_rows: sampled rows (samples,)
        filter_rows: indicator array allowing us to restrict sampling to a
            subset of rows (k,)

    Returns:
        score: estimated F-Score
        trialstd: estimated standard deviation of F-Score estimate
    """
    if np.sum(known_rows) > 0:
        iso_reg = fit_isotonic(y_test, prob_pos[known_rows])
    else:
        iso_reg = None
    rand_rows, weights = sample_iso(
        y_pred[filter_rows],
        prob_pos[filter_rows],
        iso_reg,
        sample_budget,
        g,
        alpha,
        known_rows[filter_rows],
    )
    return rand_rows, weights


def ais_fullalgorithm(y_pred, y_test, prob_pos, sample_budget):
    """
    Combine iterative AIS sampling with F-Score computation to compute best
    estimate of F-Score
    Args:
        y_pred: model predictions (k,)
        y_test: ground-truth labels (used as oracle) (k,)
        prob_pos: model scores (k,)
        sample_budget: total labeling budget (int)
    Returns:
        prf1: estimates of (precision, recall, F1)
        stds: estimates of standard deviation of estimated (precision, recall, F1)
        budget: actual labeling budget used (int)
    """

    # Initialize relevant variables
    k_keep = 4
    measurements = []
    stds = []
    measurement_weights = []
    all_rand_rows = []
    avg_budget = 0
    g = 0.5
    alpha = 0.5
    starting_budget = 10
    iteration_count = np.floor(np.log2(sample_budget / starting_budget)).astype(int) + 1

    # Sampling loop to iteratively sample batches of points to label
    for i in range(iteration_count):
        # Restrict sampling domain in early iterations when there aren't many
        # labeled positives
        # This significantly improves performance on rare categories
        poses = y_pred.sum()
        if (3 * (i + 1)) * poses < len(y_pred):
            filter_rows = np.argpartition(prob_pos, -((3 * (i + 1)) * poses))[
                -((3 * (i + 1)) * poses) :
            ]
        else:
            filter_rows = np.arange(len(y_pred))

        # Enumerate the already-sampled rows
        if len(all_rand_rows) > 0:
            unique_rows = np.unique(np.concatenate(all_rand_rows))
            known_rows = np.zeros(len(y_pred), dtype=bool)
            known_rows[unique_rows] = True
        else:
            known_rows = np.zeros(len(y_pred), dtype=bool)
        # Double sampling budget every iteration
        if i == (iteration_count - 1):
            iter_budget = sample_budget
        else:
            iter_budget = starting_budget * (2 ** i)

        # Use AIS algorithm to sample rows to label
        rand_rows, weights = ais_singleiter(
            y_pred=y_pred,
            y_test=y_test[known_rows],
            prob_pos=prob_pos,
            sample_budget=iter_budget,
            g=g,
            alpha=alpha,
            known_rows=known_rows,
            filter_rows=filter_rows,
        )
        all_rand_rows.append(filter_rows[rand_rows])
        rand_rows = filter_rows[rand_rows]
        weights *= len(rand_rows) / len(y_pred)

        # Compute precision, recall, and F1 using sampled rows and weights
        # Also computes the standard deviation of the estimates
        prec, prec_trialstd, prec_den = get_fscore(
            y_pred,
            y_test,
            rand_rows,
            weights * y_pred[rand_rows],
        )
        rec, rec_trialstd, rec_den = get_fscore(
            y_pred,
            y_test,
            rand_rows,
            weights * y_test[rand_rows],
        )
        f1, f1_trialstd, f1_den = get_fscore(
            y_pred,
            y_test,
            rand_rows,
            weights * (0.5 * y_pred[rand_rows] + 0.5 * y_test[rand_rows]),
        )
        measurements.append([prec, rec, f1])
        stds.append([prec_trialstd, rec_trialstd, f1_trialstd])

        # Update current best estimate of F1
        if not math.isnan(f1):
            g = 0.5 * g + 0.5 * f1
        measurement_weights.append([prec_den, rec_den, f1_den])

    all_rand_rows = np.unique(np.concatenate(all_rand_rows))
    measurements = np.asarray(measurements)
    stds = np.asarray(stds)
    measurement_weights = np.array(measurement_weights) + 0.0001

    # Keep only the results from the last 'k_keep' iterations of the algorithm
    if k_keep > 0:  # Set to -1 to deactivate
        measurements = measurements[-k_keep:]
        stds = stds[-k_keep:]
        measurement_weights = measurement_weights[-k_keep:]

    # Compute a weighted average of the estimates of F-Score computed across the
    # last 'k_keep' iterations
    avg_measurements = np.zeros(3)
    avg_stds = np.zeros(3)
    for k in range(3):
        indices = ~np.isnan(measurements[:, k])
        if indices.sum() > 0:
            normalized_weights = measurement_weights[:, k][indices]
            normalized_weights /= np.sum(normalized_weights)
            avg_measurements[k] = np.average(
                measurements[:, k][indices], weights=measurement_weights[:, k][indices]
            )
            avg_stds[k] = np.sqrt(
                np.sum((stds[:, k][indices] * normalized_weights) ** 2)
            )
        else:
            avg_measurements[k] = math.nan
            avg_stds[k] = math.nan
    avg_budget = len(all_rand_rows)

    return np.array(avg_measurements), np.array(avg_stds), np.array(avg_budget)
