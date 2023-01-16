import scipy
import numpy as np
from itertools import groupby
from operator import itemgetter
from scipy.stats import mannwhitneyu


def almost_equal(value1, value2, rtol=1e-2):
    rerr = np.abs(value1 - value2)
    if isinstance(value1, np.ndarray):
        return (rerr <= rtol).all()
    else:
        return rerr <= rtol


def are_significantly_different(sample_1, sample_2, alpha=0.05):
    stat, p = mannwhitneyu(sample_1, sample_2)
    return p <= alpha


def empirical_f_test(data, ref_std, F_critical=1.5):
    var_1 = np.std(data) ** 2
    var_2 = ref_std ** 2
    F = var_1 / var_2 if var_1 > var_2 else var_2 / var_1
    return F, F <= F_critical


def pure_f_test(data, ref_std, alpha=0.1):
    def _F_critical(alpha):
        # http://socr.ucla.edu/Applets.dir/F_Table.html
        if alpha == 0.1:
            return 2.70554
        elif alpha == 0.05:
            return 3.8415
        elif alpha == 0.025:
            return 5.0239
        elif alpha == 0.01:
            return 6.635

    var_1 = np.std(data) ** 2
    var_2 = ref_std ** 2
    F = var_1 / var_2 if var_1 > var_2 else var_2 / var_1
    return F, F <= _F_critical(alpha)


def smoothness(data):
    data_size = len(data)
    if data_size < 1:
        return 1.0
    ratios = (data[1:] / data[:-1])
    rate_changes = np.abs(np.diff(ratios > 1.))
    rate_changes_count = np.count_nonzero(rate_changes)
    smoothness = (data_size - rate_changes_count) / data_size
    return smoothness


def compute_ro_B(activations, min_out, max_out, bins_count):
    bin_size = (max_out - min_out) / bins_count
    bins = np.arange(min_out, max_out, bin_size).tolist()
    divided_values = np.digitize(activations, bins)
    data = [(neu_act, bin_v) for neu_act, bin_v in zip(divided_values, activations)]
    data = list(zip(divided_values, activations))
    grouped_data = [list(map(lambda x: x[1], group)) for _, group in groupby(sorted(data), key=itemgetter(0))]
    f_g = [(len(values), np.mean(values)) for values in grouped_data]
    f_g_prime = np.array([(f_b, np.abs(2 * (g_b - min_out) / (max_out - min_out) - 1) * f_b) for f_b, g_b in f_g])
    return f_g_prime[:, 1].sum() / f_g_prime[:, 0].sum()


# Copyright 2018 Google Inc.
# https://github.com/google-research/google-research/blob/master/representation_similarity/Demo.ipynb
def gram_linear(x):
    """Compute Gram (kernel) matrix for a linear kernel.

  Args:
    x: A num_examples x num_features matrix of features.

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
    return x.dot(x.T)


def gram_rbf(x, threshold=1.0):
    """Compute Gram (kernel) matrix for an RBF kernel.

  Args:
    x: A num_examples x num_features matrix of features.
    threshold: Fraction of median Euclidean distance to use as RBF kernel
      bandwidth. (This is the heuristic we use in the paper. There are other
      possible ways to set the bandwidth; we didn't try them.)

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
    dot_products = x.dot(x.T)
    sq_norms = np.diag(dot_products)
    sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_median_distance = np.median(sq_distances)
    return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


def center_gram(gram, unbiased=False):
    """Center a symmetric Gram matrix.

  This is equvialent to centering the (possibly infinite-dimensional) features
  induced by the kernel before computing the Gram matrix.

  Args:
    gram: A num_examples x num_examples symmetric matrix.
    unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
      estimate of HSIC. Note that this estimator may be negative.

  Returns:
    A symmetric matrix with centered columns and rows.
  """
    if not np.allclose(gram, gram.T):
        raise ValueError('Input must be a symmetric matrix.')
    gram = gram.copy()

    if unbiased:
        # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
        # L. (2014). Partial distance correlation with methods for dissimilarities.
        # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
        # stable than the alternative from Song et al. (2007).
        n = gram.shape[0]
        np.fill_diagonal(gram, 0)
        means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
        means -= np.sum(means) / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        np.fill_diagonal(gram, 0)
    else:
        means = np.mean(gram, 0, dtype=np.float64)
        means -= np.mean(means) / 2
        gram -= means[:, None]
        gram -= means[None, :]
    return gram


def cka(gram_x, gram_y, debiased=False):
    """Compute CKA.

  Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.

  Returns:
    The value of CKA between X and Y.
  """
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)

    # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
    # n*(n-3) (unbiased variant), but this cancels for CKA.
    scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

    normalization_x = np.linalg.norm(gram_x)
    normalization_y = np.linalg.norm(gram_y)
    return scaled_hsic / (normalization_x * normalization_y)


def _debiased_dot_product_similarity_helper(
        xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y,
        n):
    """Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
    # This formula can be derived by manipulating the unbiased estimator from
    # Song et al. (2007).
    return (
            xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y)
            + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))


def feature_space_linear_cka(features_x, features_y, debiased=False):
    """Compute CKA with a linear kernel, in feature space.

  This is typically faster than computing the Gram matrix when there are fewer
  features than examples.

  Args:
    features_x: A num_examples x num_features matrix of features.
    features_y: A num_examples x num_features matrix of features.
    debiased: Use unbiased estimator of dot product similarity. CKA may still be
      biased. Note that this estimator may be negative.

  Returns:
    The value of CKA between X and Y.
  """
    features_x = features_x - np.mean(features_x, 0, keepdims=True)
    features_y = features_y - np.mean(features_y, 0, keepdims=True)

    dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
    normalization_x = np.linalg.norm(features_x.T.dot(features_x))
    normalization_y = np.linalg.norm(features_y.T.dot(features_y))

    if debiased:
        n = features_x.shape[0]
        # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
        sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
        sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
        squared_norm_x = np.sum(sum_squared_rows_x)
        squared_norm_y = np.sum(sum_squared_rows_y)

        dot_product_similarity = _debiased_dot_product_similarity_helper(
            dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
            squared_norm_x, squared_norm_y, n)
        normalization_x = np.sqrt(_debiased_dot_product_similarity_helper(
            normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
            squared_norm_x, squared_norm_x, n))
        normalization_y = np.sqrt(_debiased_dot_product_similarity_helper(
            normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
            squared_norm_y, squared_norm_y, n))
    if (normalization_x * normalization_y) == 0.:
        return 0.
    return dot_product_similarity / (normalization_x * normalization_y)
