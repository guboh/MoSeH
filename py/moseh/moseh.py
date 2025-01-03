# Copyright (c) 2024 Gustav Bohlin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import chi2
from scipy.linalg import sqrtm

def solve(
        input_data, target_data,
        model_function, jacobian_function,
        initial_model_order_specifier,
        expansion_operator,
        update_operator,
        group_lengths=None,
        beta=None, alpha=0.05,
        max_iterations=100
    ):
    """
    Model-Order Selection via Sequential Lagrange Multiplier Hypothesis Testing (MoSeH).

    Parameters:
    -----------
    input_data : Union[np.ndarrqay, List[np.ndarray]]
        An input data matrix, or a list of input data matrices. Each matrix corresponds to a group of data points.
        The number of rows in each matrix must match the number of input variables in the model.
        The number of columns in each matrix must match the number of data points in the corresponding group.
    target_data : Union[np.ndarrqay, List[np.ndarray]]
        A 1D array of targets, or a list of 1D arrays of target values. Each array corresponds to a group of data points.
        The number of elements in the list must match the number of input data matrices,
        and the number of elements in each sub-array must match the number of data points in the corresponding group.
    model_function : Callable[model_order_specifier, input_data, beta]
        Returns the specified model function evaluated at the given input data and coefficients beta.
    jacobian_function : Callable[model_order_specifier, input_data, beta]
        Returns the specified model's Jacobian matrix evaluated at the given input data and coefficients beta.
    initial_model_order_specifier : ModelOrderSpecifier
        The initial model order specifier. It can be, e.g., an integer, a list of integers, or some other more complex object.
        It must implement __len__(self) -> int to return the number of parameters in the model.
    expansion_operator : Callable[model_order_specifier, beta, covariance]
        Returns the full model's residual and Jacobian matrix evaluated at beta.
    update_operator : Callable[model_order_specifier, decision_index, beta]
        Returns updated model order specifier and beta.
    beta : Union[np.ndarray, List[np.ndarray]] (optional)
        The initial coefficients, or a list of initial coefficients, one for each group of data points.
    alpha : float (optional)
        The significance level for the Lagrange multiplier (score) test.
    max_iterations : int (optional)
        The maximum number of iterations.

    Returns:
    --------
    model_order_specifier : ModelOrderSpecifier
        The selected model order specifier.
    beta : np.ndarray
        The estimated coefficients.
    converged : bool
        True if the algorithm converged, False otherwise.
    """

    # If the initial model order specifier is an integer, convert it so that len() can be used on it
    if isinstance(initial_model_order_specifier, int):
        model_order_specifier = _IntAsLen(initial_model_order_specifier)
    else:
        model_order_specifier = initial_model_order_specifier

    # Check validity of the initial model order specifier
    if not hasattr(model_order_specifier, "__len__") and not callable(model_order_specifier.__len__):
        raise TypeError("Initial model order specifier must be an int or have a __len__ method.")

    # Repackage the input and target data if they are not lists
    if not isinstance(input_data, list):
        input_data = [input_data]
    if not isinstance(target_data, list):
        target_data = [target_data]

    # Check if the number of input data matrices and target data arrays match
    if len(input_data) != len(target_data):
        raise ValueError("The number of input data matrices and target data arrays must match.")

    # Check that the sizes the input and target data matrices make sense
    for i in range(len(input_data)):
        # Check that the target data is a 1D array
        if target_data.ndim != 1:
            raise ValueError("The target data must be a 1D array.")
        if input_data[i].shape[0] != len(target_data[i]):
            raise ValueError("The number of columns in the input data matrix must match the number of target values.")

    if beta is None:
        beta = [np.zeros(len(model_order_specifier)) for _ in range(len(input_data))]
    elif not isinstance(beta, list):
        beta = [beta for _ in range(len(input_data))]
    else:
        if len(beta) != len(input_data):
            raise ValueError("The number of initial coefficient vectors must match the number of input data matrices.")
        for i in range(len(beta)):
            if len(beta[i]) != len(model_order_specifier):
                raise ValueError(
                    "The total number of parameters in the model order specifier and the "
                    "number of coefficients in the initial coefficient vector must match."
                )

    # Main loop
    converged = False
    for _ in range(max_iterations):
        # Set up the residual and Jacobian matrix functions for the currently selected model
        residual_s = lambda x: model_function(model_order_specifier, x)
        jacobian_s = lambda x: jacobian_function(model_order_specifier, x)

        beta, covariance = _fisher_scoring(residual_s, jacobian_s, beta, group_lengths)

        # Expand the model
        residual_full, jacobian_full, nr_of_constraints = expansion_operator(model_order_specifier, beta, covariance)

        # Calculate the full model's score and Fisher information matrix (fim)
        score_full = jacobian_full.T @ np.linalg.solve(covariance, residual_full)
        fim_full = jacobian_full.T @ np.linalg.solve(covariance, jacobian_full)

        # Calculate the Lagrange multiplier (score) test statistic
        lm_test_statistic = score_full.T @ np.linalg.solve(fim_full, score_full) # @TODO: Can we reuse previous calculations to save time?

        # Perform the hypothesis test
        if lm_test_statistic <= chi2.ppf(1-alpha, nr_of_constraints):
            converged = True
            break

        # Calculate the Schur complement of the full model's Fisher information matrix.
        # This corresponds to the covariance of the extra full model parameters, given the current model parameters.
        nr_of_parameters = len(model_order_specifier)
        fim = fim_full[:nr_of_parameters, :nr_of_parameters]
        fim_extra_diagonal_block = fim_full[nr_of_parameters:, nr_of_parameters:]
        fim_extra_cross_term_block = fim_full[nr_of_parameters:, :nr_of_parameters]
        covariance_extra = fim_extra_diagonal_block - fim_extra_cross_term_block @ np.linalg.solve(fim, fim_extra_cross_term_block.T)

        # Transform (whiten) the Lagrange multipliers (corresponding to the negative of the score for the extra parameters)
        lagrange_multipliers = -score_full[nr_of_parameters:]
        transformed_lagrange_multipliers = np.linalg.solve(sqrtm(covariance_extra), lagrange_multipliers)

        # Calculate the decision index
        nr_of_groups = len(group_lengths) if group_lengths is not None else 1
        if (nr_of_constraints % nr_of_groups) != 0:
            raise ValueError("Someting is wrong. The number of parameters must be divisible by the number of groups.")
        nr_of_constraints_per_group = nr_of_constraints // nr_of_groups
        decision_index = np.argmax(
            np.median(
                np.abs(transformed_lagrange_multipliers).reshape(nr_of_groups, nr_of_constraints_per_group),
                axis=1
            )
        )

        # Update the model order and the coefficient vector size based on the decision index
        model_order_specifier, beta = update_operator(model_order_specifier, decision_index, beta)

    # If the initial model order specifier was an integer, convert it back to an integer
    if isinstance(initial_model_order_specifier, int):
        model_order_specifier = len(model_order_specifier)

    return model_order_specifier, beta, converged

def _fisher_scoring(model_function, jacobian_function, beta, group_lengths=None):
    """
    Estimate the parameters beta using Fisher scoring.

    Parameters:
    -----------
    model_function : Callable[beta]
        Returns the model's residual evaluated at beta.
    jacobian_function : Callable[beta]
        Returns the model's Jacobian matrix evaluated at beta.
    beta : np.ndarray
        The initial coefficients.
    group_lengths : list of int (optional)
        The lengths of subgroups of data points, i.e., the lengths of subgroups of the residuals and Jacobian matrices.
        If None, the data points are not grouped.

    Returns:
    --------
    beta : np.ndarray
        The estimated coefficients.
    covariance : np.ndarray
        The estimated covariance matrix.
    """

    # Create initual variance estimates
    residual = model_function(beta)

    if group_lengths is None:
        group_lengths = [len(residual)]

    covariance = _estimate_variance(beta, residual, group_lengths)

    # Fisher scoring
    for _ in range(100):
        # Calculate the Jacobian matrix
        jacobian = jacobian_function(beta)

        # Calculate the score
        score = jacobian.T @ np.linalg.solve(covariance, residual)

        # Calculate the Fisher information matrix (FIM)
        fim = jacobian.T @ np.linalg.solve(covariance, jacobian)

        # Update the coefficients
        beta_old = beta
        beta += np.linalg.solve(fim, score)

        # Calculate the new residual
        residual = model_function(beta)

        # Calculate the new covariance matrix
        covariance = _estimate_variance(beta, residual, group_lengths)

        # Check for convergence
        if np.linalg.norm(beta - beta_old) < 1e-6: # @TODO: Work out a better convergence criterion. Can we use the LM test statistic?
            break

    return beta, covariance


def _estimate_variance(beta, residual, group_lengths, unbiased=True):
    """
    Estimate the variance of the residuals.

    Parameters:
    -----------
    beta : np.ndarray
        The estimated coefficients.
    residual : np.ndarray
        The residuals.
    group_lengths : list of int
        The lengths of subgroups of data points, i.e., the lengths of subgroups of the residuals and Jacobian matrices.
    unbiased : bool (optional)
        If True, the unbiased estimator of the variance is used.

    Returns:
    --------
    covariance : np.ndarray
        The estimated covariance matrix.
    """
    # Create initual variance estimates
    sigma_squared = np.zeros(len(group_lengths))
    start = 0

    # Calculate the variance estimates for the data groups
    for idx, length in enumerate(group_lengths):
        end = start + length
        if unbiased:
            sigma_squared[idx] = np.sum(residual[start:end]**2) / (length - len(beta))
        else:
            sigma_squared[idx] = np.sum(residual[start:end]**2) / length

    covariance = np.diag(np.repeat(sigma_squared, group_lengths))

    return covariance

class _IntAsLen:
    def __init__(self, value):
        if not isinstance(value, int):
            raise TypeError("Value must be an int.")
        self.value = value

    def __len__(self):
        return self.value
