# Copyright 2024 (c) Gustav Bohlin
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
from scipy.stats import chi2
from scipy.linalg import sqrtm

def solve(
        residual_function, jacobian_function,
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
    residual_function : Callable[model_order_specifier, beta]
        Returns the specified model's residual evaluated at beta.
    jacobian_function : Callable[model_order_specifier, beta]
        Returns the specified model's Jacobian matrix evaluated at beta.
    initial_model_order_specifier : ModelOrderSpecifier
        The initial model order specifier. It can be, e.g., an integer, a list of integers, or some other more complex object.
        It must implement __len__(self) -> int to return the number of parameters in the model.
    expansion_operator : Callable[model_order_specifier, beta, covariance]
        Returns the full model's residual and Jacobian matrix evaluated at beta.
    update_operator : Callable[model_order_specifier, decision_index, beta]
        Returns updated model order specifier and beta.
    group_lengths : list of int (optional)
        The lengths of subgroups of data points, i.e., the lengths of subgroups of the residuals and Jacobian matrices.
        If None, the data points are not grouped.
    beta : np.ndarray (optional)
        The initial coefficients.
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

    model_order_specifier = initial_model_order_specifier # Rename for clarity further down

    if beta is None:
        beta = np.zeros(len(initial_model_order_specifier))

    elif len(model_order_specifier) != len(beta):
        raise ValueError(
            "The total number of parameters in the model order specifier and the "
            "number of coefficients in the initial coefficient vector must match."
        )

    # Main loop
    converged = False
    for _ in range(max_iterations):
        # Set up the residual and Jacobian matrix functions for the currently selected model
        residual_s = lambda x: residual_function(model_order_specifier, x)
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

    return model_order_specifier, beta, converged

def _fisher_scoring(residual_function, jacobian_function, beta, group_lengths=None):
    """
    Estimate the parameters beta using Fisher scoring.

    Parameters:
    -----------
    residual_function : Callable[beta]
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

    def _estimate_variance(beta, residual, group_lengths, unbiased=True):
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

    # Create initual variance estimates
    residual = residual_function(beta)

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
        residual = residual_function(beta)

        # Calculate the new covariance matrix
        covariance = _estimate_variance(beta, residual, group_lengths)

        # Check for convergence
        if np.linalg.norm(beta - beta_old) < 1e-6: # @TODO: Work out a better convergence criterion. Can we use the LM test statistic?
            break

    return beta, covariance
