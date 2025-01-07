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
from scipy.stats import chi2, mode
from scipy.linalg import sqrtm

def solve(
        input_data_list, target_data_list,
        model_function, jacobian_function,
        initial_model_order_specifier,
        expansion_operator,
        update_operator,
        theta_list=None, alpha=0.05,
        max_iterations=100
    ):
    """
    Model-Order Selection via Sequential Lagrange Multiplier Hypothesis Testing (MoSeH).

    Parameters:
    -----------
    input_data_list : Union[np.ndarrqay, List[np.ndarray]]
        An input data matrix, or a list of input data matrices. Each matrix corresponds to a sub-problem (group of data points).
        The number of rows in each matrix must match the number of input variables in the model.
        The number of columns in each matrix must match the number of data points in the corresponding sub-problem.
    target_data_list : Union[np.ndarrqay, List[np.ndarray]]
        A 1D array of targets, or a list of 1D arrays of target values. Each array corresponds to a sub-problem (group of data points).
        The number of elements in the list must match the number of input data matrices,
        and the number of elements in each sub-array must match the number of data points in the corresponding sub-problem.
    model_function : Callable[model_order_specifier : ModelOrderSpecifier, theta : np.ndarray, input_data : np.ndarray] -> np.ndarray
        Returns the specified model function evaluated with parameters theta at the given input data.
    jacobian_function : Callable[model_order_specifier : ModelOrderSpecifier, theta : np.ndarray, input_data : np.ndarray] -> np.ndarray
        Returns the specified model's Jacobian matrix evaluated with parameters theta at the given input data.
    initial_model_order_specifier : ModelOrderSpecifier
        The initial model order specifier. It can be, e.g., an integer, a list of integers, or some other more complex object.
        It must implement __len__(self) -> int to return the number of parameters in the model.
    expansion_operator : Callable[model_order_specifier : ModelOrderSpecifier] -> tuple[ModelOrderSpecifier, np.ndarray]
        Returns the expaned model order specifier together with an expansion matrix.
        The expansion matrix can be used to pad theta with zeros to match the size of the expanded model.
    update_operator : Callable[model_order_specifier : ModelOrderSpecifier, decision_index : int] -> tuple[ModelOrderSpecifier, np.ndarray]
        Returns updated model order specifier together with a selection matrix.
        The selection matrix can be used to select the relevant parts of theta, the score and the Fisher information matrix.
    theta_list : Union[np.ndarray, List[np.ndarray]] (optional)
        The initial parameters, or a list of initial parameters, one for each sub-problem (group of data points).
    alpha : float (optional)
        The significance level for the Lagrange multiplier (score) test.
    max_iterations : int (optional)
        The maximum number of iterations.

    Returns:
    --------
    model_order_specifier : ModelOrderSpecifier
        The selected model order specifier.
    theta : np.ndarray
        The estimated parameters.
    converged : bool
        True if the algorithm converged, False otherwise.
    """

    # If the initial model order specifier is an integer, convert it so that len() can be used on it
    if isinstance(initial_model_order_specifier, int):
        model_order_specifier = _IntAsLen(initial_model_order_specifier)
    else:
        model_order_specifier = initial_model_order_specifier

    #########################
    # Input data validation #
    #########################

    # Check validity of the initial model order specifier
    if not hasattr(model_order_specifier, "__len__") and not callable(model_order_specifier.__len__):
        raise TypeError("Initial model order specifier must be an int or have a __len__ method.")

    # Repackage the input and target data if they are not lists
    if not isinstance(input_data_list, list):
        input_data_list = [input_data_list]
    if not isinstance(target_data_list, list):
        target_data_list = [target_data_list]

    # Check if the number of input data matrices and target data arrays match
    if len(input_data_list) != len(target_data_list):
        raise ValueError("The number of input data matrices and target data arrays must match.")

    # Check that the sizes the input and target data matrices make sense
    for input_data, target_data in zip(input_data_list, target_data_list):
        # Check that the target data is a 1D array
        if target_data.ndim != 1:
            raise ValueError("The target data must be a 1D array.")
        if input_data.shape[0] != len(target_data):
            raise ValueError("The number of rows in the input data matrix must match the number of target values.")

    if theta_list is None:
        theta_list = [np.zeros(len(model_order_specifier)) for _ in range(len(input_data_list))]
    elif not isinstance(theta_list, list):
        theta_list = [theta_list for _ in range(len(input_data_list))]
    else:
        if len(theta_list) != len(input_data_list):
            raise ValueError("The number of initial coefficient vectors must match the number of input data matrices.")
        for i in range(len(theta_list)):
            if len(theta_list[i]) != len(model_order_specifier):
                raise ValueError(
                    "The total number of parameters in the model order specifier and the "
                    "number of parameters in the initial coefficient vector must match."
                )

    #############
    # Main loop #
    #############

    converged = False
    for _ in range(max_iterations):
        # Expand the model
        model_order_specifier_full, expansion_matrix = expansion_operator(model_order_specifier)

        # Create appropriate model functions and Jacobian functions for the full model
        model_function_full = lambda theta, input_data: model_function(model_order_specifier_full, theta, input_data)
        jacobian_function_full = lambda theta, input_data: jacobian_function(model_order_specifier_full, theta, input_data)
        calculation_subroutine = lambda theta, input_data, target_data: _calculate_score_fim_and_lm_test_statistic(
            model_function_full, jacobian_function_full, theta, input_data, target_data
        )

        # Calculate the score, the Fisher information matrix, and the Lagrange multiplier (score) test statistic for the full model
        # in parallel for each sub-problem
        result = Parallel(n_jobs=-1)(
            delayed(calculation_subroutine)(expansion_matrix @ theta, input_data, target_data) # Note that theta is expanded to match the full model
            for theta, input_data, target_data in zip(theta_list, input_data_list, target_data_list)
        )
        score_list, fim_list, lm_test_statistic_list = zip(*result) # Unpack the results

        # Calculate the total Lagrange multiplier (score) test statistic
        lm_test_statistic = np.sum(lm_test_statistic_list)

        # Perform the hypothesis test
        nr_of_constraints = len(model_order_specifier_full) - len(model_order_specifier)
        if lm_test_statistic <= chi2.ppf(1-alpha, nr_of_constraints):
            converged = True
            break

        # Calculate the decision indeces in parallel for each sub-problem
        result = Parallel(n_jobs=-1)(
            delayed(_decision_subroutine)(score, fim) for score, fim in zip(score_list, fim_list)
        )
        decision_indeces, _ = zip(*result) # Unpack the results

        # Calculate the decision index as the mode of the decision indeces
        # @TODO: Check other selection rules, e.g., using an average of the transformed scores instead
        decision_index, _ = mode(decision_indeces)

        # Update the model order specifier and the selection matrix, based on the decision index
        model_order_specifier, selection_matrix = update_operator(model_order_specifier, decision_index)

        # Set up helper functions for the Fisher scoring steps
        residual_function = lambda theta, input_data, target_data: model_function(model_order_specifier, theta, input_data) - target_data
        fisher_scoring_step_subroutine = lambda score, fim, theta, input_data, target_data: _fisher_scoring_step(
            score, fim, theta, input_data, target_data, residual_function
        )

        # Perform Fisher scoring steps with backtracking line search in parallel for each sub-problem
        theta_list = Parallel(n_jobs=-1)(
            delayed(fisher_scoring_step_subroutine)(
                selection_matrix @ score,                    # Note that the score is contracted to match the selected model
                selection_matrix @ fim @ selection_matrix.T, # Note that the FIM is contracted to match the selected model
                selection_matrix @ expansion_matrix @ theta, # Note that theta is expanded and contracted to match the selected model
                input_data, target_data)
            for score, fim, theta, input_data, target_data in zip(score_list, fim_list, theta_list, input_data_list, target_data_list)
        )

    # If the initial model order specifier was an integer, convert it back to an integer
    if isinstance(initial_model_order_specifier, int):
        model_order_specifier = len(model_order_specifier)

    return model_order_specifier, theta_list, converged

def _calculate_score_fim_and_lm_test_statistic(
        model_function, jacobian_function, theta, input_data, target_data):
    """
    Calculate the score, Fisher information matrix (FIM), and Lagrange multiplier (score) test statistic for the expanded model.

    Parameters:
    -----------
    model_function : Callable[theta : np.ndarray, input_data : np.ndarray] -> np.ndarray
        Returns the specified model function evaluated with parameters theta at the given input data.
    jacobian_function : Callable[theta : np.ndarray, input_data : np.ndarray] -> np.ndarray
        Returns the specified model's Jacobian matrix evaluated with parameters theta at the given input data.
    theta : np.ndarray
        The estimated parameters.
    input_data : np.ndarray
        The input data matrix.
    target_data : np.ndarray
        The target data array.

    Returns:
    --------
    score : np.ndarray
        The score vector.
    fim : np.ndarray
        The Fisher information matrix.
    lm_test_statistic : float
        The Lagrange multiplier (score) test statistic.
    """
    # Calculate the residual and the Jacobian matrix
    residual = model_function(theta, input_data) - target_data
    jacobian = jacobian_function(theta, input_data)

    # Calculate the covariance matrix
    covariance = _estimate_covariance(theta, residual)

    # Calculate the full model's score and Fisher information matrix (FIM)
    score = -jacobian.T @ np.linalg.solve(covariance, residual) # Note the minus sign, due to the chosen definition of the residual
    fim = jacobian.T @ np.linalg.solve(covariance, jacobian)

    # Calculate the Lagrange multiplier (score) test statistic
    lm_test_statistic = score.T @ np.linalg.solve(fim, score)

    return score, fim, lm_test_statistic

def _estimate_covariance(theta, residual, unbiased=True):
    """
    Estimate the covariance of the residuals.

    Parameters:
    -----------
    theta : np.ndarray
        The estimated parameters.
    residual : np.ndarray
        The residuals.
    unbiased : bool (optional)
        If True, the unbiased estimator of the variance is used.

    Returns:
    --------
    covariance : np.ndarray
        The estimated covariance matrix.
    """

    if unbiased:
        sigma_squared = np.sum(residual**2) / (len(residual) - len(theta))
    else:
        sigma_squared = np.sum(residual**2) / len(residual)

    covariance = np.diag(np.full(len(residual), sigma_squared))

    return covariance

def _decision_subroutine(score, fim):
    """
    Calculate the decision index based on the score and the Fisher information matrix (FIM).

    Parameters:
    -----------
    score : np.ndarray
        The score vector.
    fim : np.ndarray
        The Fisher information matrix.

    Returns:
    --------
    decision_index : int
        The decision index.
    transformed_score : np.ndarray
        The transformed score vector.
    """
    # Transform (whiten) the score vector
    transformed_score = np.linalg.solve(sqrtm(fim), score)
    # Calculate the decision index.
    # The index corresponds to the parameter that has the largest absolute value in the transformed score vector.
    # If the index corresponds to a constrained parameter, the constraint is relaxed.
    decision_index = np.argmax(np.abs(transformed_score))
    return decision_index, transformed_score

def _fisher_scoring_step(
        score, fim, theta,
        input_data=None, target_data=None,
        residual_function=None,
        armijo_constant=1e-4,
        max_iterations=100):
    """
    Update the selected parameters theta using a single step of Fisher's scoring algorithm with a backtracking line search.

    Parameters:
    -----------
    score : np.ndarray
        The score vector.
    fim : np.ndarray
        The Fisher information matrix.
    theta : np.ndarray
        The estimated parameters.
    input_data : np.ndarray (optional)
        The input data matrix. It is needed by the line search.
    target_data : np.ndarray (optional)
        The target data array. It is needed by the line search.
    residual_function : Callable[theta : np.ndarray, input_data : np.ndarray, target_data : np.ndarray] -> np.ndarray (optional)
        Returns the specified model's residual evaluated with parameters theta at the given input and target data.
        It is needed by the line search. If not provided, the line search is skipped.
    armijo_constant : float (optional)
        The Armijo condition parameter.
    max_iterations : int (optional)
        The maximum number of iterations in the line search.

    Returns:
    --------
    theta_selection : np.ndarray
        The estimated parameters for the selected model.
    """
    # Calculate the Fisher scoring step
    fisher_step = np.linalg.solve(fim, score)
    
    # Update the selected parameters immediately, if no line search is to be performed
    if not residual_function:
        return theta + fisher_step
    
    # Check that the input and target data are provided if the residual function is provided
    if residual_function:
        if input_data is None or target_data is None:
            raise ValueError("The input data and target data must be provided if the residual function is provided.")

    # Set up the backtracking line search
    directional_derivative = score @ fisher_step
    if directional_derivative < 0:
        raise ValueError(
            "The directional derivative of the log-likelihood must be positive. "
            "Something is wrong with the input to the Fisher scoring step."
        )
    ssr_old = np.sum(residual_function(theta, input_data, target_data)**2)
    damping_factor = 1.0

    # Perform the backtracking line search
    for _ in range(max_iterations):
        theta_new = theta + damping_factor * fisher_step
        ssr_new = np.sum(residual_function(theta_new, input_data, target_data)**2)

        if ssr_new <= ssr_old - armijo_constant * damping_factor * directional_derivative:
            return theta_new

        damping_factor /= 2

    raise ValueError("The backtracking line search did not converge.")



class _IntAsLen:
    def __init__(self, value):
        if not isinstance(value, int):
            raise TypeError("Value must be an int.")
        self.value = value

    def __len__(self):
        return self.value
