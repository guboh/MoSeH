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
        model_function=None,
        jacobian_function=None,
        initial_model_order_specifier=None,
        expansion_operator=None,
        update_operator=None,
        theta_list=None, alpha=0.05,
        max_iterations=100,
        force_continuation=False,
        verbose=True,
        max_inner_iterations=100,
        line_search=True,
        armijo_constant=1e-4,
        max_line_search_iterations=100
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
    model_function : Callable[model_order_specifier : ModelOrderSpecifier, theta : np.ndarray, input_data : np.ndarray] -> np.ndarray (optional)
        Returns the specified model function evaluated with parameters theta at the given input data.
        If not provided, a Taylor expansion based model is used.
    jacobian_function : Callable[model_order_specifier : ModelOrderSpecifier, theta : np.ndarray, input_data : np.ndarray] -> np.ndarray (optional)
        Returns the specified model's Jacobian matrix evaluated with parameters theta at the given input data.
        If not provided, a Taylor expansion based model is used.
    initial_model_order_specifier : ModelOrderSpecifier (optional)
        The initial model order specifier. It can be, e.g., an integer, a list of integers, or some other more complex object.
        It must implement __len__(self) -> int to return the number of parameters in the model.
        If not provided, a Taylor expansion based model is assumed, and a zeroth order model is used.
    expansion_operator : Callable[model_order_specifier : ModelOrderSpecifier] -> tuple[ModelOrderSpecifier, np.ndarray] (optional)
        Returns the expaned model order specifier together with an expansion matrix.
        The expansion matrix can be used to pad theta with zeros to match the size of the expanded model.
        If not provided, a Taylor expansion based model is used.
    update_operator : Callable[model_order_specifier : ModelOrderSpecifier, decision_index : int] -> tuple[ModelOrderSpecifier, np.ndarray] (optional)
        Returns updated model order specifier together with a selection matrix.
        The selection matrix can be used to select the relevant parts of theta, the score and the Fisher information matrix.
        If not provided, a Taylor expansion based model is used.
    theta_list : Union[np.ndarray, List[np.ndarray]] (optional)
        The initial parameters, or a list of initial parameters, one for each sub-problem (group of data points).
    alpha : float (optional)
        The significance level for the Lagrange multiplier (score) test.
    max_iterations : int (optional)
        The maximum number of iterations.
    force_continuation : bool (optional)
        If True, the algorithm will continue even if the Lagrange multiplier (score) test fails.
    verbose : bool (optional)
        If True, print the number of iterations, etc.
    max_inner_iterations : int (optional)
        The maximum number of iterations in the inner loop of the Fisher scoring algorithm.
    line_search : bool (optional)
        If True, a backtracking line search is performed in each Fisher scoring step.
    armijo_constant : float (optional)
        The Armijo condition parameter.
    max_line_search_iterations : int (optional)
        The maximum number of iterations in the line search.

    Returns:
    --------
    model_order_specifier : ModelOrderSpecifier
        The selected model order specifier.
    theta : np.ndarray
        The estimated parameters.
    converged : bool
        True if the algorithm converged, False otherwise.
    result_dict : dict
        A dictionary containing the Lagrange multiplier (score) test statistic, the critical value, the AIC, the BIC, and the exact BIC.
    """

    # Repackage the input and target data if they are not lists
    if not isinstance(input_data_list, list):
        input_data_list = [input_data_list]
    if not isinstance(target_data_list, list):
        target_data_list = [target_data_list]

    # If the model function is provided, the Jacobian function must also be provided
    if model_function or jacobian_function or expansion_operator or update_operator:
        if not model_function or not jacobian_function or not expansion_operator or not update_operator:
            raise ValueError("Either all or none of `model_function`, `jacobian_function`, `expansion_operator`, and `update_operator` must be provided.")
        if not initial_model_order_specifier:
            raise ValueError("The initial model order specifier must be provided if `model_function`, `jacobian_function`, `expansion_operator`, and `update_operator` are provided.")
    else:
        # If no model functions are provided, assume a Taylor expansion model
        model_function = _taylor_expansion_model_function
        jacobian_function = _taylor_expansion_jacobian_function
        expansion_operator = _taylor_expansion_expansion_operator
        update_operator = _taylor_expansion_update_operator

        if initial_model_order_specifier is None:
            # If no initial model order specifier is provided,
            # assume a zeroth order model (i.e., a constant model) as the initial model
            initial_model_order_specifier = [(0,) * input_data_list[0].shape[1]]

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
            raise ValueError("The number of initial parameter vectors must match the number of input data matrices.")
        for i in range(len(theta_list)):
            if len(theta_list[i]) != len(model_order_specifier):
                raise ValueError(
                    "The total number of parameters in the model order specifier and the "
                    "number of parameters in the initial parameter vector must match."
                )

    #############
    # Main loop #
    #############

    result_dict = {
        'LM test statistic': [],
        'Critical value': [],
        'AIC': [],
        'BIC': [],
        'BIC exact': []
    }

    converged = False

    # The padding matrix is used to pad theta with zeros to account for newly added parameters.
    # It is initialized as the identity matrix, since no parameters have been added yet.
    padding_matrix = np.eye(len(model_order_specifier))
    for iteration in range(max_iterations):
        print("--- Iteration {} ---".format(iteration + 1)) if verbose else None
        print("Model order: {}".format(len(model_order_specifier))) if verbose else None

        # Set up helper functions for the Fisher scoring algorithm
        model_function_selected = lambda theta, input_data: model_function(model_order_specifier, theta, input_data)
        jacobian_function_selected = lambda theta, input_data: jacobian_function(model_order_specifier, theta, input_data)
        fisher_scoring_subroutine = lambda theta, input_data, target_data: _fisher_scoring(
            model_function_selected,
            jacobian_function_selected,
            theta, input_data, target_data,
            max_iterations=max_inner_iterations,
            line_search=line_search,
            armijo_constant=armijo_constant,
            max_line_search_iterations=max_line_search_iterations
        )

        # Perform Fisher scoring with backtracking line search in parallel for each sub-problem
        result = Parallel(n_jobs=-1)(
            delayed(fisher_scoring_subroutine)(padding_matrix @ theta, input_data, target_data) # Note that theta is padded with zeros, to match the selected model
            for theta, input_data, target_data in zip(theta_list, input_data_list, target_data_list)
        )
        theta_list, _ = zip(*result) # Unpack the results

        # Expand the model, preparing for the Lagrange multiplier (score) test
        model_order_specifier_full, expansion_matrix = expansion_operator(model_order_specifier)

        # Create appropriate model functions and Jacobian functions for the full model, necessary for the Lagrange multiplier (score) test
        model_function_full = lambda theta, input_data: model_function(model_order_specifier_full, theta, input_data)
        jacobian_function_full = lambda theta, input_data: jacobian_function(model_order_specifier_full, theta, input_data)
        calculation_subroutine = lambda theta, input_data, target_data: _calculate_score_fim_and_lm_test_statistic(
            model_order_specifier, model_function_full, jacobian_function_full, theta, input_data, target_data
        )

        # Calculate the score, the Fisher information matrix, and the Lagrange multiplier (score) test statistic for the full model
        # in parallel for each sub-problem
        result = Parallel(n_jobs=-1)(
            delayed(calculation_subroutine)(expansion_matrix @ theta, input_data, target_data) # Note that theta is expanded to match the full model
            for theta, input_data, target_data in zip(theta_list, input_data_list, target_data_list)
        )
        score_list, fim_list, lm_test_statistic_list, aic_list, bic_list, bic_exact_list = zip(*result) # Unpack the results

        # Calculate the total Lagrange multiplier (score) test statistic, AIC, BIC, and exact BIC
        lm_test_statistic = np.sum(lm_test_statistic_list)
        aic = np.sum(aic_list)
        bic = np.sum(bic_list)
        bic_exact = np.sum(bic_exact_list)

        # Prepare for the hypothesis test
        nr_of_constraints = len(input_data) * (len(model_order_specifier_full) - len(model_order_specifier))
        critical_value = chi2.ppf(1-alpha, nr_of_constraints)

        # Print the results of the current iteration
        print("LM test statistic: {}".format(lm_test_statistic)) if verbose else None
        print("Critical value: {}".format(critical_value)) if verbose else None
        print("AIC: {}, BIC: {}, BIC_exact: {}".format(aic, bic, bic_exact)) if verbose else None

        # Save results for later analysis
        result_dict['LM test statistic'].append(lm_test_statistic)
        result_dict['Critical value'].append(critical_value)
        result_dict['AIC'].append(aic)
        result_dict['BIC'].append(bic)
        result_dict['BIC exact'].append(bic_exact)

        # Perform the hypothesis test
        if lm_test_statistic <= critical_value:
            converged = True
            print("Converged!") if verbose else None
            if not force_continuation:
                break
            converged = False
            print("Overriding... Forced continue!") if verbose else None

        # Set up a helper function for the constraint relaxation decision
        decision_subroutine = lambda score, fim: _calculate_decision_index_and_transformed_score(model_order_specifier, score, fim)

        # Calculate the decision indeces in parallel for each sub-problem
        result = Parallel(n_jobs=-1)(
            delayed(decision_subroutine)(score, fim) for score, fim in zip(score_list, fim_list)
        )
        decision_indeces, _ = zip(*result) # Unpack the results

        # Calculate the decision index as the mode of the decision indeces
        # @TODO: Check other selection rules, e.g., using an average of the transformed scores instead
        decision_index, _ = mode(decision_indeces)
        print("Decision index: {}\n".format(decision_index)) if verbose else None

        # Update the model order specifier and the selection matrix, based on the decision index
        model_order_specifier, selection_matrix = update_operator(model_order_specifier, decision_index)

        # Update the padding matrix, to account for the newly added parameters in the next iteration
        padding_matrix = selection_matrix @ expansion_matrix

    # Pad the theta vectors with zeros to match the final model order specifier if not converged.
    # Otherwise, the theta vectors are already padded with zeros.
    if not converged:
        theta_list = Parallel(n_jobs=-1)(
            delayed(lambda x: padding_matrix @ x)(theta)
            for theta in theta_list
        )

    # If the initial model order specifier was an integer, convert it back to an integer
    if isinstance(initial_model_order_specifier, int):
        model_order_specifier = len(model_order_specifier)

    return model_order_specifier, theta_list, converged, result_dict

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

def _fisher_scoring(
        model_function, jacobian_function,
        theta, input_data, target_data,
        max_iterations=100,
        line_search=True,
        armijo_constant=1e-4,
        max_line_search_iterations=100,

        ):
    """
    Update the parameters theta using the Fisher scoring algorithm with (optional) backtracking line search.

    Parameters:
    -----------
    model_function : Callable[theta : np.ndarray, input_data : np.ndarray] -> np.ndarray
        Returns the specified model function evaluated with parameters theta at the given input data.
    jacobian_function : Callable[model_order_specifier : ModelOrderSpecifier, theta : np.ndarray, input_data : np.ndarray] -> np.ndarray
        Returns the specified model's Jacobian matrix evaluated with parameters theta at the given input data.
    theta : np.ndarray
        The estimated parameters.
    input_data : np.ndarray (optional)
        The input data matrix. It is needed by the line search.
    target_data : np.ndarray (optional)
        The target data array. It is needed by the line search.
    max_iterations : int (optional)
        The maximum number of iterations in the Fisher scoring algorithm.
    line_search : bool (optional)
        If True, a backtracking line search is performed in each Fisher scoring step.
    armijo_constant : float (optional)
        The Armijo condition parameter.
    max_line_search_iterations : int (optional)
        The maximum number of iterations in the line search.

    Returns:
    --------
    theta_updated : np.ndarray
        The updated parameters.
    converged : bool
        True if the algorithm converged, False otherwise.
    """
    # Fisher scoring
    converged = False
    theta_updated = theta
    for _ in range(max_iterations):
        # Calculate the residual and the Jacobian matrix
        residual = model_function(theta_updated, input_data) - target_data
        jacobian = jacobian_function(theta_updated, input_data)

        # Estimate the covariance matrix
        covariance = _estimate_covariance(theta_updated, residual)

        # Calculate the score and the Fisher information matrix (FIM)
        score = -jacobian.T @ np.linalg.solve(covariance, residual) # Note the minus sign, due to the chosen definition of the residual
        fim = jacobian.T @ np.linalg.solve(covariance, jacobian)

        # Calculate the Fisher scoring step
        fisher_step = np.linalg.solve(fim, score)

        if line_search:
            # Perform the backtracking line search
            directional_derivative = score @ fisher_step
            if directional_derivative < 0.1:
                raise ValueError(
                    "The directional derivative of the log-likelihood must be positive. "
                    "Something is wrong with the input to the Fisher scoring step."
                )
            ssr_old = np.sum((model_function(theta_updated, input_data) - target_data)**2)
            damping_factor = 1.0

            # Perform the backtracking line search
            line_search_converged = False
            for _ in range(max_line_search_iterations):
                theta_test = theta_updated + damping_factor * fisher_step
                ssr_new = np.sum((model_function(theta_test, input_data) - target_data)**2)

                if ssr_new <= ssr_old - armijo_constant * damping_factor * directional_derivative:
                    line_search_converged = True
                    break

                damping_factor /= 2

            if not line_search_converged:
                raise ValueError("The backtracking line search did not converge.")
            theta_old = theta_updated
            theta_updated = theta_test
        else:
            theta_old = theta_updated
            theta_updated = theta_updated + fisher_step # Note that a new np.ndarray is allocated here

        # Check for Fisher scoring algorithm convergence
        if np.linalg.norm(theta_updated - theta_old) < 1e-6:
            converged = True
            break

    return theta_updated, converged

def _calculate_score_fim_and_lm_test_statistic(
        model_order_specifier, model_function_full, jacobian_function_full, theta, input_data, target_data):
    """
    Calculate the score, Fisher information matrix (FIM), and Lagrange multiplier (score) test statistic for the expanded model.

    Parameters:
    -----------
    model_order_specifier : ModelOrderSpecifier
        The model order specifier of the currently selected model. It can be, e.g., an integer, a list of integers, or some other more complex object.
        It must implement __len__(self) -> int to return the number of parameters in the model.
    model_function_full : Callable[theta : np.ndarray, input_data : np.ndarray] -> np.ndarray
        Returns the specified model function evaluated with parameters theta at the given input data.
    jacobian_function_full : Callable[theta : np.ndarray, input_data : np.ndarray] -> np.ndarray
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
    residual = model_function_full(theta, input_data) - target_data
    jacobian = jacobian_function_full(theta, input_data)

    # Calculate the covariance matrix
    covariance = _estimate_covariance(theta, residual)

    # Calculate the full model's score and Fisher information matrix (FIM)
    score = -jacobian.T @ np.linalg.solve(covariance, residual) # Note the minus sign, due to the chosen definition of the residual
    fim = jacobian.T @ np.linalg.solve(covariance, jacobian)

    nr_of_data_points = len(target_data)
    nr_of_selected_parameters = len(model_order_specifier)
    ssr = np.sum(residual**2)
    aic = nr_of_data_points * np.log(ssr / nr_of_data_points) + 2 * nr_of_selected_parameters
    bic = nr_of_data_points * np.log(ssr / nr_of_data_points) + nr_of_selected_parameters * np.log(nr_of_selected_parameters)
    bic_exact = nr_of_data_points * np.log(ssr / nr_of_data_points) + np.log(np.linalg.det(fim[:nr_of_selected_parameters, :nr_of_selected_parameters]))

    # Calculate the Lagrange multiplier (score) test statistic
    lm_test_statistic = score.T @ np.linalg.solve(fim, score)

    return score, fim, lm_test_statistic, aic, bic, bic_exact

def _calculate_decision_index_and_transformed_score(model_order_specifier, score_full, fim_full):
    """
    Calculate the decision index based on the score and the Fisher information matrix (FIM).

    Parameters:
    -----------
    model_order_specifier : ModelOrderSpecifier
        The model order specifier of the currently selected model. It can be, e.g., an integer, a list of integers, or some other more complex object.
        It must implement __len__(self) -> int to return the number of parameters in the model.
    score_full : np.ndarray
        The score vector of the full model.
    fim_full : np.ndarray
        The Fisher information matrix of the full model.

    Returns:
    --------
    decision_index : int
        The decision index.
    transformed_score : np.ndarray
        The transformed score vector.
    """
    # Calculate the Schur complement of the full model's Fisher information matrix.
    # This corresponds to the covariance of the extra full model parameters, given the current model parameters.
    nr_of_selected_parameters = len(model_order_specifier)
    fim = fim_full[:nr_of_selected_parameters, :nr_of_selected_parameters]
    fim_extra_diagonal_block = fim_full[nr_of_selected_parameters:, nr_of_selected_parameters:]
    fim_extra_cross_term_block = fim_full[nr_of_selected_parameters:, :nr_of_selected_parameters]
    covariance_extra = fim_extra_diagonal_block - fim_extra_cross_term_block @ np.linalg.solve(fim, fim_extra_cross_term_block.T)

    # Transform (whiten) the score vector
    transformed_score = np.linalg.solve(sqrtm(covariance_extra), score_full[nr_of_selected_parameters:])

    # Calculate the decision index.
    # The index corresponds to the parameter that has the largest absolute value in the transformed score vector.
    # If the index corresponds to a constrained parameter, the constraint is relaxed.
    decision_index = nr_of_selected_parameters + np.argmax(np.abs(transformed_score))
    return decision_index, transformed_score

def _assemble_taylor_expansion_design_matrix(model_order_specifier, input_data):
    """
    Helper function to assemble the design matrix for a given model order specifier and input data for the Taylor expansion model.

    The model order specifier is a list of tuples, where each tuple contains the powers of the two input variables to be included in the model.

    Parameters:
    -----------
    model_order_specifier : List[tuple]
        The model order specifier.
    input_data : np.ndarray
        The input data matrix.

    Returns:
    --------
    design_matrix : np.ndarray
        The design matrix.
    """
    design_matrix = np.ones((len(input_data), len(model_order_specifier)))
    for idx, specifier in enumerate(model_order_specifier):
        for input_variable_idx, power in enumerate(specifier):
            design_matrix[:, idx] *= input_data[:, input_variable_idx]**power

    return design_matrix

def _taylor_expansion_model_function(model_order_specifier, theta, input_data):
    """
    The model function implementation.

    The predicted value equals the design_matrix times the theta vector for this model.
    """
    design_matrix = _assemble_taylor_expansion_design_matrix(model_order_specifier, input_data)

    return design_matrix @ theta

def _taylor_expansion_jacobian_function(model_order_specifier, theta, input_data):
    """
    The Jacobian function implementation.

    The Jacobian matrix equals the design matrix for this model.
    """
    design_matrix = _assemble_taylor_expansion_design_matrix(model_order_specifier, input_data)

    return design_matrix

def _taylor_expansion_expansion_operator(model_order_specifier):
    """
    The expansion operator implementation for the Taylor expansion model.
    
    This function adds new parameters to the model order specifier.
    Each new parameter added corresponds to next order derivative terms
    with respect to each of the input variables.

    Parameters:
    -----------
    model_order_specifier : List[tuple]
        The model order specifier to expand.

    Returns:
    --------
    new_model_order_specifier : List[tuple]
        The expanded model order specifier.
    """
    def _expand_single_parameter_specifier(specifier):
        """
        Helper function to expand a single parameter order specifier,
        adding one higher order derivative term for each input variable.

        Parameters:
        -----------
        specifier : tuple
            The parameter order specifier to expand.

        Returns:
        --------
        new_specifiers : List[tuple]
            A list of the expanded parameter order specifiers.
        """
        d = len(specifier) # Get the dimension of the tuple (corresponding to the number of input variables)
        new_specifiers = []
        for i in range(d):
            # Increment the ith element, keeping the others the same
            new_tuple = tuple(n + 1 if idx == i else n for idx, n in enumerate(specifier))
            new_specifiers.append(new_tuple)
        return new_specifiers
    
    new_specifiers = [] # Initialize an empty list to store the new specifiers
    for specifier in model_order_specifier:
        # Expand each individual parameter specifier
        candidate_specifiers = _expand_single_parameter_specifier(specifier)
        # Add any candidate specifiers that are not already in the model order specifier
        for candidate_specifier in candidate_specifiers:
            if (candidate_specifier not in model_order_specifier) and (candidate_specifier not in new_specifiers):
                new_specifiers.append(candidate_specifier)

    # Assemble the new model order specifier
    new_model_order_specifier = model_order_specifier + new_specifiers

    # Construct the expansion matrix
    nr_of_selected_parameters = len(model_order_specifier)
    nr_of_new_parameters = len(new_specifiers)
    expansion_matrix = np.vstack([
        np.eye(nr_of_selected_parameters),
        np.zeros((nr_of_new_parameters, nr_of_selected_parameters))
    ])

    return new_model_order_specifier, expansion_matrix

def _taylor_expansion_update_operator(model_order_specifier, decision_index):
    """
    The update operator implementation for the Taylor expansion model.
    
    This function adds a new parameter to the model order specifier, based on the decision index.

    Parameters:
    -----------
    model_order_specifier : List[tuple]
        The current model order specifier.
    decision_index : int
        The decision index.

    Returns:
    --------
    new_model_order_specifier : List[tuple]
        The updated model order specifier.
    selection_matrix : np.ndarray
        The selection matrix.
    """
    # Get the full model order specifier and the expansion matrix
    full_model_order_specifier, expansion_matrix = _taylor_expansion_expansion_operator(model_order_specifier)
    nr_of_parameters_full = len(full_model_order_specifier)

    if decision_index < len(model_order_specifier):
        # If the decision index is one of the existing parameters,
        # the selection matrix will be the transpose of the expansion matrix.
        # No new parameters will be added.
        selection_matrix = expansion_matrix.T
        new_model_order_specifier = model_order_specifier
    else:
        # If the decision index is one of the new parameters,
        # we need to add a new parameter to the model.
        selection_matrix = np.vstack([
            expansion_matrix.T,
            np.zeros(nr_of_parameters_full)
        ])
        selection_matrix[-1, decision_index] = 1 # Select the new parameter
        new_model_order_specifier = model_order_specifier + [full_model_order_specifier[decision_index]]

    return new_model_order_specifier, selection_matrix

class _IntAsLen:
    def __init__(self, value):
        if not isinstance(value, int):
            raise TypeError("Value must be an int.")
        self.value = value

    def __len__(self):
        return self.value
