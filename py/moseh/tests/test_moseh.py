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

import pytest
import moseh.moseh as moseh
import numpy as np

def test_moseh_solve_call():
    """
    Test the moseh.solve function call with a dummy model function.
    """
    # Arrange - Set up a dummy model function, Jacobian function, expansion operator and update operator
    def model_function(model_order_specifier, beta, input_data):
        data_length = len(input_data)
        return np.arange(1, data_length + 1)

    def jacobian_function(model_order_specifier, beta, input_data):
        data_length = len(input_data)
        model_order = len(model_order_specifier)
        if model_order > data_length:
            raise ValueError("Model order cannot be greater than data length")
        jacobian = np.zeros((data_length, model_order))
        for i in range(model_order):
            jacobian[i, i] = 1
        return jacobian

    def expansion_operator(model_order_specifier):
        model_order = len(model_order_specifier)
        expansion_matrix = np.vstack((np.eye(model_order), np.zeros(model_order)))
        new_model_order_specifier = moseh._IntAsLen(model_order + 1)
        return new_model_order_specifier, expansion_matrix

    def update_operator(model_order_specifier, decision_index):
        model_order = len(model_order_specifier)
        if decision_index < model_order:
            selection_matrix = np.column_stack((np.eye(model_order), np.zeros(model_order)))
            new_model_order_specifier = model_order_specifier
        else:
            selection_matrix = np.eye(model_order + 1)
            new_model_order_specifier = moseh._IntAsLen(model_order + 1)
        return new_model_order_specifier, selection_matrix

    # Set up the initial model order specifier
    initial_model_order_specifier = 2

    # Set up the input and target data
    input_data = np.array([1, 2, 3, 4, 5])
    target_data = np.array([2, 4, 6, 8, 10])

    # Act
    _, _, converged = moseh.solve(
        input_data, target_data,
        model_function, jacobian_function,
        initial_model_order_specifier,
        expansion_operator,
        update_operator
    )

    # Assert
    assert converged

def dummy_linear_model_residual_function(theta, x_matrix, y):
    """
    Returns the residual = x_matrix @ theta - y.
    Suitable for linear regression-like scenarios.

    Parameters
    ----------
    theta : np.ndarray
        Parameter vector.
    x_matrix : np.ndarray
        Design matrix.
    y : np.ndarray
        Target vector.

    Returns
    -------
    residual : np.ndarray
        The residual vector.
    """
    return x_matrix @ theta - y

def dummy_compute_linear_model_score_and_fim(theta, x_matrix, y):
    """
    Example of computing a score and FIM for a simple linear model:

        y = x_matrix @ theta

    For least-squares, the gradient (score) is x_matrix.T @ (y - x_matrix @ theta)
    and the Fisher information matrix is x_matrix.T @ x_matrix / variance.

    Parameters
    ----------
    theta : np.ndarray
        Parameter vector.
    x_matrix : np.ndarray
        Design matrix.
    y : np.ndarray
        Target vector.
    
    Returns
    -------
    score : np.ndarray
        The score vector.
    fim : np.ndarray
        The Fisher information matrix (FIM)
    """
    residual = x_matrix @ theta - y
    variance = np.sum(residual**2) / (len(y) - len(theta))
    score = -x_matrix.T @ residual / variance
    fim = x_matrix.T @ x_matrix / variance
    
    return score, fim

@pytest.mark.parametrize(
    "theta, score, fim",
    [
        (
            np.array([0., 0.]),
            np.array([0.85714286, 2.]),
            np.array(
                [[0.57142857, 0.85714286],
                 [0.85714286, 2.        ]]
            )
        )
    ]
)
def test_fisher_scoring_no_line_search(theta, score, fim):
    """
    Test the Fisher scoring step without line search.

    Parameters
    ----------
    theta : np.ndarray
        Initial parameter vector.
    input_data : np.ndarray
        The input data matrix.
    target_data : np.ndarray
        The target data array.
    """
    # Arrange - Compute the expected updated theta
    fisher_step = np.linalg.solve(fim, score)
    theta_target = theta + fisher_step

    # Act
    theta_new = moseh._fisher_scoring_step(
        score=score,
        fim=fim,
        theta=theta
    )

    # Assert
    np.testing.assert_allclose(theta_new, theta_target, atol=1e-7)

@pytest.mark.parametrize(
    "theta, input_data, target_data, theta_target",
    [
        (
            np.array([0., 0.]),
            np.array(
                [[1., 0.],
                [1., 1.],
                [1., 2.],
                [1., 3.]]
            ),
            np.array([0., 1., 2., 3.]),
            np.array([0., 1.])
        ),
        (
            np.array([0., 0.]),
            np.array(
                [[1., 0.],
                [1., 1.],
                [1., 2.],
                [1., 3.]]
            ),
            np.array([1., 2., 3., 4.]),
            np.array([1., 1.])
        ),
    ]
)
def test_fisher_scoring_line_search_converges(theta, input_data, target_data, theta_target):
    """
    Test the Fisher scoring step with line search.

    Parameters
    ----------
    theta : np.ndarray
        Initial parameter vector.
    input_data : np.ndarray
        The input data matrix.
    target_data : np.ndarray
        The target data array.
    theta_target : np.ndarray
        The expected updated parameter vector.
    """
    # Arrange
    score, fim = dummy_compute_linear_model_score_and_fim(theta, input_data, target_data)

    # Act
    theta_new = moseh._fisher_scoring_step(
        score=score,
        fim=fim,
        theta=theta,
        input_data=input_data,
        target_data=target_data,
        residual_function=dummy_linear_model_residual_function,  # Enable line search
        armijo_constant=1e-4,
        max_iterations=10
    )

    # Assert
    np.testing.assert_allclose(theta_new, theta_target, atol=1e-7)

def test_fisher_scoring_line_search_fails():
    """
    Force the line search to fail by making the FIM extremely small
    (leading to a huge fisher_step) and an extreme score.
    """
    # Arrange - Set up a simple 2D linear model
    input_data = np.eye(2)
    target_data = np.array([10., 10.])
    theta = np.array([0., 0.])

    # Artificially inflate the score and shrink the FIM to get a very large Fisher step
    score = np.array([1e9, 1e9])
    fim = np.eye(2) * 1e-12

    # Act & Assert
    with pytest.raises(ValueError, match="backtracking line search did not converge"):
        _ = moseh._fisher_scoring_step(
            score=score,
            fim=fim,
            theta=theta,
            input_data=input_data,
            target_data=target_data,
            residual_function=dummy_linear_model_residual_function,
            armijo_constant=1e-4,
            max_iterations=3
        )
