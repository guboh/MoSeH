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
    _, _, converged, _ = moseh.solve(
        input_data, target_data,
        model_function, jacobian_function,
        initial_model_order_specifier,
        expansion_operator,
        update_operator
    )

    # Assert
    assert converged

def dummy_linear_model_function(theta, x_matrix):
    """
    Returns the estimated output for a linear model:

        y = x_matrix @ theta + noise

    Parameters
    ----------
    theta : np.ndarray
        Parameter vector.
    x_matrix : np.ndarray
        Design matrix.

    Returns
    -------
    output : np.ndarray
        The model output.
    """
    return x_matrix @ theta

def dummy_linear_jacobian_function(theta, x_matrix):
    """
    Returns the Jacobian matrix for a linear model:

        y = x_matrix @ theta + noise

    Parameters
    ----------
    theta : np.ndarray
        Parameter vector.
    x_matrix : np.ndarray
        Design matrix.
    
    Returns
    -------
    jacobian_matrix : np.ndarray
        The Jacobian matrix.
    """
    return x_matrix

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
def test_fisher_scoring_no_line_search(theta, input_data, target_data, theta_target):
    """
    Test the Fisher scoring algorithm without line search.

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
    # Act
    theta_new, converged = moseh._fisher_scoring(
        model_function=dummy_linear_model_function,
        jacobian_function=dummy_linear_jacobian_function,
        theta=theta,
        input_data=input_data,
        target_data=target_data,
        line_search=False
    )

    # Assert
    np.testing.assert_allclose(theta_new, theta_target, atol=1e-7)
    assert converged
