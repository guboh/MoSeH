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

import moseh.moseh as moseh
import numpy as np

def test_moseh():
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

    initial_model_order_specifier = 2

    input_data = np.array([1, 2, 3, 4, 5])
    target_data = np.array([2, 4, 6, 8, 10])

    model_order_specifier, beta, converged = moseh.solve(
        input_data, target_data,
        model_function, jacobian_function,
        initial_model_order_specifier,
        expansion_operator,
        update_operator
    )
