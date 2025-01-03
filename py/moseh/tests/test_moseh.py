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
    def residual_function(model_order_specifier, beta):
        return np.array([1, 2, 3, 4, 5])

    def jacobian_function(model_order_specifier, beta):
        return np.array([[1, 1], [1, 0], [1, 0], [1, 0], [1, 0]])

    def expansion_operator(model_order_specifier, beta, covariance):
        return np.array([1, 2, 3, 4, 5]), np.array([[1, 1, 0], [1, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0]]), 1

    def update_operator(model_order_specifier, decision_index, beta):
        return model_order_specifier, beta

    initial_model_order_specifier = 2

    model_order_specifier, beta, converged = moseh.solve(
        residual_function, jacobian_function,
        initial_model_order_specifier,
        expansion_operator,
        update_operator,
        beta=np.zeros(2),
        max_iterations=100
    )
