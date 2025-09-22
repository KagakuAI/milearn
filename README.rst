``milearn``: Multi-instance machine learning in Python
==========================================================

``milearn`` is designed to mimic the scikit-learn interface to simplify its usage and integration with other tools.

Key Features
------------

- Traditional and neural network-based MIL algorithms (regression and classification)
- Integrated stepwise model hyperparameter optimization (recommended for small datasets)

Installation
------------

.. code-block:: bash

    pip install milearn

Quick Start
-----------

.. code-block:: python

    from milearn.data.mnist import load_mnist, create_bags_reg
    from milearn.preprocessing import BagMinMaxScaler
    from sklearn.model_selection import train_test_split
    from milearn.network.module.hopt import DEFAULT_PARAM_GRID
    from milearn.network.regressor import DynamicPoolingNetworkRegressor

    # 1. Create MNIST regression dataset
    data, targets = load_mnist()
    bags, labels, key = create_bags_reg(data, targets, bag_size=10, num_bags=10000,
                                        bag_agg="mean", random_state=42)

    # 2. Train/test split and scale features
    x_train, x_test, y_train, y_test, key_train, key_test = train_test_split(bags, labels, key, random_state=42)
    scaler = BagMinMaxScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # 3. Train model
    model = DynamicPoolingNetworkRegressor()
    model.hopt(x_train_scaled, y_train, # recomended for small datasets only
               param_grid=DEFAULT_PARAM_GRID, verbose=True)
    model.fit(x_train_scaled, y_train)

    # 4. Get predictions
    y_pred = model.predict(x_test_scaled) # predicted labels
    w_pred = model.get_instance_weights(x_test_scaled) # predicted instance weights

Tutorials
-----------

Several examples of the ``milearn`` application to the classification/regression problem and key instance detection 
for the MNIST dataset can be found in `notebooks <notebooks>`_ .

Paper
-----------
Application cases demonstrated in the paper can be found in:

- MNIST classification: `Notebook <https://github.com/KagakuAI/milearn/blob/main/notebooks/Tutorial_2_KID_for_mnist_classification.ipynb>`_

- MNIST regression: `Notebook <https://github.com/KagakuAI/milearn/blob/main/notebooks/Tutorial_3_KID_for_mnist_regression.ipynb>`_

- Molecular conformers: `Notebook <https://github.com/KagakuAI/QSARmil/blob/main/notebooks/Tutorial_2_KID_for_conformers.ipynb>`_

- Molecular fragments: `Notebook <https://github.com/KagakuAI/QSARmil/blob/main/notebooks/Tutorial_4_KID_for_fragments.ipynb>`_

- Protein protein interaction: `Notebook <https://github.com/KagakuAI/SEQmil/blob/main/notebooks/Tutorial_1_KID_for_protein_protein_interaction.ipynb>`_

