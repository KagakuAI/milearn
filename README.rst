``milearn``: A General-Purpose Multi-Instance Learning Toolkit
==========================================================

``milearn`` is a flexible Python toolkit for building and evaluating models using **Multi-Instance Learning (MIL)**. 
It is designed for general-purpose use in machine learning applications where data is structured as bags of instances with only bag-level supervision.

Key Features
------------

- 🧠 Varieties of traditional, upgraded, and neural network-based supervised learning algorithms
- 🔎 Key instance detection with model-based and model-agnostic approaches
- ⚙️ Scikit-learn compatible API for easy integration

What is Multi-Instance Learning?
--------------------------------

In **Multi-Instance Learning (MIL)**, data is grouped into **bags**, each containing multiple **instances** 
(e.g., data points, documents, image patches). Only the **bag** has a label — not the individual instances.

**Example:**

- An email (bag) is labeled spam if **at least one sentence (instance)** is spammy
- A video (bag) is labeled as violent if **at least one frame (instance)** shows violence
- A document (bag) is classified as toxic based on certain **key phrases** (instances)
- A molecule (bag) is represented by its **conformers** or **fragments** (instances)
- A biological sequence (bag) is broken into **subsequences** of RNA, DNA, or protein domains (instances)

Installation
------------

Install directly from GitHub:

.. code-block:: bash

    # Create and activate a new conda environment
    conda create -n milearn python=3.9 -y
    conda activate milearn

    # Install directly from GitHub
    pip install git+https://github.com/KagakuAI/milearn.git

The installed ``milearn`` environment can then be added to the Jupyter platform:

.. code-block:: bash

    conda install ipykernel
    python -m ipykernel install --user --name milearn --display-name "milearn"


Quick Start
-----------

Several examples of the ``milearn`` application to the classification/regression problem and key instance detection 
for the MNIST dataset can be found in `tutorials <tutorials>`_ .

