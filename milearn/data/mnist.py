import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


def load_mnist(flatten=True):
    """
    Load MNIST dataset as NumPy arrays.

    Args:
        flatten (bool): If True, flatten each 28x28 image into a 784-length vector.

    Returns:
        data (np.ndarray): array of images, shape [n_samples, 28, 28] or [n_samples, 784]
        targets (np.ndarray): array of digit labels, shape [n_samples]
    """
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    data = mnist.data.numpy()
    targets = mnist.targets.numpy()

    if flatten:
        data = data.reshape((data.shape[0], -1))
    return data, targets


def create_bags_or(data, targets, bag_size=10, num_bags=1000, key_digit=3, key_instances_per_bag=1, random_state=42):
    """
    Create bags for MIL using OR logic: bag is positive if it contains at least one key digit.

    Args:
        data (np.ndarray): instance features, shape [n_samples, n_features]
        targets (np.ndarray): instance labels, shape [n_samples]
        bag_size (int): number of instances per bag
        num_bags (int): total number of bags to create
        key_digit (int): digit considered as key
        key_instances_per_bag (int): number of key instances per positive bag
        random_state (int): random seed

    Returns:
        bags (list of np.ndarray): list of bags, each shape [bag_size, n_features]
        bag_labels (list): 0/1 labels for each bag
        key_indices_per_bag (list of list): positions of key instances in each bag
    """
    rng = np.random.RandomState(random_state)
    key_indices_all = np.where(targets == key_digit)[0]
    non_key_indices_all = np.where(targets != key_digit)[0]

    bags, bag_labels, key_indices_per_bag = [], [], []

    for _ in range(num_bags):
        is_positive = rng.rand() < 0.5

        if is_positive:
            key_sample_indices = rng.choice(key_indices_all, size=key_instances_per_bag, replace=False)
            remaining = bag_size - key_instances_per_bag
            non_key_sample_indices = rng.choice(non_key_indices_all, size=remaining, replace=False)
            full_indices = np.concatenate([key_sample_indices, non_key_sample_indices])
            rng.shuffle(full_indices)
            key_pos_in_bag = [i for i, idx in enumerate(full_indices) if idx in key_sample_indices]
            label = 1
        else:
            full_indices = rng.choice(non_key_indices_all, size=bag_size, replace=False)
            key_pos_in_bag = []
            label = 0

        bags.append(data[full_indices])
        bag_labels.append(label)
        key_indices_per_bag.append(key_pos_in_bag)

    return bags, bag_labels, key_indices_per_bag


def create_bags_and(data, targets, bag_size=10, num_bags=1000, key_digits=(3, 7), random_state=42):
    """
    Create bags for MIL using AND logic: bag is positive if it contains both key digits.

    Args:
        data (np.ndarray): instance features
        targets (np.ndarray): instance labels
        bag_size (int): number of instances per bag
        num_bags (int): total number of bags
        key_digits (tuple): pair of digits considered as key
        random_state (int): random seed

    Returns:
        bags (list of np.ndarray): list of bags
        bag_labels (list): 0/1 labels for each bag
        key_indices_per_bag (list of list): positions of key instances in each bag
    """
    rng = np.random.RandomState(random_state)
    idx_key1 = np.where(targets == key_digits[0])[0]
    idx_key2 = np.where(targets == key_digits[1])[0]
    idx_nonkey = np.where(~np.isin(targets, key_digits))[0]

    bags, bag_labels, key_indices_per_bag = [], [], []

    for _ in range(num_bags):
        is_positive = rng.rand() < 0.5

        if is_positive:
            idx1 = rng.choice(idx_key1)
            idx2 = rng.choice(idx_key2)
            remaining = bag_size - 2
            idx_other = rng.choice(idx_nonkey, size=remaining, replace=False)
            full_indices = np.array([idx1, idx2] + list(idx_other))
            rng.shuffle(full_indices)
            label = 1
            key_pos = [i for i, idx in enumerate(full_indices) if idx in [idx1, idx2]]
        else:
            while True:
                full_indices = rng.choice(len(targets), size=bag_size, replace=False)
                bag_targets = targets[full_indices]
                if not (key_digits[0] in bag_targets and key_digits[1] in bag_targets):
                    break
            label = 0
            key_pos = []

        bags.append(data[full_indices])
        bag_labels.append(label)
        key_indices_per_bag.append(key_pos)

    return bags, bag_labels, key_indices_per_bag


def create_bags_xor(data, targets, bag_size=10, num_bags=1000, key_digits=(3, 7), random_state=42):
    """
    Create bags for MIL using XOR logic: bag is positive if it contains exactly one of the key digits.

    Args:
        data (np.ndarray): instance features
        targets (np.ndarray): instance labels
        bag_size (int): number of instances per bag
        num_bags (int): total number of bags
        key_digits (tuple): pair of digits considered as key
        random_state (int): random seed

    Returns:
        bags (list of np.ndarray): list of bags
        bag_labels (list): 0/1 labels
        key_indices_per_bag (list of list): positions of key instances in each bag
    """
    rng = np.random.RandomState(random_state)
    idx_key1 = np.where(targets == key_digits[0])[0]
    idx_key2 = np.where(targets == key_digits[1])[0]
    idx_nonkey = np.where(~np.isin(targets, key_digits))[0]

    bags, bag_labels, key_indices_per_bag = [], [], []

    for _ in range(num_bags):
        is_positive = rng.rand() < 0.5

        if is_positive:
            use_digit = key_digits[rng.randint(2)]
            idx_key = rng.choice(idx_key1 if use_digit == key_digits[0] else idx_key2)
            idx_other = rng.choice(idx_nonkey, size=bag_size - 1, replace=False)
            full_indices = np.array([idx_key] + list(idx_other))
            rng.shuffle(full_indices)
            label = 1
            key_pos = [i for i, idx in enumerate(full_indices) if targets[idx] == use_digit]
        else:
            while True:
                full_indices = rng.choice(len(targets), size=bag_size, replace=False)
                bag_targets = targets[full_indices]
                count_1 = np.sum(bag_targets == key_digits[0])
                count_2 = np.sum(bag_targets == key_digits[1])
                if (count_1 > 0 and count_2 > 0) or (count_1 == 0 and count_2 == 0):
                    break
            label = 0
            key_pos = []

        bags.append(data[full_indices])
        bag_labels.append(label)
        key_indices_per_bag.append(key_pos)

    return bags, bag_labels, key_indices_per_bag


def create_bags_reg(data, targets, bag_size=5, num_bags=1000, bag_agg="mean", random_state=42):
    """
    Create regression-style bags: bag label is aggregated from instance labels.

    Args:
        data (np.ndarray): instance features
        targets (np.ndarray): instance labels (numeric)
        bag_size (int): instances per bag
        num_bags (int): number of bags
        bag_agg (str): aggregation method for labels ('mean' or 'sum')
        random_state (int): random seed

    Returns:
        bags (list of np.ndarray): list of bags
        labels (list): aggregated bag labels
        instance_digits (list of list): original instance labels per bag
    """
    if bag_agg == "mean":
        agg_func = np.mean
    elif bag_agg == "sum":
        agg_func = np.sum
    else:
        raise TypeError(f"Unknown value for bag_agg: {bag_agg}")

    rng = np.random.RandomState(random_state)
    indices = np.arange(len(data))
    bags, labels, instance_digits = [], [], []

    for _ in range(num_bags):
        selected_indices = rng.choice(indices, size=bag_size, replace=False)
        bag = data[selected_indices]
        digits = targets[selected_indices]
        label = agg_func(digits)
        bags.append(bag)
        labels.append(label.item())
        instance_digits.append(digits.tolist())

    return bags, labels, instance_digits


def show_digit(vector, title=None):
    """
    Display a single MNIST digit vector as a 28x28 image.

    Args:
        vector (np.ndarray): flattened digit of length 784
        title (str): optional figure title
    """
    if vector.shape[0] != 784:
        raise ValueError("Expected a vector of length 784.")

    image = vector.reshape(28, 28)
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()


def visualize_bag_with_weights(bag, weights, digits=None, title=None, cmap='gray', sort=False):
    """
    Visualize instances in a bag with attention weights.

    Args:
        bag (list or np.ndarray): images, each flattened or [28,28]
        weights (list or np.ndarray): attention weights for each instance
        digits (list): optional labels of instances (used for sorting)
        title (str): optional figure title
        cmap (str): colormap for images
        sort (bool): sort images by digits if provided
    """
    bag = np.array(bag)
    weights = np.array(weights)

    if sort and digits is not None:
        digits = np.array(digits)
        sort_idx = np.argsort(digits)
        bag = bag[sort_idx]
        weights = weights[sort_idx]

    bag_size = len(bag)
    cols = min(5, bag_size)
    rows = (bag_size + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows))
    axes = axes.flatten()

    for i in range(bag_size):
        image = bag[i].reshape(28, 28)
        axes[i].imshow(image, cmap=cmap)
        axes[i].set_title(f"{weights[i]:.2f}", fontsize=12)
        axes[i].axis("off")

    for i in range(bag_size, len(axes)):
        axes[i].axis("off")

    if title:
        fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
