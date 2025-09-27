import numpy as np
from torchvision import datasets, transforms


def load_mnist(flatten=True):
    """Load MNIST dataset.

    Args:
        flatten (bool): If True, flatten 28x28 images to 784-dimensional vectors.

    Returns:
        tuple:
            - data (np.ndarray): images as arrays (flattened if flatten=True)
            - targets (np.ndarray): corresponding labels
    """
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    data = mnist.data.numpy()
    targets = mnist.targets.numpy()

    if flatten:
        data = data.reshape((data.shape[0], -1))
    return data, targets


def create_bags_or(data, targets, bag_size=10, num_bags=1000, key_digit=3, key_instances_per_bag=1, random_state=42):
    """Create OR-type MIL bags. Positive bags contain at least one key
    instance.

    Args:
        data (np.ndarray): instance data
        targets (np.ndarray): instance labels
        bag_size (int): number of instances per bag
        num_bags (int): number of bags to generate
        key_digit (int): digit considered as key instance
        key_instances_per_bag (int): number of key instances in positive bags
        random_state (int): random seed

    Returns:
        tuple:
            - bags (list of np.ndarray): list of bags
            - bag_labels (list of int): bag-level labels (0/1)
            - key_indices_per_bag (list of list of int): positions of key instances in each bag
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

        bag = data[full_indices]
        bags.append(bag)
        bag_labels.append(label)
        key_indices_per_bag.append(key_pos_in_bag)

    return bags, bag_labels, key_indices_per_bag
