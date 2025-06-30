import os
import subprocess       
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)     

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self.counts = {}

    def update(self, metrics_dict: dict, n=1):
        """
        Update the metrics with new values.

        Args:
            metrics_dict (dict): A dictionary of metric names and their values.
            n (int): nextber of samples in the batch.
        """
        for metric_name, value in metrics_dict.items():
            if metric_name not in self.metrics:
                self.metrics[metric_name] = 0.0
                self.counts[metric_name] = 0
            self.metrics[metric_name] += value * n
            self.counts[metric_name] += n

    def get_average(self):
        """
        Compute and return the average of all metrics.

        Returns:
            dict: A dictionary of metric names and their average values.
        """
        averages = {}
        for metric_name, total in self.metrics.items():
            averages[metric_name] = total / self.counts[metric_name]
        return averages

    def __str__(self):
        """String representation of the average metrics."""
        return ", ".join([f"{k}: {v:.4f}" for k, v in self.get_average().items()])


def get_size(start_path: str) -> float:
    total_size = 0
    for dirpath, _, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size / (1024 ** 3)


def is_awscli_installed():
    try:
        result = subprocess.run(
            ["aws", "--version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True)
        
        if result.returncode == 0:
            return True
        else:
            return False
    except FileNotFoundError:
        return False


def get_last_n_path_elements(path: str, n: int) -> str:
    """
    Extracts the last `n` elements from a given file path 
    and joins them into a new path.

    This method normalizes the input path 
    and then splits it into its constituent elements. 
    It selects the last `n` elements and joins them
    back into a path using the operating system's path separator.

    Args:
        path (str): The input file path from which to extract elements.
        n (int): The number of trailing path elements to extract. 
                If `n` exceeds the number of elements
                in the path, the entire normalized path is returned.

    Returns:
        str: A new path consisting of the last `n` elements of the input path, 
                joined by the OS path separator.
    """
    return os.sep.join(os.path.normpath(path).split(os.sep)[-n:])


def get_leaf_dirs(root_folder: str) -> list:
    """
    Retrieves a list of directories within the specified root folder 
    that do not contain any subdirectories.

    This method traverses the directory tree 
    starting from `root_folder` and identifies all directories
    that are "leaf" nodes (i.e., directories with no subdirectories). 
    These directories are returned as a list of paths.

    Args:
        root_folder (str): The path to the root directory 
                from which to start searching for leaf directories.

    Returns:
        list: A list of paths to directories 
                that have no subdirectories. Each path is a string.
    """
    return [root for root, dirs, _ in os.walk(root_folder)
            if len(dirs) == 0]
