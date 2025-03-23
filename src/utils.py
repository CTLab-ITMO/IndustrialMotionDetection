import os
import subprocess            

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
        # Run the 'aws --version' command
        result = subprocess.run(
            ["aws", "--version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True)
        
        if result.returncode == 0:
            print("awscli is installed.")
            print(f"Version: {result.stdout.strip()}")
            return True
        else:
            print("awscli is NOT installed.")
            return False
    except FileNotFoundError:
        print("awscli is NOT installed.")
        return False
