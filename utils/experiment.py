import os
import atexit
import torch
import math
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm


class Experiment:
    def __init__(self,
                 model: torch.nn.Module,
                 category: str | list[str] = None,
                 root: str = 'experiments'):
        self.model = model
        self.writer = SummaryWriter()

        # Set up folder structure
        now = datetime.now()
        dt_path = os.path.join(now.strftime('%m_%d_%Y'), now.strftime('%H_%M_%S'))
        self.path = os.path.join(
            root,
            os.path.join(category) if category else model.__class__.__name__,
            dt_path
        )
        if not os.path.isdir(self.path):
            os.makedirs(self.path)

    def loop(self, n, func, **kwargs):
        """
        Runs the given function for N steps and saves the model whenever the
        loop terminates (possibly prematurely).
        """

        # Save model whenever training loop terminates, just in case of crash
        max_digits = int(math.log10(n - 1) + 1)
        atexit.register(lambda: self.save_model(str(step).zfill(max_digits)))

        # Run for N iterations
        kwargs['desc'] = kwargs.get('desc', func.__name__)
        for step in tqdm(range(n), **kwargs):
            func(step)

        self.writer.flush()

    def add_scalar(self, name, step, value):
        """Appends scalar for this step to its corresponding .csv file"""

        name, _ = os.path.splitext(name)

        # Write to Tensorboard
        self.writer.add_scalar(name, value, step)

        # Re-group path b/c name might be a nested path
        folder, file_name = os.path.split(os.path.join(self.path, 'scalars', f'{name}.csv'))
        folder = folder.lower()
        if not os.path.isdir(folder):
            os.makedirs(folder)

        # Append to .csv file
        with open(os.path.join(folder, file_name), 'a') as file:
            file.write(f'{step}, {value}\n')

    def save_model(self, file_name):
        """Saves model weights to the weights directory under FILE_NAME."""

        weight_dir = os.path.join(self.path, 'weights')
        if not os.path.isdir(weight_dir):
            os.makedirs(weight_dir)

        torch.save(
            self.model.state_dict(),
            os.path.join(weight_dir, file_name)
        )
