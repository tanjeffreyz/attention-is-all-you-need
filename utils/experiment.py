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
        self.writer = SummaryWriter(log_dir='.runs')

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

    def append_loss(self, loss_type, epoch, loss):
        """Appends loss for this epoch to its corresponding .csv file"""

        loss_type = loss_type.lower()
        assert loss_type in {'train', 'validation'}

        # Write to Tensorboard
        self.writer.add_scalar(f'Loss/{loss_type}', loss, epoch)

        # Append to .csv file
        with open(os.path.join(self.path, f'{loss_type}_loss.csv'), 'a') as file:
            file.write(f'{epoch}, {loss}\n')

    def save_model(self, file_name):
        """Saves model weights to the weights directory under FILE_NAME."""

        weight_dir = os.path.join(self.path, 'weights')
        if not os.path.isdir(weight_dir):
            os.makedirs(weight_dir)

        torch.save(
            self.model.state_dict(),
            os.path.join(weight_dir, file_name)
        )
