"""Utility helpers for device setup, metrics, scheduling, and experiment logging."""

import math
import os
from collections.abc import Callable
from datetime import datetime
from functools import wraps
from time import time
from typing import ParamSpec, TypeVar
import pytz
import torch
import wandb
from dotenv import load_dotenv
from torch.optim.lr_scheduler import LambdaLR

P = ParamSpec("P")
T = TypeVar("T")


def torch_get_device(device_type):
    """Resolve the requested torch device.

    Returns:
        torch.device: Selected execution device.
    """
    if device_type == "cuda":
        assert torch.cuda.is_available(), "CUDA is not available :(, `python train.py +device=auto`"
        device = torch.device("cuda")
    elif device_type == "auto":
        assert not torch.cuda.is_available(), "CUDA is available :), switch to cuda"
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            try:
                import torch_xla.core.xla_model as xm  # type: ignore

                device = xm.xla_device()
            except ImportError:
                device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    return device


def torch_set_seed(seed):
    """Seed torch RNGs for reproducible runs.

    Returns:
        None: Updates torch RNG state in place.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def torch_compile_ckpt_fix(state_dict):
    """Strip ``torch.compile`` parameter prefixes from a checkpoint state dict.

    Returns:
        dict: State dict with compile-specific prefixes removed.
    """
    # when torch.compiled a model, state_dict is updated with a prefix '_orig_mod.', renaming this
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    return state_dict


def get_ist_time_now(fmt="%d-%m-%Y-%H%M%S"):
    """Format the current India Standard Time timestamp.

    Returns:
        str: Formatted IST timestamp.
    """
    tz = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(tz)
    return now_ist.strftime(fmt)


def cosine_with_linear_warmup_lr_scheduler(
    optimizer, total_steps, warmup_pct, decay_step_pct, min_lr_pct
):
    """Create a warmup-plus-cosine learning-rate schedule.

    Returns:
        LambdaLR: Scheduler with warmup, cosine decay, and final constant phase.
    """
    warmup_steps = int(warmup_pct * total_steps)
    decay_steps = warmup_steps + int(decay_step_pct * total_steps)

    def lr_lambda(current_step):
        """Compute the LR multiplier for the current optimization step.

        Returns:
            float: Learning-rate multiplier for ``current_step``.
        """
        if current_step < warmup_steps:
            # warmup
            return float(current_step + 1) / float(max(1, warmup_steps))
        elif current_step > decay_steps:
            # constant
            return min_lr_pct
        else:
            # cosine
            progress = float(current_step - warmup_steps) / float(
                max(1, decay_steps - warmup_steps)
            )
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_pct + (1 - min_lr_pct) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def calc_accuracy(output, target, topk=(1,)):
    """Compute top-k accuracy scores for a batch of predictions.

    Returns:
        list[torch.Tensor]: Accuracy tensors in the same order as ``topk``.
    """
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res


class AverageMetric:
    """Track the running average of a scalar metric."""

    def __init__(self):
        """Initialize an empty running metric.

        Returns:
            None: This initializer does not return a value.
        """
        self.reset()

    def reset(self):
        """Reset the metric state to zero.

        Returns:
            None: Clears the accumulated statistics in place.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update the metric with a new value and sample count.

        Returns:
            None: Updates the running statistics in place.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


def timer(func: Callable[P, T]) -> Callable[P, tuple[float, T]]:
    """Wrap a callable and return its runtime alongside its result.

    Returns:
        Callable[P, tuple[float, T]]: Decorated function returning elapsed seconds and result.
    """
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> tuple[float, T]:
        """Execute the wrapped callable and measure elapsed wall time.

        Returns:
            tuple[float, T]: Runtime in seconds and the wrapped return value.
        """
        t0 = time()
        ret = func(*args, **kwargs)
        t = time() - t0
        return t, ret

    return wrapper


class WandBLogger:
    """Thin wrapper around Weights & Biases logging used by training."""

    def __init__(self, project, run, config, tags, metrics, enable=False):
        """Initialize the optional W&B logger.

        Returns:
            None: Configures W&B state when logging is enabled.
        """
        self.enable = enable
        if self.enable:
            load_dotenv()
            wnb_key = os.getenv("WANDB_API_KEY")
            assert wnb_key is not None, "WANDB_API_KEY not loaded in env"
            wandb.login(key=wnb_key)
            wandb.init(project=project, name=run, config=config)
            self.tags = tags
            self.metrics = metrics
            wandb.define_metric("epoch")
            for tag in tags:
                for metric in metrics:
                    wandb.define_metric(f"{tag}/{metric}", step_metric="epoch")

    def log(self, tag, data):
        """Log a metric payload under the configured tag namespace.

        Returns:
            None: Sends metrics to W&B when enabled.
        """
        if self.enable:
            assert tag in self.tags, f"{tag=} not created"
            new_data = {"epoch": data["epoch"]}
            for metric, val in data.items():
                if metric == "epoch":
                    continue
                assert metric in self.metrics, f"{metric=} not created"
                new_data[f"{tag}/{metric}"] = val
            wandb.log(new_data)
