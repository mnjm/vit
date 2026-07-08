"""Compute train and validation loss/accuracy for a saved checkpoint."""

import argparse
import logging

import torch

from data import init_dataloaders
from model import init_deit, init_model
from utils import ClassificationMetrics, torch_compile_ckpt_fix, torch_get_device, torch_set_seed


def evaluate(model: torch.nn.Module, dataloader, device: torch.device) -> dict[str, float]:
    """Evaluate a model on one dataloader.

    Args:
        model: Model to evaluate.
        dataloader: Dataloader providing ``(image, label)`` batches.
        device: Device used for evaluation.

    Returns:
        dict[str, float]: Aggregate loss and accuracy metrics.
    """
    metrics = ClassificationMetrics()
    model.eval()
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            logits, loss = model(imgs, lbls)
            metrics.update(logits, lbls, loss)
    return metrics.compute()


def main() -> None:
    """Load a checkpoint and report train/validation metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to a saved training checkpoint")
    parser.add_argument("--device", default=None, help="Override checkpoint device_type")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    cfg.interactive = False
    cfg.torch_compile = False
    cfg.dataloader.drop_last = False
    if args.device is not None:
        cfg.device_type = args.device

    torch_set_seed(cfg.rng_seed)
    device = torch_get_device(cfg.device_type)

    train_dataloader, val_dataloader = init_dataloaders(cfg)
    model = init_model(cfg, device)
    model.to(device)
    model.load_state_dict(torch_compile_ckpt_fix(ckpt["model"]), strict=False)

    if hasattr(cfg, "deit") and getattr(cfg.deit, "enable", False):
        init_deit(model, cfg, device, logger=logging.getLogger("compute_acc"))

    train_metrics = evaluate(model, train_dataloader, device)
    val_metrics = evaluate(model, val_dataloader, device)

    print(f"checkpoint: {args.checkpoint}")
    print(
        "train:",
        f"loss={train_metrics['loss']:.4f}",
        f"acc={train_metrics['acc@1']:.2%}",
        f"acc@5={train_metrics['acc@5']:.2%}",
    )
    print(
        "val:",
        f"loss={val_metrics['loss']:.4f}",
        f"acc={val_metrics['acc@1']:.2%}",
        f"acc@5={val_metrics['acc@5']:.2%}",
    )


if __name__ == "__main__":
    main()
