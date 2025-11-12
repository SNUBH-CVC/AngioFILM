import logging
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Union

import torch
import torch.distributed as dist
from torch import inf
from torch.utils.tensorboard import SummaryWriter

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


#################################################################################
#                             Training Clip Gradients                           #
#################################################################################


def get_grad_norm(
    parameters: _tensor_or_tensors, norm_type: float = 2.0
) -> torch.Tensor:
    r"""
    Copy from torch.nn.utils.clip_grad_norm_

    Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.0)
    device = grads[0].device
    if norm_type == inf:
        norms = [g.detach().abs().max().to(device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]),
            norm_type,
        )
    return total_norm


def clip_grad_norm_(
    parameters: _tensor_or_tensors,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    clip_grad=True,
) -> torch.Tensor:
    r"""
    Copy from torch.nn.utils.clip_grad_norm_

    Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.0)
    device = grads[0].device
    if norm_type == inf:
        norms = [g.detach().abs().max().to(device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]),
            norm_type,
        )

    if clip_grad:
        if error_if_nonfinite and torch.logical_or(
            total_norm.isnan(), total_norm.isinf()
        ):
            raise RuntimeError(
                f"The total norm of order {norm_type} for gradients from "
                "`parameters` is non-finite, so it cannot be clipped. To disable "
                "this error and scale the gradients by the non-finite norm anyway, "
                "set `error_if_nonfinite=False`"
            )
        clip_coef = max_norm / (total_norm + 1e-6)
        # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
        # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
        # when the gradients do not reside in CPU memory.
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for g in grads:
            g.detach().mul_(clip_coef_clamped.to(g.device))
        # gradient_cliped = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
    return total_norm


def get_experiment_dir(results_dir: Path, args):
    prev_experiment_indices = [int(path.stem.split("-")[0]) for path in results_dir.glob("*")]
    experiment_index = max(prev_experiment_indices) + 1 if prev_experiment_indices else 0
    model_string_name = args.model.name.replace(
        "/", "-"
    )  # e.g., Latte-XL/2 --> Latte-XL-2 (for naming folders)

    # Create method-specific parameters string
    method_params = f"{args.transport.path_type}-{args.transport.prediction}"
    experiment_dirname = f"{experiment_index:02d}-{model_string_name}-{method_params}-{args.dataset.name}"
    if args.system.use_compile:
        experiment_dirname += "-Compile"  # speedup by torch compile
    if args.system.enable_xformers_memory_efficient_attention:
        experiment_dirname += "-Xformers"
    if args.system.gradient_checkpointing:
        experiment_dirname += "-Gc"
    if args.system.mixed_precision:
        experiment_dirname += "-Amp"
    if args.model.image_size == 512:
        experiment_dirname += "-512"
    experiment_dir = results_dir / experiment_dirname
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


#################################################################################
#                             Training Logger                                   #
#################################################################################


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            # format='[\033[34m%(asctime)s\033[0m] %(message)s',
            format="[%(asctime)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
        logger = logging.getLogger(__name__)

    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def create_tensorboard(tensorboard_dir):
    """
    Create a tensorboard that saves losses.
    """
    if dist.get_rank() == 0:  # real tensorboard
        # tensorboard
        writer = SummaryWriter(tensorboard_dir)

    return writer


def write_tensorboard(writer, *args):
    """
    write the loss information to a tensorboard file.
    Only for pytorch DDP mode.
    """
    if dist.get_rank() == 0:  # real tensorboard
        writer.add_scalar(args[0], args[1], args[2])


#################################################################################
#                      EMA Update/ DDP Training Utils                           #
#################################################################################


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def setup_distributed(backend="nccl"):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend=backend)
    return rank, local_rank, world_size


def load_checkpoint(checkpoint_path, model, ema, optimizer, lr_scheduler, scaler, device):
    """
    Load checkpoint and return the training state.
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model states
    model.module.load_state_dict(checkpoint["model"])
    ema.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["scheduler"])

    # Load scaler state if available and using mixed precision
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    # Return training state
    train_steps = checkpoint["train_steps"]
    epoch = checkpoint["epoch"]

    print(f"Resumed from train_steps={train_steps}, epoch={epoch}")
    return train_steps, epoch


def get_resume_experiment_dir(resume_checkpoint_path):
    """
    Get the experiment directory when resuming training.
    Extract the experiment directory from the checkpoint path.
    """
    # checkpoint path structure: work_dirs/experiment_name/checkpoints/xxxxx.pt
    checkpoint_path = Path(resume_checkpoint_path)
    if checkpoint_path.name.endswith('.pt') and checkpoint_path.parent.name == 'checkpoints':
        experiment_dir = checkpoint_path.parent.parent
        if experiment_dir.exists():
            return experiment_dir

    # Fallback: return None to create new experiment directory
    return None
