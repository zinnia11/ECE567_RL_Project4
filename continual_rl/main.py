import os
os.environ["OMP_NUM_THREADS"] = "1"

# Optional reproducibility: sbatch job arrays export CONTINUAL_RL_SEED per run.
_seed_env = os.environ.get("CONTINUAL_RL_SEED")
if _seed_env is not None:
    import random
    import numpy as np
    import torch
    _s = int(_seed_env)
    random.seed(_s)
    np.random.seed(_s)
    torch.manual_seed(_s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_s)

import sys
from torch import multiprocessing
from torch.utils.tensorboard.writer import SummaryWriter
from continual_rl.utils.argparse_manager import ArgparseManager


if __name__ == "__main__":
    # Pytorch multiprocessing requires either forkserver or spawn.
    try:
        multiprocessing.set_start_method("spawn")
    except ValueError as e:
        # Windows doesn't support forking, so fall back to spawn instead
        assert "cannot find context" in str(e)
        multiprocessing.set_start_method("spawn")

    experiment, policy = ArgparseManager.parse(sys.argv[1:])

    if experiment is None:
        raise RuntimeError("No experiment started. Most likely there is no new run to start.")

    summary_writer = SummaryWriter(log_dir=experiment.output_dir)
    experiment.try_run(policy, summary_writer=summary_writer)
