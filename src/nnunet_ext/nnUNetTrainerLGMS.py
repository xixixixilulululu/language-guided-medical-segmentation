"""nnU-Net v2 trainer for Phase 0: the default nnUNetTrainer plus a fixed seed and 50-epoch snapshots.

Nothing of nnU-Net's training recipe is changed (network, loss, SGD + poly LR, augmentation including
mirroring, 1000 epochs x 250 iterations, batch / patch size from the plans). Two additions only:

1. Seeding. The seed is read from the environment variable LGMS_SEED (required). python, numpy and
   torch are seeded at construction, and every data-augmentation worker gets its own seed
   LGMS_SEED * 100 + 50 * k + worker index, k = 1 (train) / 2 (val) (nnU-Net passes seeds=None). nnU-Net's
   multi-process loader still returns batches in non-deterministic order, so a run is seeded but not
   bit-reproducible.
2. Snapshots. After every 50th epoch the network weights are written to
   checkpoint_ep<epoch>.pth (weights + the keys nnUNetPredictor needs, no optimizer state), so that
   src/nnunet_select.py can choose the epoch with the best Dice on the full validation set. nnU-Net's
   own checkpoint_best.pth (EMA pseudo-Dice on random validation patches) is NOT used for selection.

The trainer is found by nnU-Net through the environment variable nnUNet_extTrainer=<repo>/src/nnunet_ext.
"""
import os
import random

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from torch._dynamo import OptimizedModule

import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as base_module
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

SNAPSHOT_EVERY = 50


def _seed_from_env():
    if "LGMS_SEED" not in os.environ:
        raise RuntimeError("nnUNetTrainerLGMS needs the environment variable LGMS_SEED")
    return int(os.environ["LGMS_SEED"])


class nnUNetTrainerLGMS(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        self.lgms_seed = _seed_from_env()
        # Seed before the parent constructor so that everything it creates is seeded as well
        random.seed(self.lgms_seed)
        np.random.seed(self.lgms_seed)
        torch.manual_seed(self.lgms_seed)
        torch.cuda.manual_seed_all(self.lgms_seed)
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.print_to_log_file(f"nnUNetTrainerLGMS: seed {self.lgms_seed}, snapshots every {SNAPSHOT_EVERY} epochs")

    def get_dataloaders(self):
        """Parent implementation, with explicit seeds for the augmentation workers."""
        original = base_module.NonDetMultiThreadedAugmenter

        def seeded(*args, num_processes, seeds=None, **kwargs):
            # called twice (train augmenter, then val augmenter); each call gets its own block of 50 seeds
            seeded.calls += 1
            base = self.lgms_seed * 100 + seeded.calls * 50
            return original(*args, num_processes=num_processes,
                            seeds=[base + i for i in range(num_processes)], **kwargs)

        seeded.calls = 0
        base_module.NonDetMultiThreadedAugmenter = seeded
        try:
            return super().get_dataloaders()
        finally:
            base_module.NonDetMultiThreadedAugmenter = original

    def on_epoch_end(self):
        super().on_epoch_end()  # increments self.current_epoch -> number of completed epochs
        done = self.current_epoch
        if done % SNAPSHOT_EVERY == 0 and self.local_rank == 0 and not self.disable_checkpointing:
            mod = self.network.module if self.is_ddp else self.network
            if isinstance(mod, OptimizedModule):
                mod = mod._orig_mod
            torch.save({
                "network_weights": mod.state_dict(),
                "init_args": self.my_init_kwargs,
                "trainer_name": self.__class__.__name__,
                "inference_allowed_mirroring_axes": self.inference_allowed_mirroring_axes,
                "current_epoch": done,
            }, join(self.output_folder, f"checkpoint_ep{done:04d}.pth"))
            self.print_to_log_file(f"snapshot written: checkpoint_ep{done:04d}.pth")


class nnUNetTrainerLGMS_quick(nnUNetTrainerLGMS):
    """Quick check only: 3 full-size epochs (250 iterations each) to test the pipeline and time an epoch."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 3

    def on_epoch_end(self):
        nnUNetTrainer.on_epoch_end(self)
        # snapshot after every epoch so that the selection script can be tested
        mod = self.network._orig_mod if isinstance(self.network, OptimizedModule) else self.network
        torch.save({"network_weights": mod.state_dict(), "init_args": self.my_init_kwargs,
                    "trainer_name": self.__class__.__name__,
                    "inference_allowed_mirroring_axes": self.inference_allowed_mirroring_axes,
                    "current_epoch": self.current_epoch},
                   join(self.output_folder, f"checkpoint_ep{self.current_epoch:04d}.pth"))
