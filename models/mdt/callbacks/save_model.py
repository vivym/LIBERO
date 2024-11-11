import os.path
import warnings
from typing import Any, Dict, List, Optional
import logging

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import STEP_OUTPUT


logger = logging.getLogger(__name__)


class ManuallySaveModelCallback(Callback):
    """ pl.callbacks.ModelCheckpoint is not working when manually backward, we have to save models here. """

    def __init__(self):
        pass


    @rank_zero_only
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        print('[DEBUG][ManuallySaveModelCallback], ready to save model')


