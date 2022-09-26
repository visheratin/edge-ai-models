import pytorch_lightning as pl
import torch
from transformers import AutoModelForSeq2SeqLM


class Model(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "google/t5-efficient-tiny",
        lr: float = 2e-5,
    ) -> None:
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.learning_rate = lr

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        source_ids, attn_masks, target_ids = batch
        outputs = self.model.forward(
            input_ids=source_ids, attention_mask=attn_masks, labels=target_ids
        )
        self.log("train/loss", outputs.loss, prog_bar=True, sync_dist=True)
        return {"loss": outputs.loss}

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        source_ids, attn_masks, target_ids = batch
        outputs = self.model.forward(
            input_ids=source_ids, attention_mask=attn_masks, labels=target_ids
        )
        self.log("validation/loss", outputs.loss, prog_bar=True, sync_dist=True)
        return {"loss": outputs.loss}
