import gc
import logging
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

from ....base import ConfigService
from .config import DatasetConfig
from .dataset import DataLoader, DistributedDataLoader, prepare_dataset

logger = logging.getLogger(__name__)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Training:

    def __init__(
        self,
        config_service: ConfigService,
        dataset_config: DatasetConfig,
        input_dir: Path,
        output_dir: Path,
        model_name_or_path: str,
        local_rank: int = -1,
    ):
        self.config_service = config_service
        self.dataset_config = dataset_config
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.model_name_or_path = model_name_or_path
        self.local_rank = local_rank
        self.batch_size = config_service.get("seq2seq.training.batch_size", 64)
        self.learning_rate = config_service.get("seq2seq.training.learning_rate", 3e-5)
        self.num_epochs = config_service.get("seq2seq.training.num_epochs", 100)
        self.early_stopping = config_service.get("seq2seq.training.early_stopping", 15)
        self.warmup_proportion = config_service.get("seq2seq.training.warmup_proportion", 0.1)
        self.gradient_accumulation_steps = config_service.get("seq2seq.training.gradient_accumulation_steps", 1)
        self.max_grad_norm = config_service.get("seq2seq.training.max_grad_norm", 1.0)
        self.weight_decay = config_service.get("seq2seq.training.weight_decay", 1e-5)

        self.n_gpus = torch.cuda.device_count()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _setup_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        if self.dataset_config.special_tokens:
            self.tokenizer.add_tokens(self.dataset_config.special_tokens)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name_or_path)
        self.model.resize_token_embeddings(len(self.tokenizer))

        if self.n_gpus > 1:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model).cuda()
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )
        else:
            self.model = self.model.to(self.device)

    def _setup_optimizer(self, num_training_steps: int):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
                "lr": self.learning_rate,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": self.learning_rate,
            },
        ]

        warmup_steps = int(num_training_steps * self.warmup_proportion)
        self.optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            eps=1e-8,
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

    def train(self):
        vocab_path = self.input_dir / 'vocab.json'
        train_path = self.input_dir / 'train.pt'
        val_path = self.input_dir / 'val.pt'

        if self.n_gpus > 1:
            train_dataset, train_vocab = prepare_dataset(vocab_path, train_path)
            train_sampler = DistributedSampler(train_dataset)
            train_loader = DistributedDataLoader(
                train_dataset, train_vocab, self.batch_size // self.n_gpus, train_sampler
            )
        else:
            train_loader = DataLoader(vocab_path, train_path, self.batch_size, training=True)

        val_loader = DataLoader(vocab_path, val_path, self.batch_size, training=False)

        self._setup_model()
        num_training_steps = len(train_loader) // self.gradient_accumulation_steps * self.num_epochs
        self._setup_optimizer(num_training_steps)

        best_acc = 0.0
        epochs_not_improving = 0
        global_step = 0

        for epoch in range(self.num_epochs):
            if self.n_gpus > 1:
                train_loader.sampler.set_epoch(epoch)

            self.model.train()
            epoch_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")
            for step, batch in enumerate(pbar):
                batch = tuple(t.to(self.device) if t is not None else None for t in batch)
                source_ids, source_mask, _, target_ids, _ = batch

                if target_ids is None:
                    continue

                pad_token_id = self.tokenizer.pad_token_id
                y_ids = target_ids[:, :-1].contiguous()
                labels = target_ids[:, 1:].clone()
                labels[labels == pad_token_id] = -100

                outputs = self.model(
                    input_ids=source_ids,
                    attention_mask=source_mask,
                    decoder_input_ids=y_ids,
                    labels=labels,
                )
                loss = outputs.loss
                if self.n_gpus > 1:
                    loss = loss.sum()

                loss.backward()
                epoch_loss = loss.item()
                pbar.set_postfix({"loss": epoch_loss})

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()
                    global_step += 1

            val_acc = self._validate(val_loader)
            logger.info(f"Epoch {epoch + 1} - Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                epochs_not_improving = 0
                self._save_checkpoint("checkpoint-best")
            else:
                epochs_not_improving += 1

            if epochs_not_improving >= self.early_stopping:
                logger.info(f"Early stopping after {epochs_not_improving} epochs without improvement")
                break

            if "cuda" in self.device:
                torch.cuda.empty_cache()
            else:
                gc.collect()

        return best_acc

    def _validate(self, val_loader) -> float:
        self.model.eval()
        correct = 0
        total = 0

        max_length = self.config_service.get("seq2seq.max_length", 512)

        with torch.no_grad():
            for batch in val_loader:
                source_ids, source_mask, _, target_ids, _ = [
                    x.to(self.device) if x is not None else None for x in batch
                ]
                if target_ids is None:
                    continue

                outputs = self.model.generate(input_ids=source_ids, max_length=max_length)

                for pred, target in zip(outputs, target_ids):
                    pred_text = self.tokenizer.decode(pred, skip_special_tokens=True)
                    target_text = self.tokenizer.decode(target, skip_special_tokens=True)
                    if pred_text.strip().lower() == target_text.strip().lower():
                        correct += 1
                    total += 1

        return correct / total if total > 0 else 0.0

    def _save_checkpoint(self, name: str):
        output_path = self.output_dir / name
        output_path.mkdir(parents=True, exist_ok=True)

        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        torch.save(self.optimizer.state_dict(), output_path / "optimizer.pt")
        torch.save(self.scheduler.state_dict(), output_path / "scheduler.pt")

        logger.info(f"Saved checkpoint to {output_path}")
