"""
T5-based causal triple extractor for NTSB aviation accident narratives.

Uses T5-base for seq2seq extraction: sentence → "cause: X | effect: Y"
This treats extraction as a natural language generation task, which leverages
T5's text-to-text framework and works well for structured outputs.

Training data: rule-based triples from training narratives (pseudo-labels).
"""
import json
import re
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        T5ForConditionalGeneration,
        T5TokenizerFast,
        get_linear_schedule_with_warmup,
    )
    from torch.optim import AdamW
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.rule_based import CAUSAL_FORWARD, CAUSAL_BACKWARD, _split_sentences

# Pattern list ordered longest-first for greedy matching
_ALL_PATTERNS = sorted(CAUSAL_FORWARD + CAUSAL_BACKWARD, key=len, reverse=True)
_PATTERN_RE = {
    p: re.compile(r'\b' + re.escape(p) + r'\b', re.IGNORECASE)
    for p in _ALL_PATTERNS
}


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _infer_relation(sentence: str, cause: str, effect: str) -> Tuple[str, str]:
    """
    Find the first causal connective phrase in sentence.
    Returns (relation_phrase, direction).
    Direction is 'forward' for CAUSE→EFFECT patterns, 'backward' for EFFECT←CAUSE.
    """
    for pat in _ALL_PATTERNS:
        if _PATTERN_RE[pat].search(sentence):
            direction = 'forward' if pat in CAUSAL_FORWARD else 'backward'
            return pat, direction
    return 'caused', 'forward'


def _parse_t5_output(output_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse T5 output format: "cause: [text] | effect: [text]"
    Returns (cause, effect) or (None, None) if parsing fails.
    """
    output_text = output_text.strip()

    # Try to split on "|"
    if '|' not in output_text:
        return None, None

    parts = output_text.split('|')
    if len(parts) != 2:
        return None, None

    cause_part = parts[0].strip()
    effect_part = parts[1].strip()

    # Extract text after "cause:" and "effect:"
    cause_match = re.search(r'cause:\s*(.+)', cause_part, re.IGNORECASE)
    effect_match = re.search(r'effect:\s*(.+)', effect_part, re.IGNORECASE)

    if not cause_match or not effect_match:
        return None, None

    cause = cause_match.group(1).strip()
    effect = effect_match.group(1).strip()

    if not cause or not effect:
        return None, None

    return cause, effect


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class T5ExtractionDataset(Dataset):
    """Dataset of (input_ids, attention_mask, decoder_input_ids, labels) for T5."""

    def __init__(self, examples: List[dict]):
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for T5ExtractionDataset")
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        return {
            'input_ids':      torch.tensor(ex['input_ids'],      dtype=torch.long),
            'attention_mask': torch.tensor(ex['attention_mask'], dtype=torch.long),
            'labels':         torch.tensor(ex['labels'],         dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# T5 Causal Extractor
# ---------------------------------------------------------------------------

class T5CausalExtractor:
    """
    T5 fine-tuned for seq2seq causal triple extraction.

    Training workflow
    -----------------
    1. Call prepare_data() with rule_triples (training ev_ids only).
    2. Call train() with the returned train/val datasets.
    3. Call extract() on any DataFrame to get causal triples.

    Inference workflow (pre-trained model)
    --------------------------------------
    1. Call load(path) to restore weights.
    2. Call extract() on any DataFrame.
    """

    MODEL_SAVE_NAME = 't5_extractor'

    def __init__(
        self,
        model_name: str = 't5-base',
        max_length: int = 128,
        max_target_length: int = 64,
        use_amp: bool = True,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("torch and transformers are required for T5CausalExtractor.")

        self.model_name = model_name
        self.max_length = max_length
        self.max_target_length = max_target_length

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_cuda = self.device.type == 'cuda'

        if self.is_cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        # AMP: prefer bfloat16 on supported GPUs
        self.use_amp = bool(use_amp and self.is_cuda)
        self.amp_dtype = None
        if self.use_amp:
            try:
                self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            except Exception:
                self.amp_dtype = torch.float16

        self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)

        print(f"[T5Extractor] Device: {self.device}  AMP: {self.use_amp}  dtype: {self.amp_dtype}")

    def _autocast(self):
        if self.use_amp and self.is_cuda:
            return torch.autocast(device_type='cuda', dtype=self.amp_dtype)
        return nullcontext()

    def _to_device(self, batch: dict) -> dict:
        return {k: v.to(self.device, non_blocking=self.is_cuda) for k, v in batch.items()}

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _make_example(self, sentence: str, cause: str, effect: str) -> dict:
        """Create input-output example for T5."""
        # Input: "sentence: [text]"
        input_text = f"sentence: {sentence}"
        # Target: "cause: [cause] | effect: [effect]"
        target_text = f"cause: {cause} | effect: {effect}"

        enc_in = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
        )
        enc_out = self.tokenizer(
            target_text,
            truncation=True,
            max_length=self.max_target_length,
            padding='max_length',
        )

        return {
            'input_ids':      enc_in['input_ids'],
            'attention_mask': enc_in['attention_mask'],
            'labels':         enc_out['input_ids'],
        }

    def _make_negative_example(self, sentence: str) -> dict:
        """Create a negative example (no cause/effect)."""
        input_text = f"sentence: {sentence}"
        # Target: "no causal relation"
        target_text = "no causal relation"

        enc_in = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
        )
        enc_out = self.tokenizer(
            target_text,
            truncation=True,
            max_length=self.max_target_length,
            padding='max_length',
        )

        return {
            'input_ids':      enc_in['input_ids'],
            'attention_mask': enc_in['attention_mask'],
            'labels':         enc_out['input_ids'],
        }

    def prepare_data(
        self,
        df: pd.DataFrame,
        rule_triples: list,
        train_ev_ids: List[str],
        text_col: str = 'narr_clean',
        id_col: str = 'ev_id',
        val_fraction: float = 0.15,
        neg_ratio: float = 2.0,
        rng_seed: int = 42,
    ) -> Tuple[T5ExtractionDataset, T5ExtractionDataset]:
        """
        Build training and validation datasets from rule-based triples.

        Positive examples: sentences from training triples with cause/effect targets.
        Negative examples: sentences from training narratives with no causal pattern.

        Returns (train_ds, val_ds).
        """
        train_ev_set = set(str(e) for e in train_ev_ids)

        # ----- Positives ------------------------------------------------
        positives: List[dict] = []
        seen_pos: set = set()
        for triple in rule_triples:
            if str(triple.get('ev_id', '')) not in train_ev_set:
                continue
            sent  = triple.get('sentence', '').strip()
            cause = triple.get('cause',    '').strip()
            eff   = triple.get('effect',   '').strip()
            if not sent or not cause or not eff:
                continue
            key = (sent[:80], cause[:40], eff[:40])
            if key in seen_pos:
                continue
            seen_pos.add(key)
            ex = self._make_example(sent, cause, eff)
            positives.append(ex)

        print(f"[T5Extractor] Positive examples: {len(positives)}")

        # ----- Negatives -----------------------------------------------
        train_df = df[df[id_col].astype(str).isin(train_ev_set)].dropna(subset=[text_col])
        positive_sents = {k[0] for k in seen_pos}
        _any_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(p) for p in _ALL_PATTERNS) + r')\b',
            re.IGNORECASE,
        )
        rng = np.random.default_rng(rng_seed)
        negative_pool: List[str] = []
        for text in train_df[text_col].astype(str).tolist():
            for sent in _split_sentences(text):
                if sent[:80] not in positive_sents and not _any_pattern.search(sent):
                    negative_pool.append(sent)

        n_neg = min(int(len(positives) * neg_ratio), len(negative_pool))
        chosen_neg = rng.choice(negative_pool, size=n_neg, replace=False).tolist()
        negatives  = [self._make_negative_example(s) for s in chosen_neg]
        print(f"[T5Extractor] Negative examples: {len(negatives)}")

        # ----- Combine and split ----------------------------------------
        all_examples = positives + negatives
        rng.shuffle(all_examples)

        n_val = max(1, int(len(all_examples) * val_fraction))
        val_examples   = all_examples[:n_val]
        train_examples = all_examples[n_val:]

        print(f"[T5Extractor] Train: {len(train_examples)}  Val: {len(val_examples)}")
        return T5ExtractionDataset(train_examples), T5ExtractionDataset(val_examples)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_ds: T5ExtractionDataset,
        val_ds: T5ExtractionDataset,
        epochs: int = 5,
        batch_size: int = 16,
        lr: float = 1e-4,
        save_path: Optional[str] = None,
        patience: int = 3,
    ) -> dict:
        """
        Fine-tune T5 for causal extraction.
        Returns training history dict.
        """
        num_workers = 0
        loader_kw = {'num_workers': num_workers, 'pin_memory': self.is_cuda}

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **loader_kw)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **loader_kw)

        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        total_steps  = len(train_loader) * epochs
        warmup_steps = max(1, int(0.1 * total_steps))
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        scaler = torch.cuda.amp.GradScaler(
            enabled=self.use_amp and self.amp_dtype == torch.float16
        ) if self.is_cuda else None

        best_val_loss   = float('inf')
        no_improve      = 0
        best_weights    = None
        epoch_losses    = []
        epoch_val_losses = []

        for epoch in range(1, epochs + 1):
            # --- Train ---
            self.model.train()
            total_loss = 0.0
            for batch in train_loader:
                batch = self._to_device(batch)
                optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels'],
                    )
                    loss = outputs.loss
                if scaler and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / max(1, len(train_loader))

            # --- Validate ---
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = self._to_device(batch)
                    with self._autocast():
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['labels'],
                        )
                        val_loss += outputs.loss.item()

            avg_val_loss = val_loss / max(1, len(val_loader))
            epoch_losses.append(round(avg_train_loss, 4))
            epoch_val_losses.append(round(avg_val_loss, 4))

            print(f"[T5Extractor] Epoch {epoch}/{epochs} — "
                  f"train loss: {avg_train_loss:.4f}  val loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improve  = 0
                best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                if save_path:
                    self.save(save_path)
            else:
                no_improve += 1
                if patience > 0 and no_improve >= patience:
                    print(f"[T5Extractor] Early stopping at epoch {epoch}")
                    break

        if best_weights is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_weights.items()})
            print(f"[T5Extractor] Restored best weights (val loss={best_val_loss:.4f})")

        return {
            'train_loss':   epoch_losses,
            'val_loss':     epoch_val_losses,
            'best_val_loss': best_val_loss,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def extract(
        self,
        df: pd.DataFrame,
        text_col: str = 'narr_clean',
        id_col: str = 'ev_id',
        batch_size: int = 32,
        max_new_tokens: int = 64,
        restrict_ev_ids: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Run T5 extraction on all narratives in df.

        For each narrative:
          1. Split into sentences.
          2. Run T5 generation to produce "cause: X | effect: Y"
          3. Parse output.
          4. Infer relation phrase from sentence text.
          5. Return triple dict.

        Returns list of {ev_id, cause, relation, effect, direction, method, sentence}.
        """
        if restrict_ev_ids is not None:
            restrict_set = set(str(e) for e in restrict_ev_ids)
            df = df[df[id_col].astype(str).isin(restrict_set)]

        self.model.eval()
        all_triples: List[dict] = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc='[T5Extractor] Extracting'):
            ev_id = str(row.get(id_col, ''))
            text  = str(row.get(text_col, ''))
            if not text.strip():
                continue

            sentences = _split_sentences(text)
            if not sentences:
                continue

            # Process sentences in batches
            for start in range(0, len(sentences), batch_size):
                end = min(start + batch_size, len(sentences))
                batch_sents = sentences[start:end]
                input_texts = [f"sentence: {s}" for s in batch_sents]

                # Encode inputs
                enc = self.tokenizer(
                    input_texts,
                    truncation=True,
                    max_length=self.max_length,
                    padding='max_length',
                    return_tensors='pt',
                )

                input_ids = enc['input_ids'].to(self.device)
                attn_mask = enc['attention_mask'].to(self.device)

                with torch.no_grad():
                    with self._autocast():
                        outputs = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attn_mask,
                            max_new_tokens=max_new_tokens,
                            num_beams=1,  # greedy decoding
                            early_stopping=True,
                        )

                # Decode outputs
                generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

                for sent, gen_text in zip(batch_sents, generated_texts):
                    cause, effect = _parse_t5_output(gen_text)
                    if not cause or not effect:
                        continue
                    if len(cause.split()) < 2 or len(effect.split()) < 2:
                        continue  # single-word spans are noise

                    relation, direction = _infer_relation(sent, cause, effect)
                    all_triples.append({
                        'ev_id':     ev_id,
                        'cause':     cause,
                        'relation':  relation,
                        'effect':    effect,
                        'direction': direction,
                        'method':    't5',
                        'sentence':  sent,
                    })

        return all_triples

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(save_dir))
        self.tokenizer.save_pretrained(str(save_dir))
        meta = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'max_target_length': self.max_target_length,
        }
        with open(save_dir / 'extractor_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"[T5Extractor] Saved to {save_dir}")

    def load(self, path: str):
        load_dir = Path(path)
        self.model = T5ForConditionalGeneration.from_pretrained(str(load_dir))
        self.tokenizer = T5TokenizerFast.from_pretrained(str(load_dir))
        self.model.to(self.device)
        print(f"[T5Extractor] Loaded from {load_dir}")
