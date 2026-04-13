"""
BERT-based causal triple extractor for NTSB aviation accident narratives.

Uses DistilBERT with a 5-class token classification head to identify cause and
effect spans via BIO (Begin-Inside-Outside) tagging.  Replaces the sequence
classifier so that all three models (Rule-based, BERT, LLM) are compared on
identical causal-triple-extraction metrics.

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
        DistilBertForTokenClassification,
        DistilBertTokenizerFast,
        get_linear_schedule_with_warmup,
    )
    from torch.optim import AdamW
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.rule_based import CAUSAL_FORWARD, CAUSAL_BACKWARD, _split_sentences

# ---------------------------------------------------------------------------
# BIO label constants
# ---------------------------------------------------------------------------
O, B_CAUSE, I_CAUSE, B_EFFECT, I_EFFECT = 0, 1, 2, 3, 4
LABEL_NAMES = ['O', 'B-CAUSE', 'I-CAUSE', 'B-EFFECT', 'I-EFFECT']
NUM_LABELS = 5

# Pattern list ordered longest-first for greedy matching
_ALL_PATTERNS = sorted(CAUSAL_FORWARD + CAUSAL_BACKWARD, key=len, reverse=True)
_PATTERN_RE = {
    p: re.compile(r'\b' + re.escape(p) + r'\b', re.IGNORECASE)
    for p in _ALL_PATTERNS
}


# ---------------------------------------------------------------------------
# Span alignment helpers
# ---------------------------------------------------------------------------

def _find_char_span(text: str, query: str) -> Tuple[int, int]:
    """Return (start, end) char positions of query in text, case-insensitive.

    Falls back to partial match on first 5 words if exact not found.
    Returns (-1, -1) on failure.
    """
    idx = text.lower().find(query.lower().strip())
    if idx >= 0:
        return idx, idx + len(query.strip())
    # partial: first 5 words
    words = query.strip().split()[:5]
    if not words:
        return -1, -1
    partial = ' '.join(words)
    idx = text.lower().find(partial.lower())
    if idx >= 0:
        return idx, idx + len(partial)
    return -1, -1


def _align_spans_to_bio(
    sentence: str,
    cause: str,
    effect: str,
    offset_mapping: List[Tuple[int, int]],
) -> List[int]:
    """
    Map cause/effect substrings to BIO token labels using character offsets.

    Returns a list of ints (one per token).  Special tokens ([CLS], [SEP],
    padding) at positions where offset == (0,0) get label -100 so they are
    ignored by CrossEntropyLoss.
    """
    labels = [O] * len(offset_mapping)

    cause_start, cause_end = _find_char_span(sentence, cause)
    effect_start, effect_end = _find_char_span(sentence, effect)

    first_cause = True
    first_effect = True

    for i, (tok_s, tok_e) in enumerate(offset_mapping):
        if tok_s == tok_e:           # special / padding token
            labels[i] = -100
            continue

        in_cause  = cause_start  >= 0 and tok_s < cause_end  and tok_e > cause_start
        in_effect = effect_start >= 0 and tok_s < effect_end and tok_e > effect_start

        if in_cause:
            labels[i] = B_CAUSE if first_cause else I_CAUSE
            first_cause = False
        elif in_effect:
            labels[i] = B_EFFECT if first_effect else I_EFFECT
            first_effect = False

    return labels


def _tokens_to_text(token_strings: List[str]) -> str:
    """Reconstruct text from WordPiece subword tokens."""
    text = ''
    for tok in token_strings:
        if tok.startswith('##'):
            text += tok[2:]
        else:
            if text:
                text += ' '
            text += tok
    return text.strip()


def _decode_bio(
    token_strings: List[str],
    bio_preds: List[int],
) -> Tuple[Optional[str], Optional[str]]:
    """Decode BIO predictions to (cause_text, effect_text) strings, or None."""
    cause_toks: List[str] = []
    effect_toks: List[str] = []
    in_cause = False
    in_effect = False

    for tok, label in zip(token_strings, bio_preds):
        if label == B_CAUSE:
            cause_toks = [tok]
            in_cause = True
            in_effect = False
        elif label == I_CAUSE and in_cause:
            cause_toks.append(tok)
        elif label == B_EFFECT:
            effect_toks = [tok]
            in_effect = True
            in_cause = False
        elif label == I_EFFECT and in_effect:
            effect_toks.append(tok)
        else:
            in_cause = False
            in_effect = False

    cause  = _tokens_to_text(cause_toks)  if cause_toks  else None
    effect = _tokens_to_text(effect_toks) if effect_toks else None
    return cause, effect


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


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class CausalBIODataset:
    """
    Dataset of (input_ids, attention_mask, labels) for BIO token classification.
    Each example is one sentence.
    """

    def __init__(self, examples: List[dict]):
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for CausalBIODataset")
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
# BERT Causal Extractor
# ---------------------------------------------------------------------------

class BERTCausalExtractor:
    """
    DistilBERT fine-tuned for sentence-level BIO causal span extraction.

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

    MODEL_SAVE_NAME = 'bert_extractor'

    def __init__(
        self,
        model_name: str = 'distilbert-base-uncased',
        max_length: int = 128,
        use_amp: bool = True,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("torch and transformers are required for BERTCausalExtractor.")

        self.model_name = model_name
        self.max_length = max_length

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

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        self.model = DistilBertForTokenClassification.from_pretrained(
            model_name,
            num_labels=NUM_LABELS,
        )
        self.model.to(self.device)

        print(f"[BERTExtractor] Device: {self.device}  AMP: {self.use_amp}  dtype: {self.amp_dtype}")

    def _autocast(self):
        if self.use_amp and self.is_cuda:
            return torch.autocast(device_type='cuda', dtype=self.amp_dtype)
        return nullcontext()

    def _to_device(self, batch: dict) -> dict:
        return {k: v.to(self.device, non_blocking=self.is_cuda) for k, v in batch.items()}

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _make_example(self, sentence: str, cause: str, effect: str) -> Optional[dict]:
        """Tokenize one sentence and align BIO labels. Returns None if no labels found."""
        enc = self.tokenizer(
            sentence,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_offsets_mapping=True,
        )
        bio = _align_spans_to_bio(sentence, cause, effect, enc['offset_mapping'])
        # Require at least one non-O, non-ignored label
        if not any(l in (B_CAUSE, B_EFFECT) for l in bio):
            return None
        return {
            'input_ids':      enc['input_ids'],
            'attention_mask': enc['attention_mask'],
            'labels':         bio,
        }

    def _make_negative_example(self, sentence: str) -> dict:
        """Create a negative example (all-O labels) for a sentence."""
        enc = self.tokenizer(
            sentence,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_offsets_mapping=True,
        )
        bio = [
            -100 if (s == e) else O
            for s, e in enc['offset_mapping']
        ]
        return {
            'input_ids':      enc['input_ids'],
            'attention_mask': enc['attention_mask'],
            'labels':         bio,
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
    ) -> Tuple['CausalBIODataset', 'CausalBIODataset']:
        """
        Build training and validation CausalBIODatasets from rule-based triples.

        Positive examples: sentences from training triples with BIO labels.
        Negative examples: sentences from training narratives that contain no
            causal pattern, sampled at neg_ratio × positive count.

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
            if ex is not None:
                positives.append(ex)

        print(f"[BERTExtractor] Positive BIO examples: {len(positives)}")

        # ----- Negatives -----------------------------------------------
        # Collect sentences from training narratives that have no causal hit
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
        print(f"[BERTExtractor] Negative BIO examples: {len(negatives)}")

        # ----- Combine and split ----------------------------------------
        all_examples = positives + negatives
        rng.shuffle(all_examples)

        n_val = max(1, int(len(all_examples) * val_fraction))
        val_examples   = all_examples[:n_val]
        train_examples = all_examples[n_val:]

        print(f"[BERTExtractor] Train: {len(train_examples)}  Val: {len(val_examples)}")
        return CausalBIODataset(train_examples), CausalBIODataset(val_examples)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_ds: 'CausalBIODataset',
        val_ds: 'CausalBIODataset',
        epochs: int = 5,
        batch_size: int = 16,
        lr: float = 2e-5,
        save_path: Optional[str] = None,
        patience: int = 3,
    ) -> dict:
        """
        Fine-tune DistilBERT for BIO token classification.
        Returns training history dict including bias-variance analysis.
        """
        num_workers = 0  # safest default for Windows
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

        # Class weights: up-weight CAUSE/EFFECT tokens vs O
        # Rough heuristic: O tokens dominate (~90%), so weight them lower
        cw = torch.ones(NUM_LABELS, device=self.device, dtype=torch.float32)
        cw[O] = 0.2        # O is the overwhelming majority class
        loss_fn = torch.nn.CrossEntropyLoss(weight=cw, ignore_index=-100)

        scaler = torch.cuda.amp.GradScaler(
            enabled=self.use_amp and self.amp_dtype == torch.float16
        ) if self.is_cuda else None

        best_val_f1     = 0.0
        no_improve      = 0
        best_weights    = None
        epoch_losses    = []
        epoch_val_f1s   = []
        epoch_train_f1s = []
        bias_variance_logs = []

        for epoch in range(1, epochs + 1):
            # --- Train ---
            self.model.train()
            total_loss = 0.0
            train_tp = train_fp = train_fn = 0
            for batch in train_loader:
                batch = self._to_device(batch)
                optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    out  = self.model(**{k: v for k, v in batch.items() if k != 'labels'})
                    # Ensure logits are float32 for loss computation
                    logits = out.logits.float() if out.logits.dtype != torch.float32 else out.logits
                    loss = loss_fn(logits.view(-1, NUM_LABELS), batch['labels'].view(-1))
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

                # Track training metrics
                preds  = out.logits.argmax(dim=-1).view(-1).cpu().detach().tolist()
                golds  = batch['labels'].view(-1).cpu().tolist()
                for p, g in zip(preds, golds):
                    if g == -100:
                        continue
                    pred_pos = p in (B_CAUSE, I_CAUSE, B_EFFECT, I_EFFECT)
                    gold_pos = g in (B_CAUSE, I_CAUSE, B_EFFECT, I_EFFECT)
                    if pred_pos and gold_pos:
                        train_tp += 1
                    elif pred_pos and not gold_pos:
                        train_fp += 1
                    elif not pred_pos and gold_pos:
                        train_fn += 1

            avg_train_loss = total_loss / max(1, len(train_loader))
            train_prec = train_tp / max(1, train_tp + train_fp)
            train_rec = train_tp / max(1, train_tp + train_fn)
            train_f1 = 2 * train_prec * train_rec / max(1e-9, train_prec + train_rec)

            # --- Validate: span-level F1 on cause+effect tokens ---
            self.model.eval()
            tp = fp = fn = 0
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = self._to_device(batch)
                    with self._autocast():
                        out = self.model(**{k: v for k, v in batch.items() if k != 'labels'})
                    # Ensure logits are float32 for loss computation
                    logits = out.logits.float() if out.logits.dtype != torch.float32 else out.logits
                    val_batch_loss = loss_fn(logits.view(-1, NUM_LABELS), batch['labels'].view(-1))
                    val_loss += val_batch_loss.item()

                    preds  = out.logits.argmax(dim=-1).view(-1).cpu().tolist()
                    golds  = batch['labels'].view(-1).cpu().tolist()
                    for p, g in zip(preds, golds):
                        if g == -100:
                            continue
                        pred_pos = p in (B_CAUSE, I_CAUSE, B_EFFECT, I_EFFECT)
                        gold_pos = g in (B_CAUSE, I_CAUSE, B_EFFECT, I_EFFECT)
                        if pred_pos and gold_pos:
                            tp += 1
                        elif pred_pos and not gold_pos:
                            fp += 1
                        elif not pred_pos and gold_pos:
                            fn += 1

            avg_val_loss = val_loss / max(1, len(val_loader))
            prec = tp / max(1, tp + fp)
            rec  = tp / max(1, tp + fn)
            f1   = 2 * prec * rec / max(1e-9, prec + rec)
            epoch_losses.append(round(avg_val_loss, 4))
            epoch_val_f1s.append(round(f1, 4))
            epoch_train_f1s.append(round(train_f1, 4))

            # Log bias-variance analysis
            bv_log = self._log_bias_variance(avg_train_loss, avg_val_loss, train_f1, f1, epoch)
            bias_variance_logs.append(bv_log)

            print(f"[BERTExtractor] Epoch {epoch}/{epochs} — "
                  f"train loss: {avg_train_loss:.4f} val loss: {avg_val_loss:.4f}  "
                  f"train F1: {train_f1:.4f} val F1: {f1:.4f}  "
                  f"(regime: {bv_log['regime']})")

            if f1 > best_val_f1:
                best_val_f1 = f1
                no_improve  = 0
                best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                if save_path:
                    self.save(save_path)
            else:
                no_improve += 1
                if patience > 0 and no_improve >= patience:
                    print(f"[BERTExtractor] Early stopping at epoch {epoch}")
                    break

        if best_weights is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_weights.items()})
            print(f"[BERTExtractor] Restored best weights (val F1={best_val_f1:.4f})")

        return {
            'train_loss':   epoch_losses,
            'train_f1':     epoch_train_f1s,
            'val_f1':       epoch_val_f1s,
            'best_val_f1':  best_val_f1,
            'bias_variance_logs': bias_variance_logs,
        }

    def _log_bias_variance(
        self,
        train_loss: float,
        val_loss: float,
        train_f1: float,
        val_f1: float,
        epoch: int,
    ) -> dict:
        """Analyze bias-variance tradeoff for an epoch."""
        loss_gap = val_loss - train_loss
        f1_gap = val_f1 - train_f1

        # Classify regime
        if loss_gap > 0.1 and f1_gap < -0.05:
            regime = 'high_variance'  # Overfitting
        elif loss_gap < -0.1 or f1_gap > 0.05:
            regime = 'high_bias'      # Underfitting
        else:
            regime = 'balanced'

        return {
            'epoch': epoch,
            'train_loss': round(train_loss, 6),
            'val_loss': round(val_loss, 6),
            'loss_gap': round(loss_gap, 6),
            'train_f1': round(train_f1, 4),
            'val_f1': round(val_f1, 4),
            'f1_gap': round(f1_gap, 4),
            'regime': regime,
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
        restrict_ev_ids: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Run BIO extraction on all narratives in df.

        For each narrative:
          1. Split into sentences.
          2. Tokenize each sentence, run BIO prediction.
          3. Decode cause/effect spans from predicted labels.
          4. Infer relation phrase from sentence text.
          5. Return triple dict.

        Returns list of {ev_id, cause, relation, effect, direction, method, sentence}.
        """
        if restrict_ev_ids is not None:
            restrict_set = set(str(e) for e in restrict_ev_ids)
            df = df[df[id_col].astype(str).isin(restrict_set)]

        self.model.eval()
        all_triples: List[dict] = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc='[BERTExtractor] Extracting'):
            ev_id = str(row.get(id_col, ''))
            text  = str(row.get(text_col, ''))
            if not text.strip():
                continue

            sentences = _split_sentences(text)
            if not sentences:
                continue

            # Batch-encode all sentences for this narrative
            enc = self.tokenizer(
                sentences,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_offsets_mapping=True,
                return_tensors=None,
            )

            n = len(sentences)
            # Process in sub-batches
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                sub_ids   = enc['input_ids'][start:end]
                sub_mask  = enc['attention_mask'][start:end]
                sub_sents = sentences[start:end]

                input_ids = torch.tensor(sub_ids,  dtype=torch.long).to(self.device)
                attn_mask = torch.tensor(sub_mask, dtype=torch.long).to(self.device)

                with torch.no_grad():
                    with self._autocast():
                        out = self.model(input_ids=input_ids, attention_mask=attn_mask)
                pred_ids = out.logits.argmax(dim=-1).cpu().tolist()  # (sub_batch, seq_len)

                for sent, preds in zip(sub_sents, pred_ids):
                    # Get actual tokens for this sentence
                    tokens = self.tokenizer.convert_ids_to_tokens(
                        self.tokenizer(
                            sent,
                            truncation=True,
                            max_length=self.max_length,
                        )['input_ids']
                    )
                    # Trim to actual sentence length (no padding)
                    n_tok = len(tokens)
                    preds_trimmed = preds[:n_tok]

                    cause, effect = _decode_bio(tokens, preds_trimmed)
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
                        'method':    'bert',
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
        meta = {'model_name': self.model_name, 'max_length': self.max_length, 'num_labels': NUM_LABELS}
        with open(save_dir / 'extractor_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"[BERTExtractor] Saved to {save_dir}")

    def load(self, path: str):
        load_dir = Path(path)
        self.model = DistilBertForTokenClassification.from_pretrained(str(load_dir))
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(str(load_dir))
        self.model.to(self.device)
        print(f"[BERTExtractor] Loaded from {load_dir}")
