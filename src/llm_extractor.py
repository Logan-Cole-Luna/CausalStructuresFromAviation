"""
LLM-based causal chain extractor (Model 4).

Uses Mistral-7B-Instruct-v0.3 (or any HuggingFace chat model) to extract
structured cause-effect triples from aviation accident narratives via prompting.

Model choice rationale
----------------------
* mistralai/Mistral-7B-Instruct-v0.3
    - 7 B parameters → ~4 GB VRAM at 4-bit, ~14 GB at fp16
    - Best-in-class instruction following and JSON output at this size
    - Strong zero-shot causal reasoning; no aviation fine-tuning required
    - Fits comfortably on RTX 5070 Ti (16 GB) with 4-bit quantization
    - Falls back to float16 if bitsandbytes is unavailable (~14 GB)
"""

import json
import re
import warnings
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import bitsandbytes  # noqa: F401
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False

try:
    import accelerate  # noqa: F401
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Prompt templates (Mistral chat format)
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are an aviation safety analyst. "
    "Your task is to extract explicit causal relationships from accident narratives. "
    "Return only a valid JSON array — no markdown, no explanation, no extra text."
)

_USER_TMPL = """\
Extract all causal relationships from the following aviation accident narrative.

For each relationship output an object with exactly these keys:
  "cause"    — the factor, condition, or event that caused something
  "relation" — the causal verb/phrase (e.g. "resulted in", "caused", "led to")
  "effect"   — what was caused or resulted

Return a JSON array: [{{"cause": "...", "relation": "...", "effect": "..."}}]
If no causal relationships are present, return: []

Narrative:
{narrative}"""

# Simpler fallback prompt used on retry after a JSON parse failure
_USER_FALLBACK_TMPL = """\
Read this aviation accident text and output ONLY a valid JSON array.
Each item must have exactly three string keys: "cause", "relation", "effect".
Example: [{{"cause": "fuel exhaustion", "relation": "caused", "effect": "loss of engine power"}}]
If none found, output: []

Text: {narrative}

JSON:"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_JUNK_NODES = {
    "the accident", "this accident", "the incident", "this incident",
    "the crash", "an accident", "an incident",
}

# Regex to pull the first JSON array out of model output
_JSON_RE = re.compile(r'\[.*\]', re.DOTALL)

# Default cache location (relative to project root)
DEFAULT_CACHE_PATH = Path("outputs/extractions/llm_response_cache.json")


def _load_cache(path: Path) -> dict:
    """Load {ev_id: raw_response} cache from disk. Returns empty dict if missing."""
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict, path: Path):
    """Atomically write cache to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    tmp.replace(path)


def _parse_triples(raw: str, ev_id: str) -> List[dict]:
    """
    Extract and validate JSON triples from raw LLM output.
    Returns a list of dicts with keys: ev_id, cause, relation, effect, source.
    """
    match = _JSON_RE.search(raw)
    if not match:
        return []

    try:
        items = json.loads(match.group())
    except json.JSONDecodeError:
        return []

    triples = []
    for item in items:
        if not isinstance(item, dict):
            continue
        cause    = str(item.get("cause", "")).strip()
        relation = str(item.get("relation", "caused")).strip()
        effect   = str(item.get("effect", "")).strip()

        if not cause or not effect:
            continue
        if cause.lower() in _JUNK_NODES or effect.lower() in _JUNK_NODES:
            continue

        triples.append({
            "ev_id":    ev_id,
            "cause":    cause,
            "relation": relation,
            "effect":   effect,
            "direction": "forward",
            "source":   "llm",
        })

    return triples


# ---------------------------------------------------------------------------
# Extractor class
# ---------------------------------------------------------------------------

class LLMCausalExtractor:
    """
    Prompt-based causal extractor using a HuggingFace instruction-tuned LLM.

    Parameters
    ----------
    model_name : str
        HuggingFace model id, e.g. "mistralai/Mistral-7B-Instruct-v0.3".
    load_in_4bit : bool
        Use 4-bit NF4 quantization via bitsandbytes (recommended for 16 GB GPUs).
        Falls back to float16 if bitsandbytes is not installed.
    max_new_tokens : int
        Maximum tokens the model may generate per narrative.
    temperature : float
        Sampling temperature. 0.0 → greedy (recommended for extraction).
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        load_in_4bit: bool = True,
        max_new_tokens: int = 300,
        temperature: float = 0.0,
    ):
        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "torch and transformers are required for LLMCausalExtractor. "
                "Install them with: pip install torch transformers"
            )

        self.model_name     = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- quantization config ----
        use_4bit = load_in_4bit and BNB_AVAILABLE and self.device.type == "cuda"
        if load_in_4bit and not BNB_AVAILABLE:
            warnings.warn(
                "[LLM] bitsandbytes not installed — falling back to float16. "
                "Install with: pip install bitsandbytes",
                stacklevel=2,
            )

        torch_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        print(f"[LLM] Loading {model_name}")
        print(f"[LLM] 4-bit quantization: {use_4bit}  |  dtype: {torch_dtype}  |  device: {self.device}")

        bnb_config = None
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Decoder-only models require left-padding for correct batched generation
        self.tokenizer.padding_side = "left"

        load_kwargs: dict = {"dtype": torch_dtype}
        if bnb_config is not None:
            load_kwargs["quantization_config"] = bnb_config
        # device_map="auto" requires accelerate; fall back to manual .to(device)
        if ACCELERATE_AVAILABLE and self.device.type == "cuda":
            load_kwargs["device_map"] = "auto"
        if not ACCELERATE_AVAILABLE and self.device.type == "cuda":
            warnings.warn(
                "[LLM] accelerate not installed — model will be loaded to CPU then moved to CUDA. "
                "Install with: pip install accelerate",
                stacklevel=2,
            )

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if "device_map" not in load_kwargs:
            self.model = self.model.to(self.device)
        self.model.eval()

        if self.device.type == "cuda":
            used = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[LLM] VRAM after load: {used:.1f} / {total:.1f} GB")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_prompt(self, narrative: str, fallback: bool = False) -> str:
        """Format prompt using the model's chat template if available."""
        user_content = (
            _USER_FALLBACK_TMPL if fallback else _USER_TMPL
        ).format(narrative=narrative[:1500])
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": user_content},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return (
                f"[INST] {_SYSTEM}\n\n"
                f"{user_content} [/INST]"
            )

    @torch.inference_mode()
    def _generate_batch(self, prompts: List[str]) -> List[str]:
        """Tokenize and generate responses for a batch of prompts."""
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        # device_map="auto" places model layers on CUDA, but inputs are always
        # created on CPU by the tokenizer — move them to the first CUDA device.
        if self.device.type == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=self.temperature > 0.0,
        )
        if self.temperature > 0.0:
            gen_kwargs["temperature"] = self.temperature

        output_ids = self.model.generate(**inputs, **gen_kwargs)

        # Decode only the newly generated tokens (strip the prompt)
        responses = []
        for i, ids in enumerate(output_ids):
            prompt_len = inputs["input_ids"][i].shape[0]
            new_ids = ids[prompt_len:]
            responses.append(self.tokenizer.decode(new_ids, skip_special_tokens=True))

        return responses

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def extract_batch(
        self,
        df: pd.DataFrame,
        text_col: str = "narr_clean",
        id_col: str = "ev_id",
        sample_n: Optional[int] = 200,
        batch_size: int = 4,
        seed: int = 42,
        max_retries: int = 1,
        cache_path: Optional[Path] = DEFAULT_CACHE_PATH,
    ) -> List[dict]:
        """
        Run causal extraction on a sample of the dataset.

        Responses are cached to ``cache_path`` (keyed by ev_id) so that
        subsequent runs with overlapping narratives skip inference entirely.
        Triples are always re-derived from the stored raw text, so changing
        the parsing logic never requires re-running the model.

        Parameters
        ----------
        df : DataFrame with narrative text and event-id columns.
        sample_n : Number of rows to process (None = all rows).
        batch_size : Number of prompts sent to the GPU at once.
        cache_path : Path to the JSON response cache.  Pass None to disable.

        Returns
        -------
        List of triple dicts with keys: ev_id, cause, relation, effect, source.
        """
        subset = df.dropna(subset=[text_col]).copy()
        if sample_n is not None and sample_n < len(subset):
            subset = subset.sample(n=sample_n, random_state=seed).reset_index(drop=True)

        texts  = subset[text_col].astype(str).tolist()
        ev_ids = subset[id_col].astype(str).tolist() if id_col in subset.columns else \
                 [str(i) for i in subset.index]

        # ------------------------------------------------------------------
        # Load cache and resolve which narratives still need inference
        # ------------------------------------------------------------------
        cache: dict = _load_cache(Path(cache_path)) if cache_path else {}
        cache_path_obj = Path(cache_path) if cache_path else None

        cached_hits    = sum(1 for eid in ev_ids if eid in cache)
        uncached_idx   = [i for i, eid in enumerate(ev_ids) if eid not in cache]

        if cached_hits:
            print(f"[LLM] Cache: {cached_hits}/{len(ev_ids)} narratives already cached "
                  f"→ {len(uncached_idx)} need inference")
        else:
            print(f"[LLM] Cache empty — running inference on all {len(ev_ids)} narratives")

        # ------------------------------------------------------------------
        # Inference on uncached narratives only
        # ------------------------------------------------------------------
        parse_errors = 0
        new_cache_entries = 0

        if uncached_idx:
            unc_texts  = [texts[i]  for i in uncached_idx]
            unc_ev_ids = [ev_ids[i] for i in uncached_idx]

            for start in tqdm(range(0, len(unc_texts), batch_size), desc="[LLM] Extracting"):
                batch_texts  = unc_texts[start : start + batch_size]
                batch_ev_ids = unc_ev_ids[start : start + batch_size]

                prompts   = [self._build_prompt(t) for t in batch_texts]
                responses = self._generate_batch(prompts)

                for text, ev_id, raw in zip(batch_texts, batch_ev_ids, responses):
                    # Try parsing; retry with fallback prompt on failure
                    if not _parse_triples(raw, ev_id) and max_retries > 0:
                        retry_raw = self._generate_batch([self._build_prompt(text, fallback=True)])[0]
                        # Store the retry response if it parses, else keep original
                        raw = retry_raw if _parse_triples(retry_raw, ev_id) else raw

                    cache[ev_id] = raw
                    new_cache_entries += 1

                # Write cache after every batch so a crash loses at most one batch
                if cache_path_obj:
                    _save_cache(cache, cache_path_obj)

        if new_cache_entries and cache_path_obj:
            print(f"[LLM] Cache updated — {new_cache_entries} new entries written to {cache_path_obj}")

        # ------------------------------------------------------------------
        # Derive triples from cache (all ev_ids, cached + newly inferred)
        # ------------------------------------------------------------------
        all_triples: List[dict] = []
        for ev_id in ev_ids:
            raw = cache.get(ev_id, "")
            triples = _parse_triples(raw, ev_id)
            if not triples:
                parse_errors += 1
            all_triples.extend(triples)

        total   = len(subset)
        ev_with = len({t["ev_id"] for t in all_triples})
        print(f"[LLM] Processed {total} narratives  |  "
              f"with ≥1 triple: {ev_with} ({ev_with/max(1,total):.1%})  |  "
              f"total triples: {len(all_triples)}  |  "
              f"parse errors: {parse_errors}")

        return all_triples
