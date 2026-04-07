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
    "You are an aviation safety analyst extracting causal relationships from NTSB accident narratives. "
    "Use the standard NTSB causal vocabulary whenever possible so your output aligns with "
    "established aviation safety terminology and classification. "
    "Return only a valid JSON array — no markdown, no explanation, no extra text."
)

# ---------------------------------------------------------------------------
# Terminology blocks injected into prompts
# ---------------------------------------------------------------------------

# Approved relation phrases: derived from the rule-based extraction vocabulary,
# listed in descending frequency order from the NTSB corpus.
_APPROVED_RELATIONS = (
    "resulted in, led to, caused, contributed to, triggered, produced, "
    "due to, because of, caused by, attributed to, as a result of, "
    "resulting from, stemmed from, precluded, prevented, did not prevent"
)

# NTSB causal category terminology: mirrors _CATEGORY_KEYWORDS in finding_evaluator.py.
# Guides the LLM to use vocabulary that matches the official finding taxonomy.
_CATEGORY_VOCAB = """\
NTSB causal category vocabulary — use these terms in cause/effect spans:
  Personnel issues : pilot, crew, captain, officer, student, instructor, decision,
      judgment, attention, situational awareness, fatigue, training, procedure,
      checklist, error, workload, distraction, scan, monitor, experience,
      planning, action, omission, communication, coordination
  Aircraft         : engine, fuel, power, propeller, rotor, blade, gear, brake,
      flap, control, aileron, elevator, rudder, hydraulic, electrical, battery,
      circuit, mechanical, structural, airframe, component, system, failure,
      malfunction, corrosion, wear, carburetor, manifold, exhaust, oil,
      ignition, magneto, cylinder, piston, crankshaft, bearing
  Environmental    : weather, wind, gust, turbulence, icing, ice, fog,
      visibility, cloud, ceiling, precipitation, rain, snow, density altitude,
      terrain, obstacle, bird, wildlife, night, dark
  Organizational   : maintenance, management, oversight, regulation, policy,
      inspection, supervision, dispatch, scheduling"""

_TERMINOLOGY_BLOCK = f"""\
REQUIRED — use one of these approved relation phrases whenever it accurately fits:
  {_APPROVED_RELATIONS}
Only use a different phrase if none of the above captures the relationship.

{_CATEGORY_VOCAB}\
"""

def _make_user_tmpl(examples_placeholder: bool = False) -> str:
    """Build the user prompt template string with terminology embedded.

    Using a function avoids the double-brace escaping problem that arises when
    pre-formatting a template that also contains JSON example syntax.
    """
    header = (
        "Extract all causal relationships from the following aviation accident narrative.\n\n"
        "For each relationship output an object with exactly these keys:\n"
        '  "cause"    — the specific factor, condition, or event that caused something\n'
        '  "relation" — the causal verb/phrase connecting cause to effect\n'
        '  "effect"   — what was caused or resulted\n\n'
        + _TERMINOLOGY_BLOCK + "\n\n"
        # Double-brace the JSON example so .format(narrative=...) treats them as literals
        'Return a JSON array: [{{"cause": "...", "relation": "...", "effect": "..."}}]\n'
        "If no causal relationships are present, return: []"
    )
    if examples_placeholder:
        return header + "\n\n{examples}\nNow extract triples from this narrative:\n{narrative}"
    return header + "\n\nNarrative:\n{narrative}"


_USER_TMPL        = _make_user_tmpl(examples_placeholder=False)
_USER_FEWSHOT_TMPL = _make_user_tmpl(examples_placeholder=True)

# Simpler fallback prompt used on retry after a JSON parse failure
_USER_FALLBACK_TMPL = """\
Read this aviation accident text and output ONLY a valid JSON array.
Each item must have exactly three string keys: "cause", "relation", "effect".
Use these relation phrases: resulted in, caused, led to, due to, contributed to.
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


def build_few_shot_examples(
    train_ev_ids: List[str],
    df: pd.DataFrame,
    cache: dict,
    findings_df: Optional['pd.DataFrame'] = None,
    n_per_category: int = 3,
    max_narrative_chars: int = 400,
) -> str:
    """
    Build a few-shot example block from training narratives.

    Selects up to n_per_category examples per NTSB top-level category
    from the training split. For each example, shows a narrative snippet
    alongside the triples already cached for that narrative.

    Returns a formatted string ready to inject into the prompt.
    If findings_df is provided, examples are stratified by category;
    otherwise a random sample from the cache is used.
    """
    import random

    train_set = set(str(e) for e in train_ev_ids)
    id_col = 'ev_id' if 'ev_id' in df.columns else df.columns[0]
    text_col = 'narr_clean' if 'narr_clean' in df.columns else df.columns[1]

    # Build lookup: ev_id -> (narrative, category)
    df_train = df[df[id_col].astype(str).isin(train_set)].copy()
    df_train[id_col] = df_train[id_col].astype(str)

    if findings_df is not None and 'category' in findings_df.columns:
        cat_col = 'category'
        merged = df_train.merge(
            findings_df[['ev_id', cat_col]].drop_duplicates('ev_id'),
            left_on=id_col, right_on='ev_id', how='left',
        )
        categories = ['Personnel issues', 'Aircraft', 'Environmental issues']
        selected_ids = []
        for cat in categories:
            pool = merged[merged[cat_col] == cat][id_col].tolist()
            # Prefer narratives that are already in the cache with parseable output
            pool_cached = [eid for eid in pool if eid in cache and _parse_triples(cache[eid], eid)]
            sample_pool = pool_cached if pool_cached else pool
            random.seed(42)
            selected_ids.extend(random.sample(sample_pool, min(n_per_category, len(sample_pool))))
    else:
        # Fallback: random sample from cached training narratives
        pool = [eid for eid in df_train[id_col].tolist() if eid in cache]
        random.seed(42)
        selected_ids = random.sample(pool, min(n_per_category * 3, len(pool)))

    # Build formatted example block
    lines = ["Reference examples from similar accident reports:"]
    for i, eid in enumerate(selected_ids, 1):
        row = df_train[df_train[id_col] == eid]
        if row.empty:
            continue
        narrative = str(row[text_col].iloc[0])[:max_narrative_chars]
        triples = _parse_triples(cache.get(eid, ''), eid)
        if not triples:
            continue
        # Use first 2 triples to keep prompt compact
        triple_json = json.dumps(
            [{'cause': t['cause'], 'relation': t['relation'], 'effect': t['effect']}
             for t in triples[:2]],
            ensure_ascii=False,
        )
        lines.append(f"\nExample {i}:")
        lines.append(f"Narrative: \"{narrative}...\"")
        lines.append(f"Output: {triple_json}")

    return '\n'.join(lines)


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

    def _build_prompt(
        self,
        narrative: str,
        fallback: bool = False,
        few_shot_block: str = '',
    ) -> str:
        """Format prompt using the model's chat template if available."""
        if fallback:
            user_content = _USER_FALLBACK_TMPL.format(narrative=narrative[:1500])
        elif few_shot_block:
            user_content = _USER_FEWSHOT_TMPL.format(
                examples=few_shot_block,
                narrative=narrative[:1200],  # slightly shorter to leave room for examples
            )
        else:
            user_content = _USER_TMPL.format(narrative=narrative[:1500])

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
            max_length=3000,
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
        restrict_ev_ids: Optional[List[str]] = None,
        few_shot_block: str = '',
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
        restrict_ev_ids : If given, only process narratives whose ev_id is in
            this list (e.g. the held-out test split for unified evaluation).
        few_shot_block : Pre-built example string from build_few_shot_examples().
            When non-empty, injects examples into every prompt and uses a
            separate cache to avoid contaminating the zero-shot cache.

        Returns
        -------
        List of triple dicts with keys: ev_id, cause, relation, effect, source.
        """
        subset = df.dropna(subset=[text_col]).copy()
        if sample_n is not None and sample_n < len(subset):
            subset = subset.sample(n=sample_n, random_state=seed).reset_index(drop=True)

        # Filter to specific ev_ids if requested (e.g. test split)
        if restrict_ev_ids is not None:
            restrict_set = set(str(e) for e in restrict_ev_ids)
            subset = subset[subset[id_col].astype(str).isin(restrict_set)]
            subset = subset.reset_index(drop=True)

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

        mode_str = "few-shot" if few_shot_block else "zero-shot"
        if cached_hits:
            print(f"[LLM] Cache ({mode_str}): {cached_hits}/{len(ev_ids)} narratives cached "
                  f"→ {len(uncached_idx)} need inference")
        else:
            print(f"[LLM] Cache ({mode_str}) empty — running inference on all {len(ev_ids)} narratives")

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

                prompts = [self._build_prompt(t, few_shot_block=few_shot_block)
                           for t in batch_texts]
                responses = self._generate_batch(prompts)

                for text, ev_id, raw in zip(batch_texts, batch_ev_ids, responses):
                    # Try parsing; retry with fallback prompt on failure
                    if not _parse_triples(raw, ev_id) and max_retries > 0:
                        retry_raw = self._generate_batch([self._build_prompt(text, fallback=True)])[0]
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
              f"with >=1 triple: {ev_with} ({ev_with/max(1,total):.1%})  |  "
              f"total triples: {len(all_triples)}  |  "
              f"parse errors: {parse_errors}")

        return all_triples
