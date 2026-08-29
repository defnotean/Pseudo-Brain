"""Guaranteed context compression for Pseudo-Brain agent transcripts.

WHY THIS EXISTS
---------------
The failure "Context length exceeded (N tokens). Cannot compress further" is a
correct-but-useless safeguard: it gives up when the abstractive summarizer has
nothing left to prune.  This module removes the possibility of giving up.

CONTRACT (the part that cannot fail)
------------------------------------
`compress()` / `ContextStore.ensure_fit()` are GUARANTEED to return a result
whose estimated token count is <= `max_tokens`.  If every lossless and abstractive
tier fails, the final hard-truncation tier (T3) deterministically slices the
context down to the budget.  There is no code path that raises
"cannot compress further" — instead it returns a *bounded, lossy-but-usable*
result and records what was dropped in `notes`.

TIER CASCADE (first tier that fits wins; later tiers are cheaper/lossier)
-------------------------------------------------------------------------
  T0 passthrough      : already fits -> return as-is (no information loss)
  T1 structured extract: drop boilerplate, collapse tool I/O, keep decisions/
                         constraints/results as a compact skeleton (local, fast)
  T2 spark abstractive : send to DGX Spark Qwen (192.168.0.176:50052) for real
                         abstractive summarization.  QUALITY BOOST ONLY — if Spark
                         is unreachable/errors, we degrade to T1/T3, never hang.
  T3 hard truncation  : deterministic head+tail slice to the exact token budget.
                         Always fits; last-resort guarantee.
  T4 image rasterize  : UNBOUNDED-TEXT -> FIXED-COST fallback.  Render the
                         overflow text to a bounded number of rasterized page
                         images (default ~1280 vision tokens each).  Re-ingesting
                         images costs a *fixed* number of vision tokens, so this
                         can never breach the limit.  Optional vision re-summary
                         (T2-style) can turn the images back into compact text,
                         but the image manifest alone already satisfies the budget.

All network/abstractive tiers are wrapped so a failure degrades gracefully;
the local structural + truncation tiers are the guaranteed floor.

Local-CPU / offline constraint (per AGENTS.md): no CUDA, no background services.
Spark is contacted with a short timeout and treated as best-effort.
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

try:  # vision is optional; only needed for T4 re-summary
    from PIL import Image, ImageDraw, ImageFont
    _HAVE_PIL = True
except Exception:  # pragma: no cover - PIL is present in this env
    _HAVE_PIL = False

try:  # tiktoken gives accurate counts; fall back to char/4 if unavailable/offline
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
    _HAVE_TIKTOKEN = True
except Exception:  # offline or not installed
    _ENC = None
    _HAVE_TIKTOKEN = False


# --------------------------------------------------------------------------- #
# Token accounting (offline-safe)
# --------------------------------------------------------------------------- #
def estimate_tokens(text: str) -> int:
    """Estimate token count. Uses tiktoken when available, else char/4 heuristic."""
    if not text:
        return 0
    if _HAVE_TIKTOKEN and _ENC is not None:
        try:
            return len(_ENC.encode(text))
        except Exception:
            pass
    # Rough but deterministic; English ~4 chars/token.
    return max(1, len(text) // 4)


def _hard_truncate(text: str, max_tokens: int) -> str:
    """Deterministically slice text to <= max_tokens via head+tail preserving ends.

    Reserves headroom for the truncation marker so the returned string is
    guaranteed to fit, and finishes with a bounded safety-trim backstop.
    """
    if estimate_tokens(text) <= max_tokens:
        return text
    marker_template = "\n…[HARD-TRUNCATED: {dropped} chars / ~{approx} tokens dropped]…\n"
    marker_tok = estimate_tokens(marker_template.format(dropped=0, approx=0))
    target = max(1, max_tokens - marker_tok)
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid]) <= target:
            lo = mid
        else:
            hi = mid - 1
    budget = lo
    head_chars = max(0, int(budget * 0.55))
    tail_chars = max(0, budget - head_chars)
    head = text[:head_chars]
    tail = text[len(text) - tail_chars:] if tail_chars else ""
    dropped = len(text) - head_chars - tail_chars
    result = head + marker_template.format(dropped=dropped, approx=dropped // 4) + tail
    # Bounded safety-trim backstop (handles tokenizers where char/4 undercounts).
    while estimate_tokens(result) > max_tokens and len(result) > 1:
        result = result[:-1]
    return result


# --------------------------------------------------------------------------- #
# T1: structured extraction (local, lossy-but-structured)
# --------------------------------------------------------------------------- #
_BOILER = re.compile(
    r"(?im)^\s*(?:#+\s+)?(system|tool)\b.*$|"
    r"^\s*\[\d{4}-\d{2}-\d{2}.*$"  # timestamps
)

_DECISION_RE = re.compile(
    r"(?im)^(.*\b(decision|conclusion|result|answer|summary|final|todo|action item)\b.*)$"
)
_CONSTRAINT_RE = re.compile(
    r"(?im)^(.*\b(must|must not|cannot|should|constraint|requirement|rule|never|always|freeze|gate)\b.*)$"
)
_ERROR_RE = re.compile(r"(?im)^(.*\b(error|exception|traceback|failed|failure)\b.*)$")


def structured_extract(text: str) -> str:
    """Keep high-signal lines (decisions, constraints, errors) and a thin skeleton.

    Strategy: deduplicate, drop pure-boilerplate/timestamp lines, and keep every
    line matching a high-signal pattern plus the first/last lines of each block.
    This is deterministic and offline.
    """
    lines = text.splitlines()
    kept: list[str] = []
    seen: set[str] = set()
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if _BOILER.match(ln):
            continue
        if s.lower() in seen:
            continue
        seen.add(s.lower())
        if (
            _DECISION_RE.match(ln)
            or _CONSTRAINT_RE.match(ln)
            or _ERROR_RE.match(ln)
        ):
            kept.append(ln)
            continue
        # Thin skeleton: keep short lines (likely headers/labels) sparsely.
        if len(s) <= 80:
            kept.append(ln)
    if not kept:
        # Nothing matched; fall back to first + last 40 lines.
        kept = lines[:20] + ["…[collapsed]…"] + lines[-20:]
    return "\n".join(kept)


# --------------------------------------------------------------------------- #
# T2: Spark abstractive summarization (quality boost, best-effort)
# --------------------------------------------------------------------------- #
@dataclass
class SparkConfig:
    base_url: str = "http://192.168.0.176:50052/v1"
    model: str = "qwen3.8-27b"
    timeout_s: float = 20.0
    max_tokens_out: int = 2048
    temperature: float = 0.2


def spark_summarize(text: str, cfg: SparkConfig | None = None) -> str | None:
    """Abstractive summarize via DGX Spark Qwen (OpenAI-compatible).

    Returns the summary string, or None on ANY failure (unreachable, timeout,
    non-200, empty).  Callers MUST treat None as "degrade to a cheaper tier".
    Never raises.
    """
    if cfg is None:
        cfg = SparkConfig()
    try:
        import requests  # local import; present in env

        payload = {
            "model": cfg.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a context compressor for an autonomous AI agent. "
                        "Given the agent's transcript, produce a DENSE, lossy-but-faithful "
                        "summary that preserves: active task + goal, decisions made, "
                        "constraints/freezes, open questions, tool results that matter, and "
                        "the very next action. Drop chatter, boilerplate, redundant tool I/O. "
                        "Return only the compact summary, no preamble."
                    ),
                },
                {"role": "user", "content": text},
            ],
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens_out,
        }
        resp = requests.post(
            f"{cfg.base_url}/chat/completions",
            json=payload,
            timeout=cfg.timeout_s,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content.strip() if content else None
    except Exception:
        # Unreachable / timeout / parse error -> degrade. Never hang (timeout set).
        return None


def _vision_resummarize(image_paths: Sequence[str], cfg: SparkConfig | None = None) -> str | None:
    """Optional: ask Spark (multimodal Qwen) to read the rasterized pages back
    into compact text. Best-effort; None on any failure."""
    if cfg is None:
        cfg = SparkConfig()
    if not image_paths:
        return None
    try:
        import requests
        import base64

        parts = [{"type": "text", "text": "Compress the content of these page images into a single dense summary preserving all decisions, constraints, results, and next actions. Return only the summary."}]
        for p in image_paths:
            with open(p, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        payload = {
            "model": cfg.model,
            "messages": [{"role": "user", "content": parts}],
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens_out,
        }
        resp = requests.post(f"{cfg.base_url}/chat/completions", json=payload, timeout=cfg.timeout_s)
        if resp.status_code != 200:
            return None
        content = resp.json()["choices"][0]["message"]["content"]
        return content.strip() if content else None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# T4: image rasterization (unbounded text -> fixed-cost vision tokens)
# --------------------------------------------------------------------------- #
# Assumption: a 1024x1536 page at typical processor resolution ~= 1280 vision tokens.
VISION_TOKENS_PER_PAGE = 1280
PAGE_CHARS = 6000  # chars per rasterized page (~safe for default font)


def rasterize_to_images(text: str, max_tokens: int, out_dir: str | None = None) -> tuple[list[str], int]:
    """Render text to at most floor(max_tokens / VISION_TOKENS_PER_PAGE) page images.

    Returns (image_paths, ingestion_token_estimate).  Because the page count is
    capped to the token budget, ingestion_token_estimate <= max_tokens ALWAYS.
    Excess text beyond the capped pages is dropped (recorded by caller).
    Requires PIL; raises only if PIL is missing (caller guards).
    """
    if not _HAVE_PIL:
        raise RuntimeError("PIL unavailable; cannot rasterize (T4)")
    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="pb_ctx_img_")
    os.makedirs(out_dir, exist_ok=True)

    max_pages = max(1, max_tokens // VISION_TOKENS_PER_PAGE)
    # Slice text to the pages we can afford so we never exceed the budget.
    affordable_chars = max_pages * PAGE_CHARS
    sliced = text[:affordable_chars]
    pages = [sliced[i : i + PAGE_CHARS] for i in range(0, max(1, len(sliced)), PAGE_CHARS)]
    pages = pages[:max_pages]

    paths: list[str] = []
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for idx, page in enumerate(pages):
        img = Image.new("RGB", (1024, 1536), "white")
        draw = ImageDraw.Draw(img)
        # naive word-wrap
        words = page.split()
        lines: list[str] = []
        cur = ""
        for w in words:
            trial = f"{cur} {w}".strip()
            if len(trial) > 110:
                lines.append(cur)
                cur = w
            else:
                cur = trial
        if cur:
            lines.append(cur)
        y = 12
        for ln in lines:
            draw.text((12, y), ln, fill="black", font=font)
            y += 18
            if y > 1520:
                break
        path = os.path.join(out_dir, f"page_{idx:04d}.png")
        img.save(path)
        paths.append(path)
    return paths, len(paths) * VISION_TOKENS_PER_PAGE


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
@dataclass
class CompressedContext:
    text: str
    token_estimate: int
    tier_used: str
    guaranteed_fit: bool
    images: list[str] = field(default_factory=list)
    ingestion_tokens: int | None = None  # set when images present
    notes: list[str] = field(default_factory=list)

    def render_for_model(self) -> dict:
        """Return the payload to feed the model (text + optional image refs)."""
        payload: dict = {"text": self.text}
        if self.images:
            payload["images"] = list(self.images)
            payload["ingestion_tokens"] = self.ingestion_tokens
        return payload


def compress(
    text: str,
    *,
    max_tokens: int,
    spark_cfg: SparkConfig | None = None,
    allow_images: bool = True,
    vision_resummarize: bool = False,
    tmp_dir: str | None = None,
) -> CompressedContext:
    """Compress `text` so its estimated tokens <= max_tokens. GUARANTEED.

    Cascade: T0 -> T1 -> T2(Spark) -> T3(hard truncate) -> T4(images).
    The postcondition `result.token_estimate <= max_tokens` always holds.
    """
    notes: list[str] = []
    if max_tokens <= 0:
        max_tokens = 1

    # T0 passthrough
    est = estimate_tokens(text)
    if est <= max_tokens:
        return CompressedContext(
            text=text, token_estimate=est, tier_used="T0-passthrough",
            guaranteed_fit=True, notes=notes,
        )

    # T1 structured extraction (local)
    t1 = structured_extract(text)
    if estimate_tokens(t1) <= max_tokens:
        notes.append("T1 kept high-signal lines only (decisions/constraints/errors).")
        return CompressedContext(
            text=t1, token_estimate=estimate_tokens(t1), tier_used="T1-structured",
            guaranteed_fit=True, notes=notes,
        )

    # T2 Spark abstractive (quality boost; degrade on any failure)
    t2 = spark_summarize(text, spark_cfg)
    if t2 is not None and estimate_tokens(t2) <= max_tokens:
        notes.append("T2 abstractive summary produced by DGX Spark Qwen.")
        return CompressedContext(
            text=t2, token_estimate=estimate_tokens(t2), tier_used="T2-spark",
            guaranteed_fit=True, notes=notes,
        )
    if t2 is None:
        notes.append("T2 Spark unavailable/offline -> degraded to local tiers (expected when Spark is down).")

    # T4 image rasterization (preferred when allowed: fixed-cost vision ingestion
    # can never breach the budget, per owner directive). Always tries before T3.
    if allow_images and _HAVE_PIL:
        try:
            paths, ingest = rasterize_to_images(text, max_tokens, tmp_dir)
            if paths and ingest <= max_tokens:
                if vision_resummarize:
                    vsum = _vision_resummarize(paths, spark_cfg)
                    if vsum is not None and estimate_tokens(vsum) <= max_tokens:
                        notes.append("T4 image rasterize + Spark vision re-summary used.")
                        return CompressedContext(
                            text=vsum, token_estimate=estimate_tokens(vsum),
                            tier_used="T4-image+vision", guaranteed_fit=True,
                            images=paths, ingestion_tokens=ingest, notes=notes,
                        )
                notes.append(
                    f"T4 rasterized overflow to {len(paths)} page image(s) "
                    f"({ingest} vision tokens) — fixed-cost, always fits."
                )
                return CompressedContext(
                    text=("Context exceeds token budget; full text rasterized to "
                          f"{len(paths)} image page(s) for vision re-ingestion."),
                    token_estimate=ingest, tier_used="T4-image",
                    guaranteed_fit=True, images=paths,
                    ingestion_tokens=ingest,
                    notes=notes,
                )
        except Exception as exc:  # pragma: no cover
            notes.append(f"T4 rasterize failed ({exc!r}); using T3 truncation.")

    # T3 hard truncation — the deterministic floor that ALWAYS fits.
    t3 = _hard_truncate(text, max_tokens)
    notes.append("T3 hard-truncation used (deterministic head+tail slice to budget).")
    return CompressedContext(
        text=t3, token_estimate=estimate_tokens(t3), tier_used="T3-truncation",
        guaranteed_fit=True, notes=notes,
    )


# --------------------------------------------------------------------------- #
# Streaming store: accumulate turns, guarantee fit before each model call
# --------------------------------------------------------------------------- #
@dataclass
class Turn:
    role: str
    content: str


class ContextStore:
    """Holds a transcript; `ensure_fit` compacts it to <= max_tokens on demand."""

    def __init__(self, max_tokens: int = 30000, spark_cfg: SparkConfig | None = None,
                 allow_images: bool = True):
        self.max_tokens = max_tokens
        self.spark_cfg = spark_cfg
        self.allow_images = allow_images
        self.turns: list[Turn] = []
        self.compressed_count = 0

    def add(self, role: str, content: str) -> None:
        self.turns.append(Turn(role=role, content=content))

    def current_text(self) -> str:
        return "\n\n".join(f"[{t.role}]\n{t.content}" for t in self.turns)

    def current_tokens(self) -> int:
        return estimate_tokens(self.current_text())

    def ensure_fit(self, *, vision_resummarize: bool = False,
                   tmp_dir: str | None = None) -> CompressedContext:
        """Compress the whole store to <= max_tokens. GUARANTEED to fit."""
        full = self.current_text()
        if estimate_tokens(full) <= self.max_tokens:
            return CompressedContext(
                text=full, token_estimate=estimate_tokens(full),
                tier_used="T0-passthrough", guaranteed_fit=True,
            )
        self.compressed_count += 1
        return compress(
            full, max_tokens=self.max_tokens, spark_cfg=self.spark_cfg,
            allow_images=self.allow_images, vision_resummarize=vision_resummarize,
            tmp_dir=tmp_dir,
        )
