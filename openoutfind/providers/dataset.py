# openoutfind/providers/dataset.py
"""Discovery from an operator-supplied dataset file (CSV or JSON Lines), filtered locally.

The walk fires Lead Finder-shaped filter sets. Here they are applied to rows already on
disk, so the source costs nothing, needs no network and never leaves the machine.

**File format.** One row per person, UTF-8, ``.csv`` (header row) or ``.jsonl``/``.ndjson``.

    required   linkedin_url  full_name  job_title  company_name
    optional   headline  seniority  industry  company_domain  company_keywords
               company_industry  city  state  country  employees_min  employees_max

``linkedin_url`` is the person's identity (the store keys leads on it), so a row without
one is skipped and a repeated one keeps its first row. ``company_keywords`` is a list in
JSONL, or ``;``/``|``/``,`` separated text in CSV. A missing ``headline`` is written as
``"<job_title> at <company_name>"``: upstream drops rows with no headline as forgotten
profiles, a LinkedIn heuristic that has no meaning for a curated dataset.

**Matching follows upstream's measured rules** (``discovery.py``): a title needs every word,
in any order, stopwords dropped; a company tag is a phrase held adjacent and in order inside
one tag; a place is ``"State, Country"`` or ``"Country"`` and a row with no state matches any
state of its country; seniority is a whole value; the size band is an overlap, and a row
with no size is kept rather than dropped.
"""
from __future__ import annotations

import csv
import json
import logging
import threading
import time
from pathlib import Path

from openoutfind.core.config import SiteConfig
from openoutfind.core.errors import ErrorType, OpenOutFindError
from openoutfind.core.pipeline import analyzers

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("linkedin_url", "full_name", "job_title", "company_name")

# dataset column -> the Lead Finder row key the rest of the pipeline reads
_COLUMN_TO_ROW = {
    "linkedin_url": "contact_linkedin_profile_url",
    "full_name": "contact_full_name",
    "job_title": "contact_job_title",
    "headline": "contact_headline",
    "seniority": "contact_seniority",
    "industry": "contact_industry",
    "company_name": "company_name",
    "company_domain": "company_domain",
    "company_industry": "company_industry",
    "state": "contact_location_state",
    "country": "contact_location_country",
    "city": "contact_location_city",
    "employees_min": "company_employees_range_start",
    "employees_max": "company_employees_range_end",
}

_cache_lock = threading.Lock()
_cache: dict[tuple, list[dict]] = {}


def rows(config: SiteConfig) -> list[dict]:
    """The dataset as Lead Finder-shaped rows, read once per file version."""
    path = _path(config)
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    with _cache_lock:
        if key not in _cache:
            _cache.clear()
            _cache[key] = _load(path)
        return _cache[key]


def count(config: SiteConfig, filters: dict) -> int:
    """How many dataset rows match ``filters``."""
    return sum(1 for row in rows(config) if _matches(row, filters))


def search(config: SiteConfig, filters: dict, limit: int, offset: int):
    """One page of matching rows; returns ``discovery.Fetched``.

    ``leads_found`` is reported at offset 0 only, like the provider it stands in for.
    """
    from openoutfind.discovery import Fetched, Page

    started = time.monotonic()
    matching = [row for row in rows(config) if _matches(row, filters)]
    page = [dict(row) for row in matching[offset:offset + limit]]
    found = len(matching) if offset == 0 else None
    return Fetched(Page(page, found), time.monotonic() - started)


# ── loading ──────────────────────────────────────────────────────────────────

def _path(config: SiteConfig) -> Path:
    if not config.dataset_path:
        raise _bad("OPENOUTFIND_DATASET_PATH is not set")
    path = Path(config.dataset_path).expanduser()
    if not path.is_file():
        raise _bad(f"dataset file not found: {path}")
    return path


def _load(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        records = _read_csv(path)
    elif suffix in (".jsonl", ".ndjson"):
        records = _read_jsonl(path)
    else:
        raise _bad(f"dataset must be .csv or .jsonl, got {path.name!r}")

    seen: set[str] = set()
    out: list[dict] = []
    skipped = 0
    for record in records:
        row = _to_row(record)
        url = row.get("contact_linkedin_profile_url")
        if not url or url in seen:
            skipped += 1
            continue
        seen.add(url)
        out.append(row)
    if skipped:
        logger.debug("dataset: skipped %d row(s) with no linkedin_url or a repeated one", skipped)
    logger.info("dataset: %d lead(s) loaded from %s", len(out), path.name)
    return out


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader.fieldnames or [], path)
        return list(reader)


def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise _bad(f"{path.name} line {number} is not valid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise _bad(f"{path.name} line {number} is not a JSON object")
            records.append(record)
    if records:
        _require_columns(records[0].keys(), path)
    return records


def _require_columns(columns, path: Path) -> None:
    absent = [name for name in REQUIRED_COLUMNS if name not in set(columns)]
    if absent:
        raise _bad(f"{path.name} is missing required column(s): {', '.join(absent)}")


def _to_row(record: dict) -> dict:
    """One dataset record as a Lead Finder row; blanks are left out, never empty strings."""
    row: dict = {}
    for column, key in _COLUMN_TO_ROW.items():
        value = record.get(column)
        text = str(value).strip() if value is not None else ""
        if not text:
            continue
        if column in ("employees_min", "employees_max"):
            try:
                row[key] = int(float(text))
            except ValueError:
                continue
        else:
            row[key] = text

    tags = record.get("company_keywords")
    if isinstance(tags, list):
        tags = [str(t).strip() for t in tags if str(t).strip()]
    elif tags:
        text = str(tags).replace("|", ";").replace(",", ";")
        tags = [t.strip() for t in text.split(";") if t.strip()]
    if tags:
        row["company_keywords"] = tags

    if "company_industry" not in row and "contact_industry" in row:
        row["company_industry"] = row["contact_industry"]
    if "contact_headline" not in row and row.get("contact_job_title") and row.get("company_name"):
        row["contact_headline"] = f"{row['contact_job_title']} at {row['company_name']}"
    return row


# ── matching ─────────────────────────────────────────────────────────────────

def _matches(row: dict, filters: dict) -> bool:
    for key, spec in filters.items():
        if key in ("company_headcount_min", "company_headcount_max"):
            continue
        values = _include(spec)
        if not values:
            continue
        if key == "lead_job_title" and not _title(row, values):
            return False
        if key == "company_keywords" and not _tag(row, values):
            return False
        if key == "lead_location" and not _place(row, values):
            return False
        if key == "lead_seniority" and not _seniority(row, values):
            return False
    return _size(row, filters.get("company_headcount_min"), filters.get("company_headcount_max"))


def _include(spec) -> list[str]:
    if isinstance(spec, dict):
        return [str(v) for v in spec.get("include", []) if str(v).strip()]
    return [str(spec)] if spec else []


def _title(row: dict, values: list[str]) -> bool:
    have = analyzers.title_words(row.get("contact_job_title") or "")
    return any(analyzers.query_title_words(v) <= have for v in values)


def _tag(row: dict, values: list[str]) -> bool:
    tags = analyzers.company_tags(", ".join(row.get("company_keywords") or []))
    return any(any(analyzers.holds_phrase(tag, analyzers.phrase_words(v)) for tag in tags)
               for v in values)


def _place(row: dict, values: list[str]) -> bool:
    country = (row.get("contact_location_country") or "").strip().lower()
    state = (row.get("contact_location_state") or "").strip().lower()
    for value in values:
        parts = [p.strip().lower() for p in value.split(",") if p.strip()]
        if not parts:
            continue
        want_country, want_state = parts[-1], (parts[0] if len(parts) > 1 else "")
        if country == want_country and (not want_state or not state or state == want_state):
            return True
    return False


def _seniority(row: dict, values: list[str]) -> bool:
    return (row.get("contact_seniority") or "").strip().lower() in {v.lower() for v in values}


def _size(row: dict, low, high) -> bool:
    start = row.get("company_employees_range_start")
    end = row.get("company_employees_range_end")
    if start is None and end is None:
        return True  # unknown size is kept, not dropped
    start = start if start is not None else end
    end = end if end is not None else start
    if low is not None and end < low:
        return False
    if high is not None and start > high:
        return False
    return True


def _bad(message: str) -> OpenOutFindError:
    return OpenOutFindError(ErrorType.BAD_CONFIG, message)
