# openoutfind/providers/__init__.py
"""Discovery sources behind the walk: the dataset, Lead Ocean, and BetterContact Lead Finder.

**Why this exists (hybrowlabs fork, OPS-636).** Upstream discovery is BetterContact Lead
Finder and nothing else. The walk only needs one thing from a source: *given a Lead Finder
filter set, a page size and an offset, return rows in Lead Finder's row shape and a count*.
That is ``discovery.Fetched``, and it is the seam. Everything downstream reads Lead Finder
row keys, so a source's whole job is to hand back rows in that shape.

``OPENOUTFIND_DISCOVERY_PROVIDER`` names the sources, one or several joined with ``+``:

* ``dataset`` — a CSV/JSONL file the operator supplies, filtered locally, free.
* ``leadocean`` — Lead Ocean's people search. **A page costs records** (one per person
  returned); the count does not.
* ``bettercontact`` (also ``leadfinder``) — upstream's Lead Finder, free, unchanged.
* ``dataset+leadocean+leadfinder`` — the default. The dataset is read first, then Lead
  Ocean, then Lead Finder, as one stream. A source that is not configured is skipped, so
  the default searches whichever of the three the operator actually has.

The count is the sum of the sources, reported at offset 0 only. A source that cannot say
how many it holds (Lead Ocean withholds its total) makes the whole count unknown: reporting
the others alone would tell the walk the search ends where they do.
"""
from __future__ import annotations

import json
import logging
import threading

from openoutfind.core.config import (
    SiteConfig, active_sources, discovery_provider, required_discovery_fields,
)

logger = logging.getLogger(__name__)

_LABELS = {"dataset": "the dataset", "leadocean": "Lead Ocean", "bettercontact": "Lead Finder"}


def provider_name(config: SiteConfig | None = None) -> str:
    """The configured sources as one string, e.g. ``dataset+leadocean``."""
    return discovery_provider(config or SiteConfig.load())


def is_configured(config: SiteConfig | None = None) -> bool:
    """Whether the chosen sources have what they need to search at all."""
    config = config or SiteConfig.load()
    return not required_discovery_fields(config)


def label(config: SiteConfig | None = None) -> str:
    """The human name of the sources a search will read, for log lines and status."""
    config = config or SiteConfig.load()
    return " + ".join(_LABELS[s] for s in active_sources(config)) or "no discovery source"


def search(config: SiteConfig, filters: dict, limit: int, offset: int, first_poll_s: float):
    """One page of the configured stream; returns ``discovery.Fetched``.

    Thread-safe and quiet, like ``discovery.search``: the walk fires a round's pages from
    worker threads. A source that cannot answer raises ``ProviderUnavailable`` (typed) or
    ``bad_config``; it never answers with a silent empty page, and a dead source never
    falls through to the next one.
    """
    sources = active_sources(config)
    if sources == ("bettercontact",):
        from openoutfind import discovery

        return discovery.search(config.bettercontact_api_key, filters, limit, offset,
                                first_poll_s)
    if sources == ("dataset",):
        from openoutfind.providers import dataset

        return dataset.search(config, filters, limit, offset)
    if sources == ("leadocean",):
        from openoutfind.providers import leadocean

        return leadocean.search(config, filters, limit, offset)
    return _search_stream(config, sources, filters, limit, offset, first_poll_s)


def _search_stream(config, sources, filters, limit, offset, first_poll_s):
    """The chosen sources in order, as one stream paged by ``offset``.

    Page ``offset..offset+limit`` is filled from the first source, then the next from where
    the previous ended. A source is counted only when the page needs to know where it
    starts (counts are cached per search). One that will not say how many it holds ends the
    stream: nothing after it can be placed, so it is read to its end and the sources behind
    it are neither counted nor reached.
    """
    import time

    from openoutfind.discovery import Fetched, Page

    started = time.monotonic()
    rows: list[dict] = []
    totals: list[int | None] = []
    passed = 0
    for source in sources:
        if offset and len(rows) >= limit:
            break  # the page is full; later sources need no count (only offset 0 reports one)
        total = _total(config, source, filters, first_poll_s)
        totals.append(total)
        if len(rows) < limit and (total is None or offset < passed + total):
            rows.extend(_slice(config, source, filters, limit - len(rows),
                               max(0, offset - passed), first_poll_s))
        if total is None:
            break
        passed += total

    unknown = any(total is None for total in totals)
    found = None if (offset or unknown) else sum(totals)
    return Fetched(Page(rows, found), time.monotonic() - started)

_totals_lock = threading.Lock()
_totals: dict[tuple, int | None] = {}


def _total(config, source, filters, first_poll_s):
    """One source's count for this search; ``None`` when it will not say.

    Lead Ocean and the dataset count for free. Lead Finder's count costs a page's time, so
    it is asked once per search and kept for the process.
    """
    if source == "dataset":
        from openoutfind.providers import dataset

        return dataset.count(config, filters)
    if source == "leadocean":
        from openoutfind.providers import leadocean

        return leadocean.count(config, filters)

    key = (source, json.dumps(filters, sort_keys=True, default=str))
    with _totals_lock:
        if key in _totals:
            return _totals[key]
    from openoutfind import discovery

    found = discovery.search(config.bettercontact_api_key, filters, limit=1, offset=0,
                             first_poll_s=first_poll_s).page.leads_found
    with _totals_lock:
        _totals[key] = found
    return found


def _slice(config, source, filters, limit, offset, first_poll_s):
    if source == "dataset":
        from openoutfind.providers import dataset

        return dataset.search(config, filters, limit, offset).page.leads
    if source == "leadocean":
        from openoutfind.providers import leadocean

        return leadocean.search(config, filters, limit, offset).page.leads
    from openoutfind import discovery

    return discovery.search(config.bettercontact_api_key, filters, limit, offset,
                            first_poll_s).page.leads