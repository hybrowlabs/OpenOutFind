# openoutfind/providers/__init__.py
"""Discovery sources behind the walk: Lead Ocean, an operator dataset, or upstream's
BetterContact Lead Finder.

**Why this exists (hybrowlabs fork, OPS-636).** Upstream discovery is BetterContact Lead
Finder and nothing else. The walk only needs one thing from a source: *given a Lead Finder
filter set, a page size and an offset, return rows in Lead Finder's row shape and a count*.
That is ``discovery.Fetched``, and it is the seam. Everything downstream — the headline
gate, the store, the embeddings, the qualifier — reads Lead Finder row keys
(``contact_full_name``, ``company_keywords`` ...), so a provider's whole job is to hand back
rows in that shape. Nothing else in the pipeline knows which source a row came from.

* ``bettercontact`` — upstream, unchanged (``discovery.search``).
* ``leadocean`` — HTTP search against Lead Ocean (``providers/leadocean.py``).
* ``dataset`` — a CSV/JSONL file the operator supplies, filtered locally
  (``providers/dataset.py``).
* ``leadocean+dataset`` — the fork's default: the dataset is read first, then Lead Ocean,
  as one continuous stream so offsets stay consistent across pages.

Selected by ``OPENOUTFIND_DISCOVERY_PROVIDER``. Credentials and paths are environment-only.
"""
from __future__ import annotations

from openoutfind.core.config import SiteConfig, discovery_provider, required_discovery_fields


def provider_name(config: SiteConfig | None = None) -> str:
    """The configured discovery provider's name (validated)."""
    return discovery_provider(config or SiteConfig.load())


def is_configured(config: SiteConfig | None = None) -> bool:
    """Whether the chosen provider has what it needs to search at all."""
    config = config or SiteConfig.load()
    return not required_discovery_fields(config)


def label(config: SiteConfig | None = None) -> str:
    """The human name of the source, for log lines and status."""
    return {
        "bettercontact": "BetterContact",
        "leadocean": "Lead Ocean",
        "dataset": "the dataset",
        "leadocean+dataset": "Lead Ocean + dataset",
    }[provider_name(config)]


def search(config: SiteConfig, filters: dict, limit: int, offset: int, first_poll_s: float):
    """Fetch one page from the configured source; returns ``discovery.Fetched``.

    Thread-safe and quiet, like ``discovery.search``: the walk fires a round's pages from
    worker threads. A source that cannot answer raises ``ProviderUnavailable`` (typed:
    ``provider_auth``, ``provider_out_of_credits``, ``provider_rate_limited``, or plain
    ``provider_unavailable``); it never answers with a silent empty page.
    """
    name = provider_name(config)
    if name == "bettercontact":
        from openoutfind import discovery

        return discovery.search(config.bettercontact_api_key, filters, limit, offset,
                                first_poll_s)

    from openoutfind.providers import dataset, leadocean

    if name == "dataset":
        return dataset.search(config, filters, limit, offset)
    if name == "leadocean":
        return leadocean.search(config, filters, limit, offset)
    return _search_both(config, filters, limit, offset)


def _search_both(config: SiteConfig, filters: dict, limit: int, offset: int):
    """The dataset first, then Lead Ocean, as one stream paged by ``offset``.

    A source that is not configured contributes nothing (``leadocean+dataset`` is satisfied
    by either). Page ``offset..offset+limit`` of the stream is the dataset's slice and, when
    that falls short of ``limit``, Lead Ocean's first rows from where the dataset ended. The
    count is the sum of both, and like upstream's is only reported at offset 0.
    """
    import time

    from openoutfind.discovery import Fetched, Page
    from openoutfind.providers import dataset, leadocean

    started = time.monotonic()
    use_dataset = bool(config.dataset_path)
    use_ocean = bool(config.leadocean_api_key and config.leadocean_base_url)

    rows: list[dict] = []
    dataset_total = 0
    if use_dataset:
        local = dataset.search(config, filters, limit, offset)
        rows.extend(local.page.leads)
        dataset_total = dataset.count(config, filters)
    if use_ocean and len(rows) < limit:
        remote_offset = max(0, offset - dataset_total)
        remote = leadocean.search(config, filters, limit - len(rows), remote_offset)
        rows.extend(remote.page.leads)
        found_remote = remote.page.leads_found
    else:
        found_remote = None

    found = None
    if offset == 0:
        # The stream's size. When the dataset alone filled page one, Lead Ocean was not
        # asked (no network call), so the count is a lower bound; a count above zero is all
        # the walk needs from it (zero is what marks a search dead).
        found = dataset_total + (found_remote or 0)
    return Fetched(Page(rows, found), time.monotonic() - started)
