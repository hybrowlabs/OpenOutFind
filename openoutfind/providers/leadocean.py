# openoutfind/providers/leadocean.py
"""Discovery from Lead Ocean: one JSON search call per page.

**ASSUMED CONTRACT — confirm against Lead Ocean before the Growth pilot (OPS-636).** The
real Lead Ocean API description was not available when this was written, so the transport
is the smallest thing that can carry a Lead Finder search, and *every* assumption lives in
this file:

    POST {OPENOUTFIND_LEADOCEAN_BASE_URL}/leads/search
    Authorization: Bearer {OPENOUTFIND_LEADOCEAN_API_KEY}
    body      {"filters": <Lead Finder filter set>, "limit": N, "offset": M}
    response  {"leads": [<row>, ...], "summary": {"leads_found": N}}

A row is read in Lead Finder's keys (``contact_full_name``, ``contact_job_title``,
``contact_linkedin_profile_url``, ``company_name`` ...). Where Lead Ocean names a field
differently, add it to ``FIELD_MAP`` below; nothing else changes. If the real search is
async (submit then poll), only ``_post`` changes. The filter translation is the second
place a real contract may differ: ``FILTER_MAP`` renames filter keys on the way out.

**Refusals are typed, never silent.** 401/403 is ``provider_auth``, 402 is
``provider_out_of_credits``, 429 (after a backed-off retry) is ``provider_rate_limited``,
and a network failure or a body that is not the shape above is ``provider_unavailable``.
An unreachable Lead Ocean is **not** read as a search that matches nobody, and it does not
fall back to another provider.
"""
from __future__ import annotations

import logging
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from openoutfind.core.config import SiteConfig
from openoutfind.core.errors import ErrorType, OpenOutFindError
from openoutfind.enrichment.provider import ProviderUnavailable

logger = logging.getLogger(__name__)

SEARCH_PATH = "/leads/search"
HTTP_TIMEOUT_S = 60

# Lead Ocean field name -> Lead Finder row key. Empty until the real contract says otherwise;
# rows already in Lead Finder keys pass through untouched.
FIELD_MAP: dict[str, str] = {}

# Lead Finder filter key -> Lead Ocean filter key. Empty for the same reason.
FILTER_MAP: dict[str, str] = {}

_RETRY = Retry(total=3, backoff_factor=2, status_forcelist=(429,), allowed_methods=None,
               respect_retry_after_header=True, raise_on_status=False)


class LeadOceanUnavailable(ProviderUnavailable):
    """Lead Ocean could not answer: refused, out of credits, rate-limited or unreachable."""


def search(config: SiteConfig, filters: dict, limit: int, offset: int):
    """One page from Lead Ocean; returns ``discovery.Fetched``."""
    from openoutfind.discovery import Fetched, Page

    validate(config)
    body = {"filters": _translate(filters), "limit": limit, "offset": offset}
    started = time.monotonic()
    payload = _post(config, body)
    leads = payload.get("leads")
    if not isinstance(leads, list):
        raise LeadOceanUnavailable("Lead Ocean returned no 'leads' list")
    summary = payload.get("summary") or {}
    found = summary.get("leads_found") if offset == 0 else None
    rows = [row for lead in leads if isinstance(lead, dict)
            if (row := _row(lead)).get("contact_linkedin_profile_url")]
    if len(rows) < len(leads):
        logger.debug("lead ocean: dropped %d row(s) with no profile URL", len(leads) - len(rows))
    return Fetched(Page(rows, found), time.monotonic() - started)


def validate(config: SiteConfig) -> None:
    """The base URL must be ``https://`` (loopback excepted, for a local stand-in).

    The bearer key rides every request, so a plain ``http://`` address would send it in the
    clear. ``bad_config`` stops the run before any call is made.
    """
    from urllib.parse import urlparse

    url = urlparse(config.leadocean_base_url)
    loopback = url.hostname in ("localhost", "127.0.0.1", "::1")
    if url.scheme == "https" and url.hostname:
        return
    if url.scheme == "http" and loopback:
        return
    raise OpenOutFindError(
        ErrorType.BAD_CONFIG,
        "OPENOUTFIND_LEADOCEAN_BASE_URL must be an https:// address "
        "(the API key is sent with every request)")


def _translate(filters: dict) -> dict:
    return {FILTER_MAP.get(key, key): value for key, value in filters.items()}


def _row(lead: dict) -> dict:
    return {FIELD_MAP.get(key, key): value for key, value in lead.items()}


def _post(config: SiteConfig, body: dict) -> dict:
    url = config.leadocean_base_url.rstrip("/") + SEARCH_PATH
    with requests.Session() as session:
        session.headers["Authorization"] = f"Bearer {config.leadocean_api_key}"
        adapter = HTTPAdapter(max_retries=_RETRY)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        try:
            response = session.post(url, json=body, timeout=HTTP_TIMEOUT_S)
        except requests.exceptions.RetryError as exc:
            raise LeadOceanUnavailable("Lead Ocean rate-limited this client through "
                                       "backed-off retries",
                                       ErrorType.PROVIDER_RATE_LIMITED) from exc
        except requests.RequestException as exc:
            raise LeadOceanUnavailable(f"Lead Ocean unreachable: {type(exc).__name__}") from exc

    if response.status_code in (401, 403):
        raise LeadOceanUnavailable(f"Lead Ocean rejected the API key ({response.status_code})",
                                   ErrorType.PROVIDER_AUTH)
    if response.status_code == 402:
        raise LeadOceanUnavailable("Lead Ocean credits are exhausted (402)",
                                   ErrorType.PROVIDER_OUT_OF_CREDITS)
    if response.status_code == 429:
        raise LeadOceanUnavailable("Lead Ocean is rate-limiting this client (429)",
                                   ErrorType.PROVIDER_RATE_LIMITED)
    if not response.ok:
        raise LeadOceanUnavailable(f"Lead Ocean answered HTTP {response.status_code}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise LeadOceanUnavailable("Lead Ocean returned a body that is not JSON") from exc
    if not isinstance(payload, dict):
        raise LeadOceanUnavailable("Lead Ocean returned a body that is not a JSON object")
    return payload
