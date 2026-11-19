# openoutfind/providers/leadocean.py
"""Discovery from Lead Ocean's people search, written against the vendor's public spec.

Contract (api.leadocean.io/openapi.json and llms-full.txt, read 19 Nov 2026 IST, no key
needed to read them and none used here):

    POST {OPENOUTFIND_LEADOCEAN_BASE_URL}/v1/people/search
    header    x-api-key: {OPENOUTFIND_LEADOCEAN_API_KEY}
    body      {<filters>, "limit": N, "cursor": <meta.nextCursor>}
    sizing    {<filters>, "count": true, "limit": 1}   -> meta.total, no rows, FREE
    response  {"success": true, "data": [<nested person record>], "meta": {...}}

**Not verified against a live response.** The request side and the response envelope are
the spec's. The record's inner field names (``profile_data.profile_url``, the current role
in ``contact_data.contact_current_experiences[0]``) are the spec's data dictionary; the
shape of an address object and what ``title`` does with several words are not in it. They
are read defensively and every assumption is in this file.

**Paging.** A page is a cursor, not an offset, and the spec says not to store one. The walk
asks for page n by offset, so a cursor chain is walked from the first page, 100 rows a slice
(the default maximum), and the cursors are kept for the life of the process only. A later
page of the same search in the same process resumes from the last cursor; a later *run*
starts the walk again, and a deep page then re-reads (and is billed for) the slices before
it. ``meta.depthCapped`` (about 10,000 rows) is the bottom of the search.

**Cost.** A page costs one record per person returned; the count costs nothing. So the
count is asked for first, on its own, and a search that cannot be answered is never bought.

**What it will not do.** ``unsupportedFilters`` means Lead Ocean dropped a filter, so the
search would come back broader than asked: the source answers such a search with no rows and
a total of 0, and says so loudly in the log, rather than buying people nobody asked for. A
seniority or country Lead Ocean has no value for is answered the same way. ``/v1/enrich`` is
never called, and a row with no profile URL is dropped.

**Refusals are typed, never silent.** 401/403 is ``provider_auth``, 402
``provider_out_of_credits``, 429 ``provider_rate_limited``. 400, 404, 405, 410 and 422 are
``bad_config``: a wrong path or a refused cursor must not read as an outage, which the walk
swallows and which would quietly make a combined search dataset-only.
"""
from __future__ import annotations

import json
import logging
import threading
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from openoutfind.core.config import SiteConfig
from openoutfind.core.errors import ErrorType, OpenOutFindError
from openoutfind.enrichment.provider import ProviderUnavailable

logger = logging.getLogger(__name__)

SEARCH_PATH = "/v1/people/search"
HTTP_TIMEOUT_S = 60
PAGE_LIMIT = 100  # the spec's default maximum; an account may be allowed 200, we never ask

_BAD_CONFIG_STATUSES = {400, 404, 405, 410, 422}

_RETRY = Retry(total=3, backoff_factor=2, status_forcelist=(429,), allowed_methods=None,
               respect_retry_after_header=True, raise_on_status=False)

# Cursors are a place in one walk, so they live for the process and nowhere else. Keyed on
# the exact body of the search. ``cursors[i]`` is the cursor that *starts* slice i (slice 0
# starts with none); ``counted`` says the free sizing call has been made.
_cursor_lock = threading.Lock()
_cursors: dict[str, dict] = {}


class LeadOceanUnavailable(ProviderUnavailable):
    """Lead Ocean could not answer: refused, out of records, rate-limited or unreachable."""


def search(config: SiteConfig, filters: dict, limit: int, offset: int):
    """One page from Lead Ocean; returns ``discovery.Fetched``.

    ``leads_found`` is the search's total at offset 0, like the provider it stands in for.
    """
    from openoutfind.discovery import Fetched, Page

    validate(config)
    started = time.monotonic()
    total = count(config, filters)
    body = _body_for(filters)
    rows: list[dict] = []
    if body is not None and total != 0:
        rows = _every_title_word(_page(config, body, limit, offset), filters)
        place = _country_name(filters)
        if place:
            for row in rows:
                row.setdefault("contact_location_country", place)
    return Fetched(Page(rows, total if offset == 0 else None), time.monotonic() - started)


def validate(config: SiteConfig) -> None:
    """The base URL must be ``https://`` (loopback excepted, for a local stand-in).

    The API key rides every request, so a plain ``http://`` address would send it in the
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


def count(config: SiteConfig, filters: dict) -> int | None:
    """How many people match, from the free sizing call (cached for the process).

    ``0`` for a search Lead Ocean cannot answer faithfully — no value for a filter, or a
    filter it would drop — with a warning that says which. ``None`` only when Lead Ocean
    gave no total for another reason, which the stream reads as an unknown end.
    """
    validate(config)
    body = _body_for(filters)
    if body is None:
        return 0
    state = _state(_key(body))
    with _cursor_lock:
        if state["counted"]:
            return state["total"]
    payload = _post(config, {**body, "count": True, "limit": 1})
    meta = payload.get("meta") or {}
    dropped = meta.get("unsupportedFilters") or []
    if dropped or meta.get("totalWithheld"):
        logger.warning(
            "lead ocean: cannot answer this search faithfully — it would drop %s "
            "(its total is withheld). No people are bought for it; narrow the employer side "
            "(country, headcount, industry) or fewer keywords.",
            ", ".join(str(f) for f in dropped) or "a filter")
        total = 0
    elif meta.get("total") is None:
        total = None
    else:
        total = int(meta["total"])
    with _cursor_lock:
        state["total"], state["counted"] = total, True
    return total


# ── paging ───────────────────────────────────────────────────────────────────

def _page(config: SiteConfig, body: dict, limit: int, offset: int) -> list[dict]:
    """The rows at ``offset..offset+limit``, walking the cursor chain.

    Resumes from the nearest slice whose cursor is in hand. Slices before the first one
    wanted are read only to move the cursor forward when no cursor for it is known; their
    rows are thrown away.
    """
    state = _state(_key(body))
    first, last = offset // PAGE_LIMIT, (offset + limit - 1) // PAGE_LIMIT
    with _cursor_lock:
        index = max(i for i in state["cursors"] if i <= first)

    rows: list[dict] = []
    while index <= last:
        with _cursor_lock:
            if index not in state["cursors"]:
                break  # the previous slice was the last one
            cursor = state["cursors"][index]
        payload = _post(config, {**body, "limit": PAGE_LIMIT, "cursor": cursor})
        meta = payload.get("meta") or {}
        _refuse_if_broader(meta)
        records = [r for r in payload["data"] if isinstance(r, dict)]
        if index >= first:
            base = index * PAGE_LIMIT
            wanted = records[max(0, offset - base):offset + limit - base]
            rows.extend(row for row in map(_row, wanted) if row)
        nxt = meta.get("nextCursor")
        if nxt and not meta.get("depthCapped"):
            with _cursor_lock:
                state["cursors"][index + 1] = nxt
        else:
            break
        if not records:
            break
        index += 1
    return rows


def _state(key: str) -> dict:
    with _cursor_lock:
        return _cursors.setdefault(key, {"cursors": {0: None}, "total": None, "counted": False})


def _refuse_if_broader(meta: dict) -> None:
    """The count said the search was answerable; a page that says otherwise is a surprise."""
    dropped = meta.get("unsupportedFilters") or []
    if dropped:
        raise OpenOutFindError(
            ErrorType.BAD_CONFIG,
            "Lead Ocean dropped " + ", ".join(str(f) for f in dropped)
            + " mid-search — the rows are broader than asked")


# ── the search body ──────────────────────────────────────────────────────────

# Lead Finder's seniority list less the one value Lead Ocean has no band for.
_NO_LEADOCEAN_BAND = {"mid-level"}


def _body_for(filters: dict) -> dict | None:
    """A Lead Finder filter set as a Lead Ocean people-search body.

    ``None`` when Lead Ocean cannot answer it: a seniority it has no band for, or a country
    it cannot name. That is "no people", not an error, and nothing is bought.

    **Title.** Lead Ocean matches a phrase per entry and ORs the entries, and Lead Finder's
    words are any-order and all required, with stopwords gone — "head of procurement" is the
    words ``head`` and ``procurement``, which no one phrase can express. So each word is its
    own entry, which is wider than asked, and ``search`` filters the rows back to those that
    hold every word (``_every_title_word``). The count is therefore an upper bound. The first
    live sizing call should check this reading of ``title``.

    **Place.** A country by ISO code. A state cannot be asked of Lead Ocean for the person
    (its state filter is the employer's headquarters, and is dropped for a broad one), so it
    is not sent: the search is the whole country.
    """
    body: dict = {}
    for key, spec in filters.items():
        values = _include(spec)
        if key == "lead_job_title" and values:
            words = sorted({word for value in values for word in value.split()})
            body["title"] = words
        elif key == "company_keywords" and values:
            body["companyKeywords"] = values
        elif key == "lead_seniority" and values:
            if any(v in _NO_LEADOCEAN_BAND for v in values):
                return None
            body["seniority"] = values
        elif key == "lead_location" and values:
            code = _country_code(values[0])
            if code is None:
                logger.warning("lead ocean: no country code for %r — search not run", values[0])
                return None
            body["country"] = [code]
        elif key == "company_headcount_min":
            body["minEmployees"] = int(spec)
        elif key == "company_headcount_max":
            body["maxEmployees"] = int(spec)
    return body


def _include(spec) -> list[str]:
    if isinstance(spec, dict):
        return [str(v) for v in spec.get("include", []) if str(v).strip()]
    return []


def _every_title_word(rows: list[dict], filters: dict) -> list[dict]:
    """Keep the rows whose job title holds every word the search asked for, as Lead Finder
    would have matched it (any order, stopwords dropped)."""
    from openoutfind.core.pipeline import analyzers

    wanted = {word for value in _include(filters.get("lead_job_title"))
              for word in analyzers.query_title_words(value)}
    if not wanted:
        return rows
    return [row for row in rows
            if wanted <= analyzers.title_words(row.get("contact_job_title") or "")]


def _country_name(filters: dict) -> str:
    values = _include(filters.get("lead_location"))
    return values[0].split(",")[-1].strip() if values else ""


def _country_code(place: str) -> str | None:
    """``"California, United States"`` or ``"United States"`` as ``"US"`` (ISO 3166-1 alpha-2)."""
    import pytz

    name = place.split(",")[-1].strip().lower()
    if len(name) == 2 and name.upper() in pytz.country_names:
        return name.upper()
    for code, full in pytz.country_names.items():
        if full.lower() == name:
            return code
    return None


def _key(body: dict) -> str:
    return json.dumps(body, sort_keys=True)


# ── the row ──────────────────────────────────────────────────────────────────

def _row(record: dict) -> dict | None:
    """One nested Lead Ocean person record as a Lead Finder row.

    The current role is ``contact_current_experiences[0]``. A record with no profile URL is
    dropped: the store keys a person on it. ``person_id`` is kept for a later enrichment,
    which this client never performs. An empty value is left out, never stored as ``""``.
    """
    profile = record.get("profile_data") or {}
    contact = record.get("contact_data") or {}
    experiences = contact.get("contact_current_experiences") or []
    role = experiences[0] if experiences and isinstance(experiences[0], dict) else {}
    url = str(profile.get("profile_url") or "").strip()
    if not url:
        return None

    seniority = role.get("job_seniority")
    employees = role.get("company_employees") or {}
    row: dict = {
        "contact_linkedin_profile_url": url,
        "contact_full_name": profile.get("profile_full_name"),
        "contact_first_name": profile.get("profile_first_name"),
        "contact_last_name": profile.get("profile_last_name"),
        "contact_headline": profile.get("profile_headline"),
        "contact_job_title": role.get("job_title"),
        "contact_seniority": seniority.get("job_level") if isinstance(seniority, dict) else None,
        "contact_industry": role.get("company_industry"),
        "company_name": role.get("company_name"),
        "company_domain": role.get("company_domain"),
        "company_industry": role.get("company_industry"),
        "company_employees_range_start": employees.get("min"),
        "company_employees_range_end": employees.get("max"),
        "person_id": ((record.get("meta") or {}).get("source_ids") or {}).get("person_id"),
    }
    address = profile.get("profile_address")
    if isinstance(address, dict):
        row["contact_location_city"] = address.get("city")
        row["contact_location_state"] = address.get("state") or address.get("region")
        row["contact_location_country"] = address.get("country")
    return {key: value for key, value in row.items() if value not in (None, "", [])}


# ── transport ────────────────────────────────────────────────────────────────

def _post(config: SiteConfig, body: dict) -> dict:
    url = config.leadocean_base_url.rstrip("/") + SEARCH_PATH
    sent = {key: value for key, value in body.items() if value is not None}
    with requests.Session() as session:
        session.headers["x-api-key"] = config.leadocean_api_key
        adapter = HTTPAdapter(max_retries=_RETRY)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        try:
            response = session.post(url, json=sent, timeout=HTTP_TIMEOUT_S)
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
        raise LeadOceanUnavailable("Lead Ocean records are exhausted (402)",
                                   ErrorType.PROVIDER_OUT_OF_CREDITS)
    if response.status_code == 429:
        raise LeadOceanUnavailable("Lead Ocean is rate-limiting this client (429)",
                                   ErrorType.PROVIDER_RATE_LIMITED)
    if response.status_code in _BAD_CONFIG_STATUSES:
        raise OpenOutFindError(
            ErrorType.BAD_CONFIG,
            f"Lead Ocean refused the search ({response.status_code}) at {SEARCH_PATH}")
    if not response.ok:
        raise LeadOceanUnavailable(f"Lead Ocean answered HTTP {response.status_code}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise LeadOceanUnavailable("Lead Ocean returned a body that is not JSON") from exc
    if not isinstance(payload, dict) or payload.get("success") is False:
        raise LeadOceanUnavailable("Lead Ocean returned an error envelope")
    if not isinstance(payload.get("data"), list):
        raise LeadOceanUnavailable("Lead Ocean returned no 'data' list")
    return payload
