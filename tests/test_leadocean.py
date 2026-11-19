# tests/test_leadocean.py
"""Lead Ocean's people search (hybrowlabs fork, OPS-636), and the stream that reads it beside
the dataset and Lead Finder.

Offline by construction: ``requests.Session`` is replaced by a fake that answers like the
vendor's public spec (``POST /v1/people/search``, ``x-api-key``, cursor paging, a free
``count=true&limit=1`` sizing call). Nothing here reaches a network, and no key is real.
The record shape is the spec's data dictionary, not a captured response.
"""
import csv
import logging
from unittest.mock import MagicMock, patch

import pytest
import requests

from openoutfind import providers
from openoutfind.core import readiness
from openoutfind.core.config import SiteConfig
from openoutfind.core.errors import ErrorType, OpenOutFindError
from openoutfind.enrichment.provider import ProviderUnavailable
from openoutfind.providers import leadocean

BASE = "https://ocean.example"


@pytest.fixture(autouse=True)
def _fresh_process_state():
    """Cursors and counts live for the process; a test must not inherit another's."""
    leadocean._cursors.clear()
    providers._totals.clear()
    yield
    leadocean._cursors.clear()
    providers._totals.clear()


def person(i, title="Procurement Manager", url=True, **role):
    record = {
        "profile_data": {"profile_full_name": f"P{i}", "profile_headline": f"{title} at Co{i}"},
        "contact_data": {"contact_current_experiences": [{
            "job_title": title, "company_name": f"Co{i}", "company_domain": f"co{i}.com",
            "company_industry": "Packaging", "job_seniority": {"job_level": "manager"},
            "company_employees": {"min": 11, "max": 50}, **role}]},
        "meta": {"source_ids": {"person_id": f"id{i}"}},
    }
    if url:
        record["profile_data"]["profile_url"] = f"https://li/lo{i}"
    return record


class FakeOcean:
    """The vendor's search as a test double: ``total`` people, ``slice_size`` a page, cursors
    ``c1``, ``c2`` ... and a depth cap at ``depth`` rows. ``calls`` records every body."""

    def __init__(self, total=1000, depth=10_000, meta=None, status=200, raises=None,
                 payload=None, **person_role):
        self.total, self.depth, self.meta = total, depth, meta or {}
        self.status, self.raises, self.payload = status, raises, payload
        self.role = person_role
        self.calls, self.headers, self.urls = [], [], []

    def __call__(self):
        session = MagicMock()
        session.__enter__.return_value = session
        headers = {}
        session.headers = headers
        session.post.side_effect = lambda url, json=None, timeout=None: self._post(
            url, json, headers)
        return session

    def pages(self):
        return [c for c in self.calls if not c.get("count")]

    def _post(self, url, body, headers):
        self.urls.append(url)
        self.headers.append(dict(headers))
        self.calls.append(body)
        if self.raises:
            raise self.raises
        response = MagicMock()
        response.status_code = self.status
        response.ok = self.status < 400
        if self.payload is not None:
            response.json.return_value = self.payload
            return response
        if body.get("count"):
            response.json.return_value = {"success": True, "data": [], "meta": {
                "total": self.total, "totalIsExact": True, "countOnly": True, "credits": 0,
                **self.meta}}
            return response
        index = int((body.get("cursor") or "c0")[1:])
        start = index * body["limit"]
        held = self.total if self.total is not None else 1000  # an uncounted search still pages
        stop = min(start + body["limit"], held, self.depth)
        capped = stop >= self.depth and held > self.depth
        more = stop < min(held, self.depth)
        response.json.return_value = {"success": True, "credits": stop - start,
            "data": [person(i, **self.role) for i in range(start, stop)],
            "meta": {"count": stop - start, "nextCursor": f"c{index + 1}" if more else None,
                     **({"depthCapped": True} if capped and not more else {}), **self.meta}}
        return response


def configured(configure, **extra):
    return configure(discovery_provider="leadocean", leadocean_api_key="lo-key",
                     leadocean_base_url=BASE, **extra)


def fetch(config, fake, filters=None, limit=200, offset=0):
    with patch("requests.Session", side_effect=lambda: fake()):
        return leadocean.search(config, filters or {}, limit, offset).page


class TestTheRequestIsTheVendors:
    def test_it_posts_to_the_people_search_with_an_api_key_header(self, configure):
        config, fake = configured(configure), FakeOcean(total=3)

        fetch(config, fake, limit=3)

        assert set(fake.urls) == {f"{BASE}/v1/people/search"}
        assert all(h["x-api-key"] == "lo-key" for h in fake.headers)
        assert not any("Authorization" in h for h in fake.headers)

    def test_sizing_is_the_free_count_call_and_a_page_never_asks_for_it(self, configure):
        config, fake = configured(configure), FakeOcean(total=250)

        page = fetch(config, fake, limit=100)

        sizing, bought = fake.calls[0], fake.pages()[0]
        assert sizing["count"] is True and sizing["limit"] == 1
        assert "count" not in bought and bought["limit"] == 100 and "offset" not in bought
        assert page.leads_found == 250

    def test_the_count_is_asked_once_per_search_however_many_pages_follow(self, configure):
        config, fake = configured(configure), FakeOcean(total=1000)

        fetch(config, fake, limit=100, offset=0)
        fetch(config, fake, limit=100, offset=100)

        assert len([c for c in fake.calls if c.get("count")]) == 1

    def test_the_count_is_withheld_past_offset_zero(self, configure):
        config, fake = configured(configure), FakeOcean(total=300)

        assert fetch(config, fake, limit=100, offset=100).leads_found is None

    def test_enrichment_is_never_called(self, configure):
        config, fake = configured(configure), FakeOcean(total=5)

        fetch(config, fake, limit=5)

        assert all(url.endswith("/v1/people/search") for url in fake.urls)


class TestFiltersAreTranslated:
    def body(self, configure, filters):
        return leadocean._body_for(filters)

    def test_title_words_are_separate_entries_and_the_rest_map_across(self, configure):
        filters = {
            "company_headcount_min": 1, "company_headcount_max": 10000,
            "lead_job_title": {"include": ["head procurement"], "exact_match": False},
            "company_keywords": {"include": ["cannabis extraction"], "exact_match": False},
            "lead_location": {"include": ["California, United States"]},
            "lead_seniority": {"include": ["head"]},
        }

        body = leadocean._body_for(filters)

        assert body == {"minEmployees": 1, "maxEmployees": 10000,
                        "title": ["head", "procurement"],
                        "companyKeywords": ["cannabis extraction"],
                        "country": ["US"],  # the state cannot be asked of the person
                        "seniority": ["head"]}
        assert "order" not in body

    def test_a_seniority_lead_ocean_has_no_band_for_is_no_people_and_buys_nothing(
            self, configure):
        config, fake = configured(configure), FakeOcean(total=99)

        page = fetch(config, fake, {"lead_seniority": {"include": ["mid-level"]}})

        assert page.leads == [] and page.leads_found == 0
        assert fake.calls == []

    def test_a_country_it_cannot_name_is_no_people_and_says_so(self, configure, caplog):
        config, fake = configured(configure), FakeOcean(total=99)

        with caplog.at_level(logging.WARNING):
            page = fetch(config, fake, {"lead_location": {"include": ["Atlantis"]}})

        assert page.leads == [] and fake.calls == []
        assert any("Atlantis" in r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize("place, code", [("United States", "US"), ("Texas, United States", "US"),
                                             ("india", "IN"), ("DE", "DE")])
    def test_places_become_iso_country_codes(self, place, code):
        assert leadocean._country_code(place) == code

    def test_the_title_filter_keeps_only_people_whose_title_holds_every_word(self):
        """Lead Ocean ORs the words; Lead Finder required all of them."""
        rows = [leadocean._row(person(i, title=t)) for i, t in enumerate(
            ["Head of Procurement", "Head of Sales", "Procurement Analyst"])]

        kept = leadocean._every_title_word(
            rows, {"lead_job_title": {"include": ["head procurement"]}})

        assert [r["contact_job_title"] for r in kept] == ["Head of Procurement"]

    def test_a_search_with_no_title_is_not_filtered(self):
        rows = [leadocean._row(person(1))]

        assert leadocean._every_title_word(rows, {}) == rows


class TestTheRowIsMapped:
    def test_a_nested_record_becomes_a_lead_finder_row(self):
        row = leadocean._row(person(7, office_address=None))

        assert row["contact_linkedin_profile_url"] == "https://li/lo7"
        assert row["contact_full_name"] == "P7"
        assert row["contact_headline"] == "Procurement Manager at Co7"
        assert row["contact_job_title"] == "Procurement Manager"
        assert row["contact_seniority"] == "manager"
        assert row["company_name"] == "Co7" and row["company_domain"] == "co7.com"
        assert row["company_employees_range_start"] == 11
        assert row["company_employees_range_end"] == 50
        assert row["person_id"] == "id7"

    def test_a_record_with_no_profile_url_is_dropped(self):
        assert leadocean._row(person(1, url=False)) is None

    def test_empty_values_are_left_out_not_stored_as_blanks(self):
        row = leadocean._row({"profile_data": {"profile_url": "https://li/x",
                                               "profile_full_name": ""}})

        assert row == {"contact_linkedin_profile_url": "https://li/x"}

    def test_the_searched_country_is_stamped_on_rows_that_carry_none(self, configure):
        config, fake = configured(configure), FakeOcean(total=1)

        page = fetch(config, fake, {"lead_location": {"include": ["Texas, United States"]}},
                     limit=1)

        assert page.leads[0]["contact_location_country"] == "United States"


class TestPagingIsACursorChain:
    def test_a_walk_page_of_200_is_two_slices_of_100(self, configure):
        config, fake = configured(configure), FakeOcean(total=1000)

        page = fetch(config, fake, limit=200, offset=0)

        assert [c.get("cursor") for c in fake.pages()] == [None, "c1"]
        assert all(c["limit"] == 100 for c in fake.pages())
        assert len(page.leads) == 200

    def test_the_next_page_resumes_from_the_cursor_and_does_not_re_read_the_first(
            self, configure):
        config, fake = configured(configure), FakeOcean(total=1000)

        fetch(config, fake, limit=200, offset=0)
        fake.calls.clear()
        second = fetch(config, fake, limit=200, offset=200)

        assert [c["cursor"] for c in fake.pages()] == ["c2", "c3"]
        assert second.leads[0]["contact_full_name"] == "P200"
        assert second.leads[-1]["contact_full_name"] == "P399"

    def test_a_deep_page_in_a_fresh_process_walks_the_chain_from_the_start(self, configure):
        """Cursors are not stored (the spec refuses a stale one), so a new process pays to
        reach a deep page again. This is the documented cost."""
        config, fake = configured(configure), FakeOcean(total=1000)

        page = fetch(config, fake, limit=100, offset=300)

        assert [c.get("cursor") for c in fake.pages()] == [None, "c1", "c2", "c3"]
        assert page.leads[0]["contact_full_name"] == "P300"
        assert len(page.leads) == 100

    def test_the_depth_cap_is_the_bottom_and_no_cursor_is_chased_past_it(self, configure):
        config, fake = configured(configure), FakeOcean(total=5000, depth=300)

        rows = fetch(config, fake, limit=200, offset=200).leads
        beyond = fetch(config, fake, limit=200, offset=400).leads

        assert len(rows) == 100  # rows 200..299, then the cap
        assert beyond == []

    def test_the_last_short_page_ends_the_search(self, configure):
        config, fake = configured(configure), FakeOcean(total=130)

        page = fetch(config, fake, limit=200, offset=0)

        assert len(page.leads) == 130
        assert len(fake.pages()) == 2


class TestAnUnanswerableSearchIsNotBought:
    def test_dropped_filters_withhold_the_total_and_no_person_is_bought(self, configure, caplog):
        config = configured(configure)
        fake = FakeOcean(total=None, meta={"total": None, "unsupportedFilters": ["companyKeywords"],
                         "totalWithheld": {"reason": "unsupported_filters",
                                           "filters": ["companyKeywords"]}})

        with caplog.at_level(logging.WARNING):
            page = fetch(config, fake, {"company_keywords": {"include": ["packaging"]}})

        assert page.leads == [] and page.leads_found == 0
        assert fake.pages() == []  # only the free sizing call was made
        assert any("companyKeywords" in r.getMessage() for r in caplog.records)

    def test_a_filter_dropped_only_once_pages_arrive_stops_the_run(self, configure):
        """The count said the search was answerable; a page that says otherwise is a surprise."""
        config = configured(configure)
        fake = FakeOcean(total=500)

        def post(url, body, headers):
            response = FakeOcean._post(fake, url, body, headers)
            if not body.get("count"):
                response.json.return_value["meta"]["unsupportedFilters"] = ["hqCity"]
            return response

        fake._post = post
        with patch("requests.Session", side_effect=lambda: fake()):
            with pytest.raises(OpenOutFindError) as raised:
                leadocean.search(config, {}, 100, 0)

        assert raised.value.error_type == ErrorType.BAD_CONFIG
        assert "hqCity" in str(raised.value)

    def test_a_total_that_is_simply_absent_is_unknown_not_zero(self, configure):
        config = configured(configure)
        fake = FakeOcean(total=None)

        assert fetch(config, fake, limit=1).leads_found is None


class TestRefusalsAreTyped:
    @pytest.mark.parametrize("status, error_type", [
        (401, ErrorType.PROVIDER_AUTH), (403, ErrorType.PROVIDER_AUTH),
        (402, ErrorType.PROVIDER_OUT_OF_CREDITS), (429, ErrorType.PROVIDER_RATE_LIMITED),
        (500, ErrorType.PROVIDER_UNAVAILABLE), (503, ErrorType.PROVIDER_UNAVAILABLE)])
    def test_a_refusal_never_reads_as_an_empty_search(self, configure, status, error_type):
        config = configured(configure)

        with patch("requests.Session", side_effect=lambda: FakeOcean(status=status)()):
            with pytest.raises(ProviderUnavailable) as raised:
                leadocean.search(config, {}, 10, 0)

        assert raised.value.error_type == error_type

    @pytest.mark.parametrize("status", [400, 404, 405, 410, 422])
    def test_a_wrong_path_or_a_refused_cursor_is_bad_config_not_an_outage(
            self, configure, status):
        """An outage is swallowed by the walk, which would quietly make a combined search
        dataset-only. A mistake in the address must stop the run."""
        config = configured(configure)

        with patch("requests.Session", side_effect=lambda: FakeOcean(status=status)()):
            with pytest.raises(OpenOutFindError) as raised:
                leadocean.search(config, {}, 10, 0)

        assert raised.value.error_type == ErrorType.BAD_CONFIG
        assert str(status) in str(raised.value)

    def test_an_unreachable_service_is_provider_unavailable(self, configure):
        config = configured(configure)
        fake = FakeOcean(raises=requests.ConnectionError("down"))

        with patch("requests.Session", side_effect=lambda: fake()):
            with pytest.raises(ProviderUnavailable) as raised:
                leadocean.search(config, {}, 10, 0)

        assert raised.value.error_type == ErrorType.PROVIDER_UNAVAILABLE

    @pytest.mark.parametrize("payload", [["not", "an", "object"], {"success": False},
                                         {"success": True}, {"success": True, "data": "x"}])
    def test_a_body_that_is_not_the_envelope_is_an_error_not_silence(self, configure, payload):
        config = configured(configure)

        with patch("requests.Session", side_effect=lambda: FakeOcean(payload=payload)()):
            with pytest.raises(ProviderUnavailable):
                leadocean.search(config, {}, 10, 0)

    @pytest.mark.parametrize("url", ["http://ocean.example", "ocean.example", "ftp://x", ""])
    def test_a_base_url_that_is_not_https_is_refused_before_any_call(self, configure, url):
        config = configure(discovery_provider="leadocean", leadocean_api_key="k",
                           leadocean_base_url=url)

        with patch("requests.Session") as session:
            with pytest.raises(OpenOutFindError) as raised:
                leadocean.search(config, {}, 10, 0)

        assert raised.value.error_type == ErrorType.BAD_CONFIG
        session.assert_not_called()

    def test_a_loopback_http_url_is_allowed_for_a_local_stand_in(self, configure):
        config = configure(discovery_provider="leadocean", leadocean_api_key="k",
                           leadocean_base_url="http://localhost:8080")
        fake = FakeOcean(total=1)

        fetch(config, fake, limit=1)

        assert set(fake.urls) == {"http://localhost:8080/v1/people/search"}

    def test_readiness_rejects_a_plain_http_base_url_before_any_round(self, configure):
        configure(discovery_provider="leadocean", leadocean_api_key="k",
                  leadocean_base_url="http://ocean.example", product_docs="p",
                  campaign_target="t", ai_model="openai:gpt-4o", llm_api_key="k")

        with pytest.raises(OpenOutFindError) as raised:
            readiness.check_ready()

        assert raised.value.error_type == ErrorType.BAD_CONFIG

    def test_a_half_configured_lead_ocean_is_named_when_the_dataset_carries_the_run(
            self, configure, tmp_path, caplog, operator):
        path = tmp_path / "p.csv"
        path.write_text("linkedin_url,full_name,job_title,company_name\nhttps://li/a,A,CTO,Co\n",
                        encoding="utf-8")
        configure(discovery_provider="dataset+leadocean", dataset_path=str(path),
                  leadocean_api_key="k", product_docs="p", campaign_target="t",
                  ai_model="openai:gpt-4o", llm_api_key="k")

        with caplog.at_level("WARNING"):
            readiness.check_ready()

        assert any("half configured" in r.getMessage() for r in caplog.records)


HEADER = ["linkedin_url", "full_name", "job_title", "company_name"]


def write_people(tmp_path, n):
    path = tmp_path / "big.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        for i in range(n):
            writer.writerow([f"https://li/p{i}", f"D{i}", "Procurement Manager", "Co"])
    return path


class TestTheStream:
    """The dataset, then Lead Ocean, then Lead Finder, as one stream paged by offset."""

    def config(self, configure, tmp_path, dataset=250, lead_finder=False, **extra):
        values = dict(discovery_provider="dataset+leadocean" + ("+leadfinder" if lead_finder
                                                                  else ""),
                      dataset_path=str(write_people(tmp_path, dataset)),
                      leadocean_api_key="lo-key", leadocean_base_url=BASE, **extra)
        if lead_finder:
            values["bettercontact_api_key"] = "bc"
        return configure(**values)

    def search(self, config, fake, limit, offset):
        with patch("requests.Session", side_effect=lambda: fake()):
            return providers.search(config, {}, limit=limit, offset=offset, first_poll_s=1).page

    def test_a_first_page_the_dataset_fills_still_reports_lead_oceans_count_and_buys_nobody(
            self, configure, tmp_path):
        """Reporting only the dataset's count told the walk the search ended where the
        dataset did, and Lead Ocean was never paged past one more page."""
        config, fake = self.config(configure, tmp_path), FakeOcean(total=5000)

        page = self.search(config, fake, 200, 0)

        assert len(page.leads) == 200
        assert page.leads_found == 250 + 5000
        assert fake.pages() == []  # the count is free; nobody was bought

    def test_the_stream_continues_into_lead_ocean_where_the_dataset_ends(
            self, configure, tmp_path):
        config, fake = self.config(configure, tmp_path), FakeOcean(total=5000)

        second = self.search(config, fake, 200, 200)

        names = [r["contact_full_name"] for r in second.leads]
        assert names[:50] == [f"D{i}" for i in range(200, 250)]
        assert names[50:53] == ["P0", "P1", "P2"] and len(names) == 200
        assert [c.get("cursor") for c in fake.pages()] == [None, "c1"]

    def test_a_later_page_resumes_lead_oceans_cursor_and_never_re_buys_a_slice(
            self, configure, tmp_path):
        config, fake = self.config(configure, tmp_path), FakeOcean(total=5000)

        self.search(config, fake, 200, 200)
        fake.calls.clear()
        third = self.search(config, fake, 200, 400)

        # 150 rows into Lead Ocean: slice 1 (cursor c1, already in hand), 2 and 3
        assert [c["cursor"] for c in fake.pages()] == ["c1", "c2", "c3"]
        assert third.leads[0]["contact_full_name"] == "P150"

    def test_a_page_wholly_inside_the_dataset_buys_nobody(self, configure, tmp_path):
        config, fake = self.config(configure, tmp_path), FakeOcean(total=5000)

        self.search(config, fake, 100, 100)

        assert fake.pages() == []

    def test_a_dropped_filter_at_lead_ocean_leaves_the_dataset_to_carry_the_search(
            self, configure, tmp_path, caplog):
        config = self.config(configure, tmp_path, dataset=3)
        fake = FakeOcean(total=None, meta={"total": None, "unsupportedFilters": ["hqCity"],
                                           "totalWithheld": {"reason": "unsupported_filters"}})

        with caplog.at_level(logging.WARNING):
            page = self.search(config, fake, 10, 0)

        assert [r["contact_full_name"] for r in page.leads] == ["D0", "D1", "D2"]
        assert page.leads_found == 3
        assert any("hqCity" in r.getMessage() for r in caplog.records)

    def test_an_unknown_total_ends_the_stream_at_that_source(self, configure, tmp_path):
        """Nothing behind a source that will not say how many it holds can be placed."""
        config = self.config(configure, tmp_path, dataset=3, lead_finder=True)
        fake = FakeOcean(total=None)

        with patch("openoutfind.enrichment.bettercontact.search_and_poll") as finder:
            page = self.search(config, fake, 10, 0)

        assert page.leads_found is None
        finder.assert_not_called()

    def test_lead_finder_is_read_after_the_others_when_they_run_out(self, configure, tmp_path):
        config = self.config(configure, tmp_path, dataset=2, lead_finder=True)
        fake = FakeOcean(total=1)
        finder_rows = [{"contact_full_name": "LF", "contact_linkedin_profile_url": "https://li/lf"}]

        def search_and_poll(api_key, url, body, first_poll):
            rows = finder_rows if body["limit"] != 1 or body["offset"] else []
            return {"leads": rows, "summary": {"leads_found": 4}}, 0.1

        with patch("openoutfind.enrichment.bettercontact.search_and_poll",
                   side_effect=search_and_poll) as finder:
            page = self.search(config, fake, 10, 0)

        names = [r["contact_full_name"] for r in page.leads]
        assert names[:2] == ["D0", "D1"] and names[2] == "P0" and names[-1] == "LF"
        assert page.leads_found == 2 + 1 + 4
        assert finder.call_args_list[0].args[0] == "bc"

    def test_lead_finders_count_is_asked_once_per_search(self, configure, tmp_path):
        config = self.config(configure, tmp_path, dataset=2, lead_finder=True)
        fake = FakeOcean(total=1)
        body = {"leads": [], "summary": {"leads_found": 4}}

        with patch("openoutfind.enrichment.bettercontact.search_and_poll",
                   return_value=(body, 0.1)) as finder:
            self.search(config, fake, 1, 0)
            self.search(config, fake, 1, 1)
            counts = [c for c in finder.call_args_list if c.args[2]["limit"] == 1]

        assert len(counts) <= 2  # one count, plus at most the page that is one row long

    def test_a_lead_ocean_outage_stops_the_search_and_never_falls_back(
            self, configure, tmp_path):
        config = self.config(configure, tmp_path, dataset=2, lead_finder=True)
        fake = FakeOcean(raises=requests.ConnectionError("down"))

        with patch("openoutfind.enrichment.bettercontact.search_and_poll") as finder:
            with patch("requests.Session", side_effect=lambda: fake()):
                with pytest.raises(ProviderUnavailable):
                    providers.search(config, {}, limit=50, offset=0, first_poll_s=1)

        finder.assert_not_called()

    def test_the_dataset_alone_satisfies_the_default(self, configure, tmp_path):
        assert providers.is_configured(configure(
            dataset_path=str(write_people(tmp_path, 1)), discovery_provider="dataset+leadocean"))

    def test_lead_ocean_alone_satisfies_the_default(self, configure):
        assert providers.is_configured(configure(
            leadocean_api_key="k", leadocean_base_url="https://x",
            discovery_provider="dataset+leadocean"))

    def test_neither_does_not(self, configure):
        assert not providers.is_configured(configure(discovery_provider="dataset+leadocean"))


class TestTheWalkReadsAnUncountedEmptySearchAsNothing:
    """Fails on the first version: no rows and no count marked the search dead for good."""

    def test_an_empty_first_page_with_no_count_leaves_no_dead_search(self, db):
        from types import SimpleNamespace

        from openoutfind.core.models import Search
        from openoutfind.core.pipeline import walk
        from openoutfind.discovery import Fetched, Page

        search = Search.objects.create(key="k", terms={}, move="written")
        arm = SimpleNamespace(number=1, search=search, terms=SimpleNamespace(
            to_filters=lambda: {}, key="k"), move="written", count=0, theta=0.1)

        taken = walk._take_in(arm, Fetched(Page([], None), 0.1), 0)

        assert taken is None
        search.refresh_from_db()
        assert search.dead is False

    def test_an_empty_first_page_with_a_zero_count_is_still_dead(self, db):
        from types import SimpleNamespace

        from openoutfind.core.models import Search
        from openoutfind.core.pipeline import walk
        from openoutfind.discovery import Fetched, Page

        search = Search.objects.create(key="k2", terms={}, move="written")
        arm = SimpleNamespace(number=1, search=search, terms=SimpleNamespace(
            to_filters=lambda: {}, key="k2"), move="written", count=0, theta=0.1)

        assert walk._take_in(arm, Fetched(Page([], 0), 0.1), 0) is None
        search.refresh_from_db()
        assert search.dead is True
