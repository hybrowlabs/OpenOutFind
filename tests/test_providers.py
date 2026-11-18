# tests/test_providers.py
"""Discovery providers (hybrowlabs fork, OPS-636): Lead Ocean, the dataset file, both
together, and upstream's BetterContact still reachable.

Offline by construction: Lead Ocean's HTTP session and BetterContact's transport are
mocked, the dataset is a file under ``tmp_path``, and nothing here reaches a network.
"""
import csv
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import requests

from openoutfind import providers
from openoutfind.contacts import service as hub
from openoutfind.core import readiness
from openoutfind.core.errors import ErrorType, OpenOutFindError
from openoutfind.core.pipeline import discover, walk
from openoutfind.core.pipeline.terms import Terms
from openoutfind.enrichment.provider import ProviderUnavailable
from openoutfind.providers import dataset, leadocean

HEADER = ["linkedin_url", "full_name", "job_title", "company_name", "company_keywords",
          "seniority", "state", "country", "employees_min", "employees_max"]

PEOPLE = [
    ["https://li/ann", "Ann", "Head of Procurement", "GreenLeaf", "cannabis extraction; packaging",
     "head", "California", "United States", "11", "50"],
    ["https://li/bob", "Bob", "Procurement Manager", "Hemp Co", "hemp; cannabis",
     "manager", "Texas", "United States", "51", "200"],
    ["https://li/cy", "Cy", "Founder", "Bakery", "bakery", "founder", "", "India", "", ""],
    ["https://li/ann", "Ann again", "Duplicate", "GreenLeaf", "", "", "", "", "", ""],
    ["", "No Url", "Ghost", "Nowhere", "", "", "", "", "", ""],
]


@pytest.fixture
def people_csv(tmp_path):
    path = tmp_path / "people.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerows(PEOPLE)
    return path


@pytest.fixture
def on_dataset(configure, people_csv):
    return configure(discovery_provider="dataset", dataset_path=str(people_csv))


def _ocean_config(configure, **extra):
    return configure(discovery_provider="leadocean", leadocean_api_key="lo-key",
                     leadocean_base_url="https://ocean.example/api/", **extra)


def _response(status=200, payload=None, json_error=False):
    response = MagicMock()
    response.status_code = status
    response.ok = status < 400
    if json_error:
        response.json.side_effect = ValueError("not json")
    else:
        response.json.return_value = payload
    return response


def _session(response=None, raises=None):
    """A stand-in for ``requests.Session`` whose ``post`` is the only call made."""
    session = MagicMock()
    session.__enter__.return_value = session
    session.headers = {}
    if raises:
        session.post.side_effect = raises
    else:
        session.post.return_value = response
    return session


class TestTheDatasetSource:
    def test_rows_arrive_in_lead_finder_shape_and_are_deduplicated(self, on_dataset):
        rows = dataset.rows(on_dataset)

        assert [r["contact_linkedin_profile_url"] for r in rows] == [
            "https://li/ann", "https://li/bob", "https://li/cy"]
        assert rows[0]["contact_full_name"] == "Ann"
        assert rows[0]["company_keywords"] == ["cannabis extraction", "packaging"]
        assert rows[0]["company_employees_range_start"] == 11

    def test_a_missing_headline_is_written_so_the_live_profile_gate_keeps_the_row(
            self, on_dataset):
        from openoutfind.discovery import is_live_profile

        row = dataset.rows(on_dataset)[0]

        assert row["contact_headline"] == "Head of Procurement at GreenLeaf"
        assert is_live_profile(row)

    @pytest.mark.parametrize("filters, expected", [
        ({"lead_job_title": {"include": ["procurement"]}}, {"Ann", "Bob"}),
        ({"lead_job_title": {"include": ["procurement head"]}}, {"Ann"}),
        ({"company_keywords": {"include": ["cannabis extraction"]}}, {"Ann"}),
        ({"company_keywords": {"include": ["cannabis"]}}, {"Ann", "Bob"}),
        ({"lead_location": {"include": ["California, United States"]}}, {"Ann"}),
        ({"lead_location": {"include": ["United States"]}}, {"Ann", "Bob"}),
        ({"lead_seniority": {"include": ["founder"]}}, {"Cy"}),
        # Cy has no size, so a band keeps him rather than dropping him.
        ({"company_headcount_min": 100, "company_headcount_max": 500}, {"Bob", "Cy"}),
        ({}, {"Ann", "Bob", "Cy"}),
    ])
    def test_filters_follow_the_providers_matching_rules(self, on_dataset, filters, expected):
        page = dataset.search(on_dataset, filters, limit=50, offset=0).page

        assert {row["contact_full_name"] for row in page.leads} == expected
        assert page.leads_found == len(expected)

    def test_paging_and_the_count_only_at_offset_zero(self, on_dataset):
        first = dataset.search(on_dataset, {}, limit=2, offset=0).page
        second = dataset.search(on_dataset, {}, limit=2, offset=2).page

        assert [r["contact_full_name"] for r in first.leads] == ["Ann", "Bob"]
        assert [r["contact_full_name"] for r in second.leads] == ["Cy"]
        assert first.leads_found == 3 and second.leads_found is None

    def test_a_dataset_missing_a_required_column_is_bad_config(self, configure, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("linkedin_url,full_name\nhttps://li/x,X\n", encoding="utf-8")
        config = configure(discovery_provider="dataset", dataset_path=str(path))

        with pytest.raises(OpenOutFindError) as raised:
            dataset.rows(config)

        assert raised.value.error_type == ErrorType.BAD_CONFIG
        assert "job_title" in str(raised.value) and "company_name" in str(raised.value)

    def test_a_missing_file_is_bad_config_not_an_empty_search(self, configure, tmp_path):
        config = configure(discovery_provider="dataset", dataset_path=str(tmp_path / "no.csv"))

        with pytest.raises(OpenOutFindError) as raised:
            dataset.search(config, {}, 10, 0)

        assert raised.value.error_type == ErrorType.BAD_CONFIG

    def test_jsonl_is_read_too(self, configure, tmp_path):
        path = tmp_path / "people.jsonl"
        path.write_text(json.dumps({
            "linkedin_url": "https://li/dee", "full_name": "Dee", "job_title": "CTO",
            "company_name": "Acme", "company_keywords": ["saas", "b2b"]}) + "\n",
            encoding="utf-8")
        config = configure(discovery_provider="dataset", dataset_path=str(path))

        row = dataset.rows(config)[0]

        assert row["contact_full_name"] == "Dee"
        assert row["company_keywords"] == ["saas", "b2b"]

    def test_readiness_rejects_an_unreadable_dataset_before_any_round(self, configure, tmp_path):
        configure(discovery_provider="dataset", dataset_path=str(tmp_path / "gone.csv"),
                  product_docs="p", campaign_target="t", ai_model="openai:gpt-4o",
                  llm_api_key="k")

        with pytest.raises(OpenOutFindError) as raised:
            readiness.check_ready()

        assert raised.value.error_type == ErrorType.BAD_CONFIG


class TestTheLeadOceanSource:
    def test_it_posts_the_filters_with_a_bearer_key_and_reads_rows(self, configure):
        config = _ocean_config(configure)
        rows = [{"contact_full_name": "Eve", "contact_linkedin_profile_url": "https://li/eve"}]
        session = _session(_response(payload={"leads": rows, "summary": {"leads_found": 42}}))

        with patch("requests.Session", return_value=session):
            page = leadocean.search(config, {"lead_seniority": {"include": ["owner"]}},
                                    limit=200, offset=0).page

        url = session.post.call_args.args[0]
        body = session.post.call_args.kwargs["json"]
        assert url == "https://ocean.example/api/leads/search"
        assert session.headers["Authorization"] == "Bearer lo-key"
        assert body == {"filters": {"lead_seniority": {"include": ["owner"]}},
                        "limit": 200, "offset": 0}
        assert page.leads == rows and page.leads_found == 42

    def test_the_count_is_withheld_past_offset_zero(self, configure):
        config = _ocean_config(configure)
        session = _session(_response(payload={"leads": [], "summary": {"leads_found": 0}}))

        with patch("requests.Session", return_value=session):
            page = leadocean.search(config, {}, 200, 400).page

        assert page.leads == [] and page.leads_found is None

    @pytest.mark.parametrize("status, error_type", [
        (401, ErrorType.PROVIDER_AUTH),
        (403, ErrorType.PROVIDER_AUTH),
        (402, ErrorType.PROVIDER_OUT_OF_CREDITS),
        (429, ErrorType.PROVIDER_RATE_LIMITED),
        (500, ErrorType.PROVIDER_UNAVAILABLE),
    ])
    def test_a_refusal_is_typed_and_never_an_empty_page(self, configure, status, error_type):
        config = _ocean_config(configure)

        with patch("requests.Session", return_value=_session(_response(status))):
            with pytest.raises(ProviderUnavailable) as raised:
                leadocean.search(config, {}, 10, 0)

        assert raised.value.error_type == error_type

    def test_an_unreachable_service_is_provider_unavailable(self, configure):
        config = _ocean_config(configure)
        session = _session(raises=requests.ConnectionError("boom"))

        with patch("requests.Session", return_value=session):
            with pytest.raises(ProviderUnavailable) as raised:
                leadocean.search(config, {}, 10, 0)

        assert raised.value.error_type == ErrorType.PROVIDER_UNAVAILABLE

    @pytest.mark.parametrize("response", [
        _response(payload=["not", "an", "object"]),
        _response(payload={"no_leads_key": True}),
        _response(json_error=True),
    ])
    def test_a_body_that_is_not_the_contract_is_an_error_not_silence(self, configure, response):
        config = _ocean_config(configure)

        with patch("requests.Session", return_value=_session(response)):
            with pytest.raises(ProviderUnavailable):
                leadocean.search(config, {}, 10, 0)

    def test_a_row_with_no_profile_url_is_dropped_not_stored_under_none(self, configure):
        config = _ocean_config(configure)
        rows = [{"contact_full_name": "Anon"},
                {"contact_full_name": "Hal", "contact_linkedin_profile_url": "https://li/hal"}]
        session = _session(_response(payload={"leads": rows}))

        with patch("requests.Session", return_value=session):
            page = leadocean.search(config, {}, 10, 0).page

        assert [r["contact_full_name"] for r in page.leads] == ["Hal"]

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
        session = _session(_response(payload={"leads": []}))

        with patch("requests.Session", return_value=session):
            leadocean.search(config, {}, 10, 0)

        assert session.post.call_args.args[0] == "http://localhost:8080/leads/search"

    def test_readiness_rejects_a_plain_http_base_url_before_any_round(self, configure):
        configure(discovery_provider="leadocean", leadocean_api_key="k",
                  leadocean_base_url="http://ocean.example", product_docs="p",
                  campaign_target="t", ai_model="openai:gpt-4o", llm_api_key="k")

        with pytest.raises(OpenOutFindError) as raised:
            readiness.check_ready()

        assert raised.value.error_type == ErrorType.BAD_CONFIG

    def test_a_half_configured_lead_ocean_is_named_when_the_dataset_carries_the_run(
            self, configure, people_csv, caplog, operator):
        configure(discovery_provider="leadocean+dataset", dataset_path=str(people_csv),
                  leadocean_api_key="k", product_docs="p", campaign_target="t",
                  ai_model="openai:gpt-4o", llm_api_key="k")

        with caplog.at_level("WARNING"):
            readiness.check_ready()

        assert any("half configured" in r.getMessage() for r in caplog.records)

    def test_the_field_map_renames_lead_ocean_columns(self, configure, monkeypatch):
        config = _ocean_config(configure)
        monkeypatch.setitem(leadocean.FIELD_MAP, "name", "contact_full_name")
        session = _session(_response(payload={"leads": [{"name": "Fay", "contact_linkedin_profile_url": "https://li/fay"}]}))

        with patch("requests.Session", return_value=session):
            page = leadocean.search(config, {}, 10, 0).page

        assert page.leads == [{"contact_full_name": "Fay",
                              "contact_linkedin_profile_url": "https://li/fay"}]


class TestBothTogether:
    def _both(self, configure, people_csv):
        return configure(discovery_provider="leadocean+dataset", dataset_path=str(people_csv),
                         leadocean_api_key="lo-key",
                         leadocean_base_url="https://ocean.example")

    def test_a_later_page_the_dataset_fills_makes_no_network_call(self, configure, people_csv):
        config = self._both(configure, people_csv)

        with patch("requests.Session") as session:
            page = providers.search(config, {}, limit=2, offset=1, first_poll_s=1).page

        assert [r["contact_full_name"] for r in page.leads] == ["Bob", "Cy"]
        session.assert_not_called()

    def test_a_first_page_the_dataset_fills_still_asks_lead_ocean_for_its_count(
            self, configure, tmp_path):
        """Fails on the first version, which reported only the dataset's count when the
        dataset filled page one, so the walk read the search as ending where the dataset did
        and never paged Lead Ocean past one more page."""
        config = self._big_dataset(configure, tmp_path, matching=250)
        session = _session(_response(payload={"leads": [], "summary": {"leads_found": 5000}}))

        with patch("requests.Session", return_value=session):
            first = providers.search(config, {}, limit=200, offset=0, first_poll_s=1).page

        assert len(first.leads) == 200
        asked = session.post.call_args.kwargs["json"]
        assert asked["limit"] == 1 and asked["offset"] == 0  # one row, for the count
        assert first.leads_found == 250 + 5000

    def test_lead_ocean_is_paged_to_its_depth_behind_a_large_dataset(self, configure, tmp_path):
        """250 dataset matches and 5000 at Lead Ocean: page 2 takes the dataset's last 50
        and Lead Ocean's first 150; page 3 starts 150 into Lead Ocean — and the count the
        walk stored on page 1 does not call page 3 the bottom."""
        from openoutfind.core.models import PAGE_SIZE

        config = self._big_dataset(configure, tmp_path, matching=250)
        ocean = [{"contact_full_name": f"LO{i}", "contact_linkedin_profile_url": f"https://li/lo{i}"}
                 for i in range(200)]
        session = _session(_response(payload={"leads": ocean, "summary": {"leads_found": 5000}}))

        with patch("requests.Session", return_value=session):
            first = providers.search(config, {}, 200, 0, 1).page
            second = providers.search(config, {}, 200, 200, 1).page
            session.post.reset_mock()
            providers.search(config, {}, 200, 400, 1)
            third_asked = session.post.call_args.kwargs["json"]

        assert second.leads[:50] and second.leads[0]["contact_full_name"] == "P200"
        assert third_asked["offset"] == 150 and third_asked["limit"] == 200
        # the bottom is where the stream ends, not where the dataset does
        assert 3 * PAGE_SIZE < first.leads_found

    def test_a_dataset_with_room_to_spare_reports_the_sum_of_both_counts(
            self, configure, people_csv):
        config = self._both(configure, people_csv)
        session = _session(_response(payload={"leads": [], "summary": {"leads_found": 10}}))

        with patch("requests.Session", return_value=session):
            page = providers.search(config, {}, limit=50, offset=0, first_poll_s=1).page

        assert page.leads_found == 3 + 10

    def _big_dataset(self, configure, tmp_path, matching):
        path = tmp_path / "big.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADER)
            for i in range(matching):
                writer.writerow([f"https://li/p{i}", f"P{i}", "Procurement Manager", "Co",
                                 "", "", "", "", "", ""])
        return configure(discovery_provider="leadocean+dataset", dataset_path=str(path),
                         leadocean_api_key="lo-key", leadocean_base_url="https://ocean.example")

    def test_the_stream_continues_into_lead_ocean_where_the_dataset_ends(
            self, configure, people_csv):
        config = self._both(configure, people_csv)
        remote = [{"contact_full_name": f"LO{i}", "contact_linkedin_profile_url": f"https://li/lo{i}"}
                  for i in range(3)]
        session = _session(_response(payload={"leads": remote, "summary": {"leads_found": 90}}))

        with patch("requests.Session", return_value=session):
            first = providers.search(config, {}, limit=5, offset=0, first_poll_s=1).page
            asked = session.post.call_args.kwargs["json"]
            later = providers.search(config, {}, limit=5, offset=5, first_poll_s=1).page
            later_asked = session.post.call_args.kwargs["json"]

        # 3 dataset people, then Lead Ocean from its own offset 0 for the 2 left on the page
        assert [r["contact_full_name"] for r in first.leads][:3] == ["Ann", "Bob", "Cy"]
        assert asked["limit"] == 2 and asked["offset"] == 0
        assert first.leads_found == 3 + 90
        # the next page starts 5 into the stream = 2 into Lead Ocean
        assert later_asked["offset"] == 2 and later_asked["limit"] == 5
        assert later.leads_found is None

    def test_a_lead_ocean_outage_stops_the_search_and_never_falls_back(
            self, configure, people_csv):
        config = self._both(configure, people_csv)
        session = _session(raises=requests.ConnectionError("down"))

        with patch("requests.Session", return_value=session), \
                patch("openoutfind.enrichment.bettercontact.search_and_poll") as upstream:
            with pytest.raises(ProviderUnavailable):
                providers.search(config, {}, limit=50, offset=0, first_poll_s=1)

        upstream.assert_not_called()

    def test_either_source_alone_satisfies_the_default_provider(self, configure, people_csv):
        assert providers.is_configured(configure(dataset_path=str(people_csv),
                                                 discovery_provider="leadocean+dataset"))
        assert providers.is_configured(configure(leadocean_api_key="k",
                                                 leadocean_base_url="https://x",
                                                 discovery_provider="leadocean+dataset"))


class TestSelectionAndConfig:
    def test_the_fork_defaults_to_lead_ocean_plus_dataset(self, monkeypatch):
        from openoutfind.core.config import SiteConfig

        monkeypatch.delenv("OPENOUTFIND_DISCOVERY_PROVIDER", raising=False)

        assert providers.provider_name(SiteConfig()) == "leadocean+dataset"

    def test_an_unknown_provider_is_bad_config(self, configure):
        with pytest.raises(OpenOutFindError) as raised:
            providers.provider_name(configure(discovery_provider="bing"))

        assert raised.value.error_type == ErrorType.BAD_CONFIG

    def test_a_bare_default_install_names_every_way_to_get_discovery(self, configure):
        configure(discovery_provider="leadocean+dataset")

        missing = readiness.missing_variables()["discovery"]

        assert set(missing) == {"OPENOUTFIND_DATASET_PATH", "OPENOUTFIND_LEADOCEAN_API_KEY",
                                "OPENOUTFIND_LEADOCEAN_BASE_URL"}

    def test_upstream_betterContact_is_still_reachable(self, configure):
        config = configure(discovery_provider="bettercontact", bettercontact_api_key="bc")
        body = {"leads": [{"contact_full_name": "Gus"}], "summary": {"leads_found": 7}}

        with patch("openoutfind.enrichment.bettercontact.search_and_poll",
                   return_value=(body, 1.0)) as upstream:
            page = providers.search(config, {}, limit=10, offset=0, first_poll_s=1.5).page

        assert upstream.call_args.args[0] == "bc"
        assert page.leads_found == 7


class TestTheWalkUsesTheConfiguredSource:
    """These fail on upstream: discovery was gated on, and fired at, BetterContact only."""

    def test_a_dataset_only_install_discovers_without_any_bettercontact_key(
            self, configure, people_csv, db):
        site = configure(discovery_provider="dataset", dataset_path=str(people_csv),
                         product_docs="p", campaign_target="t")
        assert site.bettercontact_api_key == ""

        with patch("openoutfind.core.pipeline.walk.run_round",
                   return_value=SimpleNamespace(fired=1)) as fire:
            fired = discover.discover(site)

        # Upstream gated this on `bettercontact.is_configured()` and never fired a round.
        assert fired is True
        fire.assert_called_once()

    def test_a_round_fires_the_dataset_and_never_calls_bettercontact(self, on_dataset, db):
        arm = SimpleNamespace(number=1, terms=Terms(title=frozenset({"procurement"})))

        with patch("openoutfind.enrichment.bettercontact.search_and_poll") as upstream, \
                patch("openoutfind.core.pipeline.walk._shape", return_value={}):
            results = walk._fire_all([arm], on_dataset)

        upstream.assert_not_called()
        fetched = results[0]
        assert {r["contact_full_name"] for r in fetched.page.leads} == {"Ann", "Bob"}

    def test_a_provider_outage_in_one_search_is_handed_back_not_raised(self, configure, db):
        config = _ocean_config(configure)
        arm = SimpleNamespace(number=1, terms=Terms(title=frozenset({"cto"})))
        session = _session(raises=requests.ConnectionError("down"))

        with patch("requests.Session", return_value=session), \
                patch("openoutfind.core.pipeline.walk._shape", return_value={}):
            results = walk._fire_all([arm], config)

        assert isinstance(results[0], ProviderUnavailable)


class TestTheUpstreamHubIsShutOffTheFork:
    """The hub is a third party's shared store. Lead Ocean rows and our dataset must not
    be given back to it, and our targeting must not be looked up there."""

    def test_the_hub_is_closed_unless_discovery_is_bettercontact(self, configure):
        assert hub.hub_enabled(configure(discovery_provider="dataset")) is False
        assert hub.hub_enabled(configure(discovery_provider="leadocean")) is False
        assert hub.hub_enabled(configure(discovery_provider="leadocean+dataset")) is False
        assert hub.hub_enabled(configure(discovery_provider="bettercontact")) is True

    def test_no_token_is_minted_and_nothing_is_posted_for_a_dataset_run(
            self, configure, db, operator):
        configure(discovery_provider="dataset", contacts_api_token="explicit")
        rows = [{"contact_linkedin_profile_url": "https://li/ann", "contact_full_name": "Ann"}]
        lead = SimpleNamespace(profile_url="https://li/ann", country_code="us")

        with patch("openoutfind.contacts.service.requests.post") as post, \
                patch("openoutfind.contacts.service.requests.get") as get:
            hub.share_profiles(rows, "us", {})
            hub.share_companies(rows, "us", {})
            hub.contribute(lead, ["ann@example.com"], hub.ORIGIN_BETTERCONTACT)
            assert hub.resolve(lead) is None
            assert hub.token_in_hand(hub.SiteConfig.load()) == ""

        post.assert_not_called()
        get.assert_not_called()
