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


class TestSelectionAndConfig:
    def test_the_fork_defaults_to_every_source_and_uses_the_ones_it_has(self, monkeypatch):
        from openoutfind.core.config import SiteConfig, active_sources

        monkeypatch.delenv("OPENOUTFIND_DISCOVERY_PROVIDER", raising=False)

        assert providers.provider_name(SiteConfig()) == "dataset+leadocean+bettercontact"
        assert active_sources(SiteConfig(dataset_path="d.csv", bettercontact_api_key="bc")) == (
            "dataset", "bettercontact")

    def test_lead_finder_is_another_name_for_bettercontact_and_order_is_fixed(self, configure):
        config = configure(discovery_provider="leadfinder+dataset")

        assert providers.provider_name(config) == "dataset+bettercontact"

    @pytest.mark.parametrize("name", ["bing", "dataset+bing", "+"])
    def test_an_unknown_source_is_bad_config(self, name):
        from openoutfind.core.config import SiteConfig, discovery_sources

        with pytest.raises(OpenOutFindError) as raised:
            discovery_sources(SiteConfig(discovery_provider=name))

        assert raised.value.error_type == ErrorType.BAD_CONFIG

    def test_a_bare_default_install_names_every_way_to_get_discovery(self, configure):
        configure(discovery_provider="dataset+leadocean+leadfinder")

        missing = readiness.missing_variables()["discovery"]

        assert set(missing) == {"OPENOUTFIND_DATASET_PATH", "OPENOUTFIND_LEADOCEAN_API_KEY",
                                "OPENOUTFIND_LEADOCEAN_BASE_URL",
                                "OPENOUTFIND_BETTERCONTACT_API_KEY"}

    def test_lead_finder_alone_satisfies_the_default(self, configure):
        """Keeping Lead Finder means a BetterContact key is a way in on its own."""
        assert providers.is_configured(configure(bettercontact_api_key="bc",
                                                 discovery_provider="dataset+leadocean+leadfinder"))

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
        config = configure(discovery_provider="leadocean", leadocean_api_key="lo-key",
                           leadocean_base_url="https://ocean.example")
        arm = SimpleNamespace(number=1, terms=Terms(title=frozenset({"cto"})))
        session = MagicMock()
        session.__enter__.return_value = session
        session.headers = {}
        session.post.side_effect = requests.ConnectionError("down")

        with patch("requests.Session", return_value=session), \
                patch("openoutfind.core.pipeline.walk._shape", return_value={}):
            results = walk._fire_all([arm], config)

        assert isinstance(results[0], ProviderUnavailable)


class TestTheUpstreamHubIsShutOffTheFork:
    """The hub is a third party's shared store. Lead Ocean rows and our dataset must not
    be given back to it, and our targeting must not be looked up there."""

    def test_the_hub_is_open_only_when_lead_finder_is_the_only_source_selected(self, configure):
        assert hub.hub_enabled(configure(discovery_provider="dataset")) is False
        assert hub.hub_enabled(configure(discovery_provider="leadocean")) is False
        assert hub.hub_enabled(configure(discovery_provider="bettercontact")) is True
        assert hub.hub_enabled(configure(discovery_provider="leadfinder")) is True
        # the default selects every source, so the hub is closed even with only a BC key
        assert hub.hub_enabled(configure(discovery_provider="dataset+leadocean+leadfinder",
                                         bettercontact_api_key="bc")) is False
        # a dataset beside Lead Finder would be given back with it: closed
        assert hub.hub_enabled(configure(discovery_provider="dataset+leadfinder",
                                         dataset_path="d.csv",
                                         bettercontact_api_key="bc")) is False

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
