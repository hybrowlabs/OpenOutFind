# openoutfind/core/config.py
"""What this install was told, read from the environment on every run.

**These are answers a human gives, and this program is not the one asking.** Remembering
what somebody typed is a convenience for somebody who types; an agent supplies its
environment on every invocation and has nothing to remember. So there is no row and no
singleton here — `OPENOUTFIND_*` is the whole configuration surface, and it is read fresh
each run. The wizard that asks these questions lives in OpenOutreach, which owns the
human half and exports these names.

**What the pipeline produced still lives in the store**, and the line is who produced the
value rather than whether it is a secret:

  * the walk's nodes and vocabulary — measured, `core/models.py`
  * the invented ideal leads — LLM-written once and permanent, and they are *leads*, so
    they are `Lead` rows carrying `synthetic=True` (`core/pipeline/icp.py`)
  * the ICP size band — written onto the query nodes it rides, where it is part of the
    query that was actually fired
  * the operator — identity, a `User` row written once, because a renamed variable must
    not rename the person a campaign belongs to

Nothing else was ever produced. The fitted GP used to be dumped into a `model_blob`
column that nothing ever read back: the fit is reproduced from the label rows every time
the evidence changes (`core/ml/qualifier.py:qualifier_for`), so the blob was a cache of a
value already derived, and it is gone.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields

ENV_PREFIX = "OPENOUTFIND_"

# What each field is read from, where the variable is not simply the field's own name
# upper-cased. `operator_country_code` would otherwise upper-case to
# `OPERATOR_COUNTRY_CODE` — one word longer than the name OpenOutSend already uses
# (`OUTSEND_OPERATOR_COUNTRY`) for the same concept.
_ALIASES = {"operator_country_code": "OPERATOR_COUNTRY"}

# The model needs both halves or it cannot be built; `llm_api_base` is not among them
# because only the `openai_compatible` provider reads it, and that builder raises its own
# error naming it. Requiring it here would stop an Anthropic install over a value it
# never looks at.
REQUIRED_LLM_FIELDS = ("ai_model", "llm_api_key")

# What the ICP is written from. Without these there is no seed, no anchor and no verdict
# — every LLM call this tool makes renders one of them into its prompt.
REQUIRED_ICP_FIELDS = ("product_docs", "campaign_target")

# Where leads are discovered. Upstream shipped one source, BetterContact Lead Finder; this
# fork (OPS-636) adds Lead Ocean and an operator-supplied dataset file behind the same walk.
# ``bettercontact`` stays selectable so upstream behaviour is one variable away.
DISCOVERY_PROVIDERS = ("leadocean+dataset", "leadocean", "dataset", "bettercontact")
DEFAULT_DISCOVERY_PROVIDER = "leadocean+dataset"

# What each single-source provider needs to search at all. ``leadocean+dataset`` is
# satisfied by *either* source being complete (``required_discovery_fields``).
_DISCOVERY_FIELDS = {
    "bettercontact": ("bettercontact_api_key",),
    "leadocean": ("leadocean_api_key", "leadocean_base_url"),
    "dataset": ("dataset_path",),
}


@dataclass(frozen=True)
class SiteConfig:
    """The configuration this run was given.

    Frozen because nothing may write it: a value edited at runtime would live until the
    process ended and then be gone, which is the silent half-persistence this replaced.
    Empty string rather than None throughout — "unset" and "deliberately blank" are the
    same state to a prompt template, and the template renders every one of them as text.
    """

    # A pydantic-ai model identifier in `provider:model` form (e.g.
    # `anthropic:claude-sonnet-4-5-20250929`, `openai:gpt-4o`, `groq:llama-3.3-70b`). The
    # provider lives inside this single string — there is no separate provider field to
    # drift out of sync. A bare model name whose prefix is unambiguous (`gpt`/`o1`/`o3`
    # →openai, `claude`→anthropic, `gemini`→google) is also accepted; everything else
    # must carry an explicit prefix. See core/llm.py:split_model_id.
    ai_model: str = ""
    llm_api_key: str = ""
    # Only consulted for the openai_compatible provider (OpenRouter / Together / Ollama / vLLM).
    llm_api_base: str = ""

    # BetterContact's key powers Lead Finder *discovery*, which is billed nothing, and the
    # paid email lookup behind `--emails`.
    bettercontact_api_key: str = ""

    # Discovery source (OPS-636). Blank means ``DEFAULT_DISCOVERY_PROVIDER``. The Lead Ocean
    # key and base URL and the dataset path come from the environment only; they are never
    # written to the store or the repo.
    discovery_provider: str = ""
    leadocean_api_key: str = ""
    leadocean_base_url: str = ""
    dataset_path: str = ""

    # The operator's own ISO-3166 alpha-2 jurisdiction — not to be confused with
    # `Lead.country_code`, the per-lead target country an ICP search surfaced someone
    # under. Optional: an EEA/UK/CH country turns the contacts-store give-back off
    # (core/geo.py:is_eea_located), any other country or none leaves it on.
    operator_country_code: str = ""

    # The campaign content: what this install sells, and to whom. The two things every
    # prompt in the tool is written from.
    product_docs: str = ""
    campaign_target: str = ""

    # Optional: the most qualified leads one company may supply (core/company_cap.py). A
    # campaign's answer like the two above, fixed for its life. Blank means no cap.
    max_per_company: str = ""

    # Central contacts store (see openoutfind/contacts/). The token names this install to
    # the hub; blank means the run registers for one itself and keeps it for the length of
    # the process.
    contacts_api_token: str = ""

    @classmethod
    def load(cls) -> "SiteConfig":
        """This run's configuration, read from `OPENOUTFIND_*`."""
        return cls(**{field.name: _from_environment(variable_for(field.name))
                      for field in fields(cls)})


def variable_for(field_name: str) -> str:
    """The environment variable one field is read from."""
    return ENV_PREFIX + _ALIASES.get(field_name, field_name.upper())


def missing(config: SiteConfig, field_names) -> list[str]:
    """The variables that would have filled the named fields this run lacks."""
    return [variable_for(name) for name in field_names if not getattr(config, name)]


def discovery_provider(config: SiteConfig) -> str:
    """The discovery source this run uses; ``bad_config`` if the name is not one we ship."""
    from openoutfind.core.errors import ErrorType, OpenOutFindError

    name = (config.discovery_provider or DEFAULT_DISCOVERY_PROVIDER).strip().lower()
    if name not in DISCOVERY_PROVIDERS:
        raise OpenOutFindError(
            ErrorType.BAD_CONFIG,
            f"{variable_for('discovery_provider')}={name!r} is not a discovery provider; "
            f"use one of {', '.join(DISCOVERY_PROVIDERS)}")
    return name


def required_discovery_fields(config: SiteConfig) -> tuple[str, ...]:
    """The fields still missing for the chosen discovery provider, as field names.

    ``leadocean+dataset`` is satisfied by either source being complete. When neither is,
    the answer names the dataset path and the Lead Ocean pair, so one error tells an
    operator every way out.
    """
    name = discovery_provider(config)
    if name != "leadocean+dataset":
        return tuple(f for f in _DISCOVERY_FIELDS[name] if not getattr(config, f))
    lead_ocean = tuple(f for f in _DISCOVERY_FIELDS["leadocean"] if not getattr(config, f))
    dataset = tuple(f for f in _DISCOVERY_FIELDS["dataset"] if not getattr(config, f))
    if not lead_ocean or not dataset:
        return ()
    return dataset + lead_ocean


def _from_environment(variable: str) -> str:
    """One `OPENOUTFIND_*` value, stripped — "" whether it was unset or blank."""
    return (os.environ.get(variable) or "").strip()
