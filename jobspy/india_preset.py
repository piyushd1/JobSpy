"""
Shared India search preset for the PM / GPM / Program / P&L role cluster.

This module provides pre-configured search parameters consumed by higher-level
wrappers (e.g. the ranking layer on the main branch).  It does NOT change the
default behaviour of scrape_jobs() for existing callers.

Usage
-----
    from jobspy.india_preset import INDIA_PM_PRESET, build_india_queries

    # Use the preset dict to feed scrape_jobs():
    df = scrape_jobs(**INDIA_PM_PRESET)

    # Or generate individual search-term strings for multi-pass scraping:
    for q in build_india_queries():
        df = scrape_jobs(search_term=q, **INDIA_PM_PRESET_BASE)
"""

from __future__ import annotations

from typing import Sequence

# ---------------------------------------------------------------------------
# Role cluster – the combined set of target titles / keywords
# ---------------------------------------------------------------------------

ROLE_KEYWORDS: list[str] = [
    "Product Manager",
    "Senior Product Manager",
    "Staff Product Manager",
    "Group Product Manager",
    "GPM",
    "Program Manager",
    "Senior Program Manager",
    "Project Manager",
    "Business Head",
    "P&L Head",
    "P&L Owner",
    "General Manager",
]

# ---------------------------------------------------------------------------
# India-specific scraper defaults
# ---------------------------------------------------------------------------

#: Sites that cover India jobs well.
INDIA_SITES: list[str] = ["naukri", "linkedin", "indeed"]

#: Base preset (no search_term – caller supplies one or uses build_india_queries).
INDIA_PM_PRESET_BASE: dict = {
    "site_name": INDIA_SITES,
    "country_indeed": "india",
    "description_format": "plain",
    "results_wanted": 30,
    "hours_old": 72,
    "linkedin_fetch_description": True,
}

#: Convenience preset with a broad OR-style search term.
INDIA_PM_PRESET: dict = {
    **INDIA_PM_PRESET_BASE,
    "search_term": " OR ".join(f'"{kw}"' for kw in ROLE_KEYWORDS[:6]),
}


def build_india_queries(
    keywords: Sequence[str] | None = None,
    *,
    batch_size: int = 3,
) -> list[str]:
    """
    Generates search-term strings by batching *keywords* into OR-groups.

    Parameters
    ----------
    keywords : sequence of str, optional
        Role keywords to combine.  Defaults to :data:`ROLE_KEYWORDS`.
    batch_size : int
        How many keywords to OR together per query string.  Smaller batches
        give more precise results but require more API calls.

    Returns
    -------
    list[str]
        A list of query strings ready for ``search_term=``.
    """
    kw = list(keywords or ROLE_KEYWORDS)
    queries: list[str] = []
    for i in range(0, len(kw), batch_size):
        batch = kw[i : i + batch_size]
        queries.append(" OR ".join(f'"{term}"' for term in batch))
    return queries
