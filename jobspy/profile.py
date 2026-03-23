from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from jobspy.india_preset import INDIA_PM_PRESET_BASE
from jobspy.model import Site
from jobspy.util import create_logger, desired_order

log = create_logger("ProfileSearch")

DEFAULT_TARGET_ROLES = [
    "product manager",
    "senior product manager",
    "group product manager",
    "staff product manager",
    "program manager",
    "project manager",
    "business head",
    "general manager",
]
DEFAULT_SENIORITY_TARGETS = [
    "senior",
    "lead",
    "staff",
    "group",
    "principal",
    "head",
    "director",
    "general manager",
]
DEFAULT_TARGET_SITES = ["naukri", "linkedin", "indeed"]
PROFILE_RESULT_COLUMNS = [
    "match_score",
    "match_band",
    "matched_roles",
    "matched_keywords",
    "match_reasons",
    "search_query",
]

ROLE_QUERY_ALIASES = {
    "pm": "product manager",
    "product": "product manager",
    "senior pm": "senior product manager",
    "sr product manager": "senior product manager",
    "gpm": "group product manager",
    "group pm": "group product manager",
    "staff pm": "staff product manager",
    "program": "program manager",
    "project": "project manager",
    "gm": "general manager",
}
ROLE_MATCHERS = {
    "product manager": [
        "product manager",
        "group product manager",
        "senior product manager",
        "staff product manager",
        "principal product manager",
        "lead product manager",
        "product lead",
    ],
    "program manager": [
        "program manager",
        "technical program manager",
        "senior program manager",
    ],
    "project manager": [
        "project manager",
        "delivery manager",
        "implementation manager",
    ],
    "business head": [
        "business head",
        "business leader",
        "business manager",
        "category head",
        "category manager",
        "profit center head",
    ],
    "general manager": [
        "general manager",
        "gm business",
        "gm growth",
        "gm revenue",
        "gm strategy",
    ],
}
ROLE_WEIGHTS = {
    "product manager": 34,
    "program manager": 28,
    "project manager": 20,
    "business head": 30,
    "general manager": 30,
}
P_AND_L_SIGNALS = [
    "p&l",
    "revenue",
    "profit",
    "profitability",
    "growth",
    "gmv",
    "monetization",
    "pricing",
    "category",
    "commercial",
    "margin",
    "ebitda",
    "marketplace",
    "business ownership",
    "business owner",
    "strategy",
    "go to market",
    "gtm",
]
SIGNAL_QUERY_MAP = {
    "p&l": ["business head p&l", "general manager p&l"],
    "revenue": ["business head revenue"],
    "growth": ["product manager growth"],
    "monetization": ["product manager monetization"],
    "pricing": ["product manager pricing"],
    "marketplace": ["product manager marketplace"],
    "commercial": ["business head commercial"],
    "strategy": ["program manager strategy"],
    "go to market": ["product manager go to market"],
    "gtm": ["product manager gtm"],
}
OFF_TARGET_TERMS = [
    "software engineer",
    "developer",
    "sde",
    "data engineer",
    "qa engineer",
    "devops engineer",
    "frontend engineer",
    "backend engineer",
]
EARLY_STAGE_PENALTIES = [
    "intern",
    "internship",
    "trainee",
    "fresher",
    "junior",
    "campus",
]


class CandidateProfile(BaseModel):
    resume_path: str | None = None
    resume_text: str | None = None
    target_roles: list[str] = Field(default_factory=lambda: list(DEFAULT_TARGET_ROLES))
    preferred_locations: list[str] = Field(default_factory=list)
    must_have_keywords: list[str] = Field(default_factory=list)
    nice_to_have_keywords: list[str] = Field(default_factory=list)
    exclude_keywords: list[str] = Field(default_factory=list)
    remote_preference: str | None = "any"
    seniority_targets: list[str] = Field(
        default_factory=lambda: list(DEFAULT_SENIORITY_TARGETS)
    )
    target_sites: list[str] = Field(
        default_factory=lambda: list(DEFAULT_TARGET_SITES)
    )

    @field_validator(
        "target_roles",
        "preferred_locations",
        "must_have_keywords",
        "nice_to_have_keywords",
        "exclude_keywords",
        "seniority_targets",
        mode="before",
    )
    @classmethod
    def _coerce_string_list(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item).strip() for item in value if str(item).strip()]

    @field_validator("target_sites", mode="before")
    @classmethod
    def _coerce_sites(cls, value):
        if value is None:
            return list(DEFAULT_TARGET_SITES)
        if isinstance(value, (str, Site)):
            value = [value]
        sites = []
        for item in value:
            if isinstance(item, Site):
                sites.append(item.value)
            else:
                sites.append(str(item).strip().lower())
        return sites

    @field_validator("remote_preference", mode="before")
    @classmethod
    def _normalize_remote_preference(cls, value):
        if value is None:
            return "any"
        normalized = _normalize_text(value)
        if normalized in {"remote", "hybrid", "onsite", "any"}:
            return normalized
        return "any"


def build_candidate_profile(
    *,
    resume_path: str | Path | None = None,
    resume_text: str | None = None,
    target_roles: list[str] | str | None = None,
    preferred_locations: list[str] | str | None = None,
    must_have_keywords: list[str] | str | None = None,
    nice_to_have_keywords: list[str] | str | None = None,
    exclude_keywords: list[str] | str | None = None,
    remote_preference: str | None = "any",
    seniority_targets: list[str] | str | None = None,
    target_sites: list[str | Site] | str | Site | None = None,
) -> CandidateProfile:
    normalized_resume_path = str(Path(resume_path)) if resume_path else None
    extracted_resume_text = _extract_resume_text(normalized_resume_path)
    combined_resume_text = _combine_resume_text(
        explicit_text=resume_text, extracted_text=extracted_resume_text
    )
    return CandidateProfile(
        resume_path=normalized_resume_path,
        resume_text=combined_resume_text,
        target_roles=target_roles or list(DEFAULT_TARGET_ROLES),
        preferred_locations=preferred_locations or [],
        must_have_keywords=must_have_keywords or [],
        nice_to_have_keywords=nice_to_have_keywords or [],
        exclude_keywords=exclude_keywords or [],
        remote_preference=remote_preference,
        seniority_targets=seniority_targets or list(DEFAULT_SENIORITY_TARGETS),
        target_sites=target_sites or list(DEFAULT_TARGET_SITES),
    )


def search_jobs_for_profile(
    profile: CandidateProfile,
    *,
    results_wanted_per_site: int = 50,
    hours_old: int = 168,
    locations: list[str] | str | None = None,
    target_sites: list[str | Site] | str | Site | None = None,
    dedupe: bool = True,
    verbose: int = 0,
) -> pd.DataFrame:
    from jobspy import scrape_jobs

    profile = (
        profile
        if isinstance(profile, CandidateProfile)
        else CandidateProfile.model_validate(profile)
    )
    effective_sites = _coerce_sites(target_sites) or _coerce_sites(
        profile.target_sites
    ) or list(DEFAULT_TARGET_SITES)
    effective_locations = _resolve_locations(profile, locations)
    queries = _generate_profile_queries(profile)
    search_frames: list[pd.DataFrame] = []
    base_scrape_kwargs = {
        **INDIA_PM_PRESET_BASE,
        "site_name": effective_sites,
        "country_indeed": "india",
        "description_format": "plain",
        "linkedin_fetch_description": True,
    }

    for location in effective_locations:
        for query in queries:
            log.info(
                "Profile search query='%s' location='%s' sites=%s",
                query,
                location,
                ",".join(site.value for site in effective_sites),
            )
            jobs_df = scrape_jobs(
                **base_scrape_kwargs,
                search_term=query,
                location=location,
                results_wanted=results_wanted_per_site,
                hours_old=hours_old,
                verbose=verbose,
            )
            if jobs_df.empty:
                continue
            jobs_df = jobs_df.copy()
            jobs_df["search_query"] = query
            search_frames.append(jobs_df)

    if not search_frames:
        return _empty_profile_results_dataframe()

    ranked_jobs = pd.concat(search_frames, ignore_index=True)
    ranked_jobs = _score_jobs_dataframe(ranked_jobs, profile)
    if dedupe:
        ranked_jobs = _dedupe_jobs_dataframe(ranked_jobs)

    base_columns = [column for column in desired_order if column in ranked_jobs.columns]
    trailing_columns = [
        column
        for column in PROFILE_RESULT_COLUMNS
        if column in ranked_jobs.columns and column not in base_columns
    ]
    remaining_columns = [
        column
        for column in ranked_jobs.columns
        if column not in base_columns and column not in trailing_columns
    ]
    ordered_columns = base_columns + remaining_columns + trailing_columns
    ranked_jobs = ranked_jobs[ordered_columns]

    if "match_score" in ranked_jobs.columns:
        ranked_jobs = ranked_jobs.sort_values(
            by=["match_score"], ascending=[False], kind="stable"
        ).reset_index(drop=True)
    return ranked_jobs


def score_jobs_for_profile(
    jobs_df: pd.DataFrame, profile: CandidateProfile | Mapping[str, Any]
) -> pd.DataFrame:
    profile = (
        profile
        if isinstance(profile, CandidateProfile)
        else CandidateProfile.model_validate(profile)
    )
    ranked_jobs = _score_jobs_dataframe(jobs_df, profile)
    return ranked_jobs.sort_values(
        by=["match_score"], ascending=[False], kind="stable"
    ).reset_index(drop=True)


def _extract_resume_text(resume_path: str | None) -> str:
    if not resume_path:
        return ""
    try:
        from pypdf import PdfReader
    except ImportError:
        warnings.warn(
            "pypdf is not installed; resume PDF extraction was skipped.",
            stacklevel=2,
        )
        return ""

    try:
        reader = PdfReader(resume_path)
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        warnings.warn(
            f"Unable to read resume PDF '{resume_path}': {exc}",
            stacklevel=2,
        )
        return ""
    return _clean_whitespace("\n".join(pages))


def _combine_resume_text(*, explicit_text: str | None, extracted_text: str | None) -> str:
    chunks = []
    for text in (explicit_text, extracted_text):
        cleaned = _clean_whitespace(text or "")
        if cleaned and cleaned not in chunks:
            chunks.append(cleaned)
    return "\n\n".join(chunks)


def _generate_profile_queries(profile: CandidateProfile) -> list[str]:
    queries: list[str] = []
    seen = set()

    def add_query(query: str):
        normalized = _clean_whitespace(query).lower()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        queries.append(normalized)

    for query in DEFAULT_TARGET_ROLES:
        add_query(query)

    for role in profile.target_roles:
        normalized_role = ROLE_QUERY_ALIASES.get(_normalize_text(role), _clean_whitespace(role).lower())
        add_query(normalized_role)

    for signal in _extract_profile_signals(profile):
        for query in SIGNAL_QUERY_MAP.get(signal, []):
            add_query(query)

    return queries[:12]


def _score_jobs_dataframe(
    jobs_df: pd.DataFrame, profile: CandidateProfile
) -> pd.DataFrame:
    jobs_df = jobs_df.copy()
    score_rows = [_score_job_match(job_row, profile) for job_row in jobs_df.to_dict("records")]
    score_df = pd.DataFrame(score_rows)
    for column in PROFILE_RESULT_COLUMNS:
        jobs_df[column] = score_df[column]
    return jobs_df


def _score_job_match(job_row: Mapping[str, Any], profile: CandidateProfile) -> dict[str, Any]:
    title = _clean_whitespace(str(job_row.get("title") or ""))
    description = _clean_whitespace(str(job_row.get("description") or ""))
    location = _clean_whitespace(str(job_row.get("location") or ""))
    company_industry = _clean_whitespace(str(job_row.get("company_industry") or ""))
    job_level = _clean_whitespace(str(job_row.get("job_level") or ""))
    job_function = _clean_whitespace(str(job_row.get("job_function") or ""))
    skills = _clean_whitespace(str(job_row.get("skills") or ""))
    work_from_home_type = _clean_whitespace(
        str(job_row.get("work_from_home_type") or "")
    )
    title_text = _normalize_text(title)
    body_text = _normalize_text(
        " ".join(
            [
                title,
                description,
                company_industry,
                job_level,
                job_function,
                skills,
                location,
                work_from_home_type,
            ]
        )
    )

    score = 0
    matched_roles: list[str] = []
    matched_keywords: list[str] = []
    reasons: list[str] = []

    for role_name, phrases in ROLE_MATCHERS.items():
        title_matches = [phrase for phrase in phrases if phrase in title_text]
        body_matches = [phrase for phrase in phrases if phrase in body_text]
        if title_matches:
            matched_roles.append(role_name)
            score += ROLE_WEIGHTS[role_name]
            reasons.append(f"title matches {role_name}")
            if len(title_matches[0].split()) >= 3:
                score += 4
        elif body_matches:
            matched_roles.append(role_name)
            score += ROLE_WEIGHTS[role_name] // 2
            reasons.append(f"description mentions {role_name}")

    seniority_text = _normalize_text(" ".join([title, job_level]))
    seniority_hits = []
    for seniority in profile.seniority_targets:
        normalized_seniority = _normalize_text(seniority)
        if normalized_seniority and normalized_seniority in seniority_text:
            seniority_hits.append(normalized_seniority)
    if seniority_hits:
        score += min(16, 6 + (len(seniority_hits) * 3))
        reasons.append(f"seniority fit: {', '.join(_unique_preserve_order(seniority_hits))}")

    profile_signals = _extract_profile_signals(profile)
    pnl_hits = [signal for signal in profile_signals if signal in body_text]
    if pnl_hits:
        matched_keywords.extend(pnl_hits)
        score += min(18, 6 + (len(pnl_hits) * 3))
        reasons.append(f"ownership signals: {', '.join(_unique_preserve_order(pnl_hits))}")

    must_have_hits = [
        _normalize_text(keyword)
        for keyword in profile.must_have_keywords
        if _normalize_text(keyword) and _normalize_text(keyword) in body_text
    ]
    if must_have_hits:
        matched_keywords.extend(must_have_hits)
        score += min(16, len(_unique_preserve_order(must_have_hits)) * 4)
        reasons.append(
            f"matched must-have keywords: {', '.join(_unique_preserve_order(must_have_hits))}"
        )

    nice_to_have_hits = [
        _normalize_text(keyword)
        for keyword in profile.nice_to_have_keywords
        if _normalize_text(keyword) and _normalize_text(keyword) in body_text
    ]
    if nice_to_have_hits:
        matched_keywords.extend(nice_to_have_hits)
        score += min(10, len(_unique_preserve_order(nice_to_have_hits)) * 2)
        reasons.append(
            "matched nice-to-have keywords: "
            + ", ".join(_unique_preserve_order(nice_to_have_hits))
        )

    preferred_location_hits = [
        location_name
        for location_name in profile.preferred_locations
        if _normalize_text(location_name) in _normalize_text(location)
    ]
    if preferred_location_hits:
        score += 8
        reasons.append(
            f"preferred location match: {', '.join(_unique_preserve_order(preferred_location_hits))}"
        )

    is_remote = bool(job_row.get("is_remote"))
    remote_text = _normalize_text(work_from_home_type)
    if profile.remote_preference == "remote":
        if is_remote or "remote" in remote_text:
            score += 6
            reasons.append("remote preference matches")
        else:
            score -= 4
    elif profile.remote_preference == "hybrid":
        if "hybrid" in remote_text:
            score += 6
            reasons.append("hybrid preference matches")
    elif profile.remote_preference == "onsite":
        if is_remote or "remote" in remote_text:
            score -= 4

    exclude_hits = [
        _normalize_text(keyword)
        for keyword in profile.exclude_keywords
        if _normalize_text(keyword) and _normalize_text(keyword) in body_text
    ]
    if exclude_hits:
        score -= min(24, len(_unique_preserve_order(exclude_hits)) * 8)
        reasons.append(f"contains excluded keywords: {', '.join(_unique_preserve_order(exclude_hits))}")

    early_stage_hits = [term for term in EARLY_STAGE_PENALTIES if term in body_text]
    if early_stage_hits:
        score -= 24
        reasons.append(
            f"penalized early-career indicators: {', '.join(_unique_preserve_order(early_stage_hits))}"
        )

    off_target_hits = [term for term in OFF_TARGET_TERMS if term in title_text]
    if off_target_hits and not matched_roles:
        score -= 36
        reasons.append(
            f"penalized off-target engineering role: {', '.join(_unique_preserve_order(off_target_hits))}"
        )

    score = max(0, min(100, score))
    return {
        "match_score": score,
        "match_band": _score_to_band(score),
        "matched_roles": ", ".join(_unique_preserve_order(matched_roles)) or None,
        "matched_keywords": ", ".join(_unique_preserve_order(matched_keywords)) or None,
        "match_reasons": "; ".join(_unique_preserve_order(reasons)) or "No strong match signals found",
        "search_query": job_row.get("search_query"),
    }


def _dedupe_jobs_dataframe(jobs_df: pd.DataFrame) -> pd.DataFrame:
    jobs_df = jobs_df.copy()
    jobs_df["_dedupe_key"] = jobs_df.apply(_build_dedupe_key, axis=1)
    deduped_rows = []
    for _, group in jobs_df.groupby("_dedupe_key", sort=False, dropna=False):
        best_row = group.sort_values(
            by=["match_score"], ascending=[False], kind="stable"
        ).iloc[0].copy()
        best_row["search_query"] = ", ".join(
            _unique_preserve_order(
                [
                    query
                    for query in group["search_query"].fillna("").tolist()
                    if str(query).strip()
                ]
            )
        ) or None
        deduped_rows.append(best_row)
    deduped_df = pd.DataFrame(deduped_rows)
    return deduped_df.drop(columns=["_dedupe_key"]).reset_index(drop=True)


def _build_dedupe_key(job_row: pd.Series) -> str:
    direct_url = _normalize_url(job_row.get("job_url_direct"))
    if direct_url:
        return direct_url
    return "|".join(
        [
            _normalize_text(job_row.get("title")),
            _normalize_text(job_row.get("company")),
            _normalize_text(job_row.get("location")),
        ]
    )


def _normalize_url(url: Any) -> str:
    if not url:
        return ""
    try:
        parts = urlsplit(str(url).strip())
    except ValueError:
        return str(url).strip().lower()
    normalized = urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, "", ""))
    return normalized.rstrip("/")


def _resolve_locations(
    profile: CandidateProfile, locations: list[str] | str | None
) -> list[str]:
    if isinstance(locations, str):
        return [locations]
    if locations:
        return [location for location in locations if str(location).strip()]
    if profile.preferred_locations:
        return list(profile.preferred_locations)
    return ["India"]


def _extract_profile_signals(profile: CandidateProfile) -> list[str]:
    signals = []
    haystack = _normalize_text(profile.resume_text or "")
    for signal in P_AND_L_SIGNALS:
        if signal in haystack:
            signals.append(signal)
    for keyword in profile.must_have_keywords + profile.nice_to_have_keywords:
        normalized_keyword = _normalize_text(keyword)
        if normalized_keyword in SIGNAL_QUERY_MAP or normalized_keyword in P_AND_L_SIGNALS:
            signals.append(normalized_keyword)
    return _unique_preserve_order(signals)


def _coerce_sites(
    target_sites: list[str | Site] | str | Site | None,
) -> list[Site]:
    if target_sites is None:
        return []
    if isinstance(target_sites, (str, Site)):
        target_sites = [target_sites]
    sites = []
    for site in target_sites:
        if isinstance(site, Site):
            sites.append(site)
        else:
            sites.append(Site[str(site).strip().upper()])
    return sites


def _empty_profile_results_dataframe() -> pd.DataFrame:
    columns = desired_order + [column for column in PROFILE_RESULT_COLUMNS if column not in desired_order]
    return pd.DataFrame(columns=columns)


def _score_to_band(score: int) -> str:
    if score >= 75:
        return "strong"
    if score >= 50:
        return "moderate"
    return "low"


def _clean_whitespace(value: str) -> str:
    return " ".join(str(value).split()).strip()


def _normalize_text(value: Any) -> str:
    text = _clean_whitespace(str(value or "")).lower()
    for old, new in {
        "_": " ",
        "/": " ",
        "-": " ",
        ",": " ",
        ".": " ",
        ":": " ",
        ";": " ",
        "(": " ",
        ")": " ",
        "[": " ",
        "]": " ",
        "{": " ",
        "}": " ",
    }.items():
        text = text.replace(old, new)
    text = " ".join(text.split())
    return text.replace("p & l", "p&l")


def _unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen = set()
    unique_values = []
    for value in values:
        if value in seen or value in {"", None}:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values
