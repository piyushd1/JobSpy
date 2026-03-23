from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jobspy import build_candidate_profile, search_jobs_for_profile

DEFAULT_ROLES = [
    "product manager",
    "senior product manager",
    "group product manager",
    "staff product manager",
    "program manager",
    "project manager",
    "business head",
    "general manager",
]
DEFAULT_LOCATIONS = ["Bengaluru", "Gurugram", "Mumbai"]
DEFAULT_MUST_HAVE = ["p&l", "growth", "monetization"]
DEFAULT_NICE_TO_HAVE = ["strategy", "marketplace", "go to market"]


def parse_multiline_list(raw_value: str) -> list[str]:
    return [
        item.strip()
        for item in raw_value.replace(",", "\n").splitlines()
        if item.strip()
    ]


def build_profile_from_ui(
    *,
    uploaded_resume,
    resume_text: str,
    target_roles: list[str],
    additional_roles: str,
    preferred_locations: str,
    must_have_keywords: str,
    nice_to_have_keywords: str,
    exclude_keywords: str,
    remote_preference: str,
    target_sites: list[str],
):
    temp_resume_path = None
    if uploaded_resume is not None:
        suffix = Path(uploaded_resume.name).suffix or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_resume.getbuffer())
            temp_resume_path = temp_file.name

    merged_roles = target_roles + parse_multiline_list(additional_roles)
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        profile = build_candidate_profile(
            resume_path=temp_resume_path,
            resume_text=resume_text,
            target_roles=merged_roles or DEFAULT_ROLES,
            preferred_locations=parse_multiline_list(preferred_locations),
            must_have_keywords=parse_multiline_list(must_have_keywords),
            nice_to_have_keywords=parse_multiline_list(nice_to_have_keywords),
            exclude_keywords=parse_multiline_list(exclude_keywords),
            remote_preference=remote_preference,
            target_sites=target_sites,
        )
    return profile, [str(item.message) for item in captured_warnings]


def render_summary_cards(results: pd.DataFrame):
    strong_matches = int((results["match_band"] == "strong").sum()) if "match_band" in results else 0
    avg_score = (
        round(float(results["match_score"].mean()), 1)
        if "match_score" in results and not results.empty
        else 0.0
    )
    strongest_company = (
        results.iloc[0]["company"]
        if not results.empty and "company" in results.columns
        else "-"
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Ranked Jobs", len(results))
    col2.metric("Strong Matches", strong_matches)
    col3.metric("Average Score", avg_score)

    st.caption(f"Top company in the current ranking: `{strongest_company}`")


def render_top_matches(results: pd.DataFrame):
    top_columns = [
        column
        for column in [
            "title",
            "company",
            "location",
            "site",
            "match_score",
            "match_band",
            "matched_roles",
            "search_query",
        ]
        if column in results.columns
    ]
    st.dataframe(results[top_columns], use_container_width=True, hide_index=True)


def render_detailed_results(results: pd.DataFrame):
    preferred_columns = [
        "title",
        "company",
        "location",
        "site",
        "job_url",
        "job_url_direct",
        "match_score",
        "match_band",
        "matched_roles",
        "matched_keywords",
        "match_reasons",
        "job_level",
        "job_function",
        "company_industry",
        "skills",
        "search_query",
        "description",
    ]
    visible_columns = [column for column in preferred_columns if column in results.columns]
    st.dataframe(results[visible_columns], use_container_width=True, hide_index=True)


def main():
    st.set_page_config(
        page_title="JobSpy India Finder",
        page_icon="JP",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(232, 246, 236, 0.95), transparent 30%),
                radial-gradient(circle at top right, rgba(255, 232, 214, 0.85), transparent 28%),
                linear-gradient(180deg, #f7f2ea 0%, #f1eee8 42%, #eef4ef 100%);
            color: #173528;
            font-family: "Georgia", "Aptos", serif;
        }
        .hero {
            padding: 1.4rem 1.6rem;
            border-radius: 20px;
            background: rgba(255, 252, 245, 0.82);
            border: 1px solid rgba(23, 53, 40, 0.10);
            box-shadow: 0 18px 35px rgba(23, 53, 40, 0.08);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 2.4rem;
            line-height: 1.1;
            color: #173528;
        }
        .hero p {
            margin: 0.6rem 0 0;
            font-size: 1rem;
            color: #41594d;
        }
        .section-card {
            padding: 0.8rem 1rem;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(23, 53, 40, 0.08);
        }
        </style>
        <div class="hero">
            <h1>JobSpy India Finder</h1>
            <p>Upload your resume, tune your PM / Program / P&amp;L preferences, and rank India jobs from Naukri, LinkedIn, and Indeed in one pass.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Candidate Profile")
        uploaded_resume = st.file_uploader(
            "Resume PDF",
            type=["pdf"],
            help="PDF parsing uses pypdf if it is installed in your environment.",
        )
        resume_text = st.text_area(
            "Resume Text Override",
            height=140,
            help="Optional fallback if you want to paste resume text directly.",
        )
        target_roles = st.multiselect(
            "Primary Role Cluster",
            options=DEFAULT_ROLES,
            default=DEFAULT_ROLES,
        )
        additional_roles = st.text_area(
            "Additional Roles",
            value="staff pm\nhead of product",
            height=90,
            help="One per line or comma-separated.",
        )
        preferred_locations = st.text_area(
            "Preferred Locations",
            value="\n".join(DEFAULT_LOCATIONS),
            height=90,
            help="One per line or comma-separated.",
        )
        must_have_keywords = st.text_area(
            "Must-Have Keywords",
            value="\n".join(DEFAULT_MUST_HAVE),
            height=90,
        )
        nice_to_have_keywords = st.text_area(
            "Nice-To-Have Keywords",
            value="\n".join(DEFAULT_NICE_TO_HAVE),
            height=90,
        )
        exclude_keywords = st.text_area(
            "Exclude Keywords",
            value="intern\nsoftware engineer\nfresher",
            height=90,
        )
        remote_preference = st.selectbox(
            "Remote Preference",
            options=["any", "remote", "hybrid", "onsite"],
            index=0,
        )
        target_sites = st.multiselect(
            "Target Portals",
            options=["naukri", "linkedin", "indeed"],
            default=["naukri", "linkedin", "indeed"],
        )

        st.markdown("### Search Controls")
        results_wanted_per_site = st.slider(
            "Results Per Site",
            min_value=10,
            max_value=100,
            value=40,
            step=10,
        )
        hours_old = st.slider(
            "Posted Within Hours",
            min_value=24,
            max_value=720,
            value=168,
            step=24,
        )
        dedupe = st.toggle("Deduplicate Similar Listings", value=True)
        verbose = st.select_slider("Verbose Logging", options=[0, 1, 2], value=0)
        run_search = st.button("Run India Search", use_container_width=True, type="primary")

    left, right = st.columns([1.2, 0.8], gap="large")
    with left:
        st.markdown("### What This UI Does")
        st.markdown(
            """
            - Builds a candidate profile from your resume PDF and manual inputs.
            - Searches India-facing roles across `naukri`, `linkedin`, and `indeed`.
            - Scores each job for PM / Program / Business ownership fit.
            - Lets you inspect the reasoning and export the shortlist as CSV.
            """
        )
    with right:
        st.markdown("### Quick Notes")
        st.markdown(
            """
            - Keep must-have keywords focused on your strongest signals.
            - Use exclude keywords to suppress internships or IC engineering roles.
            - If PDF parsing is unavailable, paste your resume text into the override box.
            """
        )

    if not run_search:
        st.info("Set your profile on the left, then click `Run India Search`.")
        return

    if uploaded_resume is None and not resume_text.strip():
        st.warning("Upload a resume PDF or paste resume text before running the search.")
        return

    try:
        profile, profile_warnings = build_profile_from_ui(
            uploaded_resume=uploaded_resume,
            resume_text=resume_text,
            target_roles=target_roles,
            additional_roles=additional_roles,
            preferred_locations=preferred_locations,
            must_have_keywords=must_have_keywords,
            nice_to_have_keywords=nice_to_have_keywords,
            exclude_keywords=exclude_keywords,
            remote_preference=remote_preference,
            target_sites=target_sites,
        )
        for warning_message in profile_warnings:
            st.warning(warning_message)
        with st.spinner("Searching and ranking jobs..."):
            results = search_jobs_for_profile(
                profile,
                results_wanted_per_site=results_wanted_per_site,
                hours_old=hours_old,
                dedupe=dedupe,
                verbose=verbose,
            )
    except Exception as exc:
        st.error(f"Search failed: {exc}")
        return

    if results.empty:
        st.warning("No jobs were returned for the current profile and search settings.")
        return

    render_summary_cards(results)

    st.markdown("### Top Matches")
    render_top_matches(results.head(25))

    csv_bytes = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Ranked CSV",
        data=csv_bytes,
        file_name="jobspy_india_ranked_jobs.csv",
        mime="text/csv",
        use_container_width=False,
    )

    with st.expander("Detailed Results", expanded=True):
        render_detailed_results(results)

    with st.expander("Profile Snapshot"):
        st.json(profile.model_dump())


if __name__ == "__main__":
    main()


