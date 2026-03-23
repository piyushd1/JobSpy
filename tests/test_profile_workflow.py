from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_public_api():
    module_candidates = ["jobspy.profile", "jobspy"]
    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        if all(
            hasattr(module, attr)
            for attr in (
                "CandidateProfile",
                "build_candidate_profile",
                "search_jobs_for_profile",
            )
        ):
            return module
    raise AssertionError(
        "Could not locate CandidateProfile, build_candidate_profile, and "
        "search_jobs_for_profile in the expected public modules."
    )


PUBLIC_API = _load_public_api()
CandidateProfile = PUBLIC_API.CandidateProfile
build_candidate_profile = PUBLIC_API.build_candidate_profile
search_jobs_for_profile = PUBLIC_API.search_jobs_for_profile


def _patch_scrape_jobs(side_effect):
    if hasattr(PUBLIC_API, "scrape_jobs"):
        return patch.object(PUBLIC_API, "scrape_jobs", side_effect=side_effect)
    package = importlib.import_module("jobspy")
    return patch.object(package, "scrape_jobs", side_effect=side_effect)


class ProfileWorkflowTests(unittest.TestCase):
    def test_build_candidate_profile_uses_manual_overrides_when_pdf_is_empty(self):
        profile = build_candidate_profile(
            resume_path=str(ROOT / "tests" / "fixtures" / "missing_resume.pdf"),
            target_roles=["product manager", "program manager"],
            must_have_keywords=["p&l", "gpm", "product strategy"],
            nice_to_have_keywords=["stakeholder management"],
            exclude_keywords=["intern", "associate"],
            preferred_locations=["Bengaluru", "Mumbai"],
            remote_preference=False,
            target_sites=["naukri", "linkedin", "indeed"],
        )

        self.assertIsInstance(profile, CandidateProfile)
        self.assertTrue(profile.target_roles)
        self.assertIn("product manager", {role.lower() for role in profile.target_roles})
        self.assertIn("program manager", {role.lower() for role in profile.target_roles})
        self.assertIn("p&l", {kw.lower() for kw in profile.must_have_keywords})
        self.assertIn("Bengaluru", profile.preferred_locations)
        self.assertIn("naukri", {str(site).lower() for site in profile.target_sites})
        self.assertIn("linkedin", {str(site).lower() for site in profile.target_sites})
        self.assertIn("indeed", {str(site).lower() for site in profile.target_sites})

    def test_search_jobs_for_profile_generates_pm_role_cluster_queries(self):
        profile = CandidateProfile(
            resume_text="Senior product leader with P&L ownership.",
            target_roles=["product manager", "program manager", "business head"],
            must_have_keywords=["p&l", "growth"],
            target_sites=["naukri", "linkedin", "indeed"],
            preferred_locations=["India"],
        )

        fake_jobs = pd.DataFrame(
            [
                {
                    "id": "job-1",
                    "site": "naukri",
                    "job_url": "https://example.com/1",
                    "title": "Senior Product Manager",
                    "company": "Acme",
                    "location": "Bengaluru",
                    "description": "Own roadmap and P&L for payments.",
                }
            ]
        )

        scrape_calls = []

        def fake_scrape_jobs(**kwargs):
            scrape_calls.append(kwargs)
            return fake_jobs.copy()

        with _patch_scrape_jobs(fake_scrape_jobs):
            result = search_jobs_for_profile(
                profile,
                results_wanted_per_site=10,
                hours_old=168,
                locations=None,
                target_sites=["naukri", "linkedin", "indeed"],
                dedupe=True,
                verbose=0,
            )

        self.assertGreaterEqual(len(scrape_calls), 1)
        all_queries = " ".join(
            str(call.get("search_term", "")) + " " + str(call.get("google_search_term", ""))
            for call in scrape_calls
        ).lower()
        self.assertIn("product manager", all_queries)
        self.assertIn("senior product manager", all_queries)
        self.assertIn("group product manager", all_queries)
        self.assertIn("staff product manager", all_queries)
        self.assertIn("program manager", all_queries)
        self.assertIn("project manager", all_queries)
        self.assertIn("business head", all_queries)
        self.assertIn("general manager", all_queries)
        self.assertIsInstance(result, pd.DataFrame)

    def test_matching_distinguishes_strong_pm_from_engineering_and_intern_roles(self):
        profile = CandidateProfile(
            resume_text=(
                "Product leader with portfolio ownership, P&L responsibility, "
                "market strategy, and cross-functional delivery."
            ),
            target_roles=["product manager", "program manager", "business head"],
            must_have_keywords=["p&l", "product strategy"],
            exclude_keywords=["intern", "software engineer"],
        )

        jobs = pd.DataFrame(
            [
                {
                    "id": "strong",
                    "site": "linkedin",
                    "job_url": "https://example.com/strong",
                    "title": "Senior Product Manager",
                    "company": "Acme",
                    "location": "India",
                    "description": "Own roadmap, pricing, and P&L for enterprise growth.",
                },
                {
                    "id": "weak-engineering",
                    "site": "linkedin",
                    "job_url": "https://example.com/eng",
                    "title": "Software Engineer Intern",
                    "company": "Acme",
                    "location": "India",
                    "description": "Summer internship for backend development.",
                },
                {
                    "id": "weak-intern",
                    "site": "naukri",
                    "job_url": "https://example.com/intern",
                    "title": "Product Intern",
                    "company": "Acme",
                    "location": "India",
                    "description": "Internship supporting product operations.",
                },
            ]
        )

        scored = None
        if hasattr(PUBLIC_API, "score_jobs_for_profile"):
            scored = PUBLIC_API.score_jobs_for_profile(jobs, profile)
        elif hasattr(PUBLIC_API, "rank_jobs_for_profile"):
            scored = PUBLIC_API.rank_jobs_for_profile(jobs, profile)
        else:
            with _patch_scrape_jobs(lambda **kwargs: jobs):
                scored = search_jobs_for_profile(profile, target_sites=["linkedin"])

        self.assertIsInstance(scored, pd.DataFrame)
        self.assertIn("match_score", scored.columns)
        self.assertIn("match_band", scored.columns)
        scores = dict(zip(scored["title"], scored["match_score"]))
        self.assertGreater(scores["Senior Product Manager"], scores["Software Engineer Intern"])
        self.assertGreater(scores["Senior Product Manager"], scores["Product Intern"])

    def test_wrapper_dedupes_jobs_and_preserves_existing_columns(self):
        profile = CandidateProfile(
            resume_text="Senior PM with platform and P&L ownership.",
            target_roles=["product manager"],
            target_sites=["naukri", "linkedin", "indeed"],
        )

        duplicate_row = {
            "id": "same",
            "site": "linkedin",
            "job_url": "https://example.com/jobs/123",
            "title": "Senior Product Manager",
            "company": "Acme",
            "location": "Bengaluru",
            "description": "Own product and P&L.",
            "job_level": "Senior",
        }
        jobs = pd.DataFrame(
            [
                duplicate_row,
                {**duplicate_row, "site": "indeed"},
            ]
        )

        def fake_scrape_jobs(**kwargs):
            return jobs.copy()

        with _patch_scrape_jobs(fake_scrape_jobs):
            result = search_jobs_for_profile(
                profile,
                results_wanted_per_site=5,
                target_sites=["linkedin", "indeed"],
                dedupe=True,
            )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("job_url", result.columns)
        self.assertIn("job_level", result.columns)
        self.assertIn("match_score", result.columns)
        self.assertIn("match_reasons", result.columns)
        self.assertLessEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()