"""Search Hugging Face and Kaggle for candidate datasets and sort them by license.

Discovery only reports. It never downloads data and never edits the registry: a person reads the license and
the data card, then adds a reviewed ``DatasetEntry`` by hand. A hub that is not installed, has no credentials
or cannot be reached is skipped with its reason, so discovery never blocks a run.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..workspace import Workspace, write_text_atomic

logger = logging.getLogger(__name__)

SEARCH_QUERIES: tuple[str, ...] = (
    "driver drowsiness",
    "drowsiness detection",
    "driver fatigue",
    "eye closure",
    "eye blink",
    "yawn detection",
    "driver monitoring",
    "distracted driver",
)

COMMERCIAL_LICENSES = frozenset(
    {
        "mit",
        "apache-2.0",
        "bsd-2-clause",
        "bsd-3-clause",
        "cc0-1.0",
        "cc0:-public-domain",
        "cc-by-sa-4.0",
        "cc-by-sa-3.0",
        "odc-by",
        "odc-by-1.0",
        "odbl",
        "odbl-1.0",
        "pddl",
        "cdla-permissive-1.0",
        "cdla-permissive-2.0",
    }
)
_UNKNOWN_LICENSES = frozenset({"other", "unknown", "unspecified"})
_RESTRICTIVE_MARKERS = ("-nc", "noncommercial", "non-commercial", "-nd", "research")
_COMMERCIAL_PREFIXES = ("cc-by-4", "cc-by-3")

REPORT_COLUMNS: tuple[str, ...] = (
    "source",
    "dataset_id",
    "license",
    "commercial_ok",
    "downloads",
    "queries",
    "url",
)
REPORT_FILE = "discovered_datasets.csv"


@dataclass(frozen=True)
class DatasetHit:
    source: str
    dataset_id: str
    license: str | None
    downloads: int | None
    query: str
    url: str


@dataclass(frozen=True)
class DiscoveryResult:
    report_path: Path
    status_by_hub: dict[str, str]
    datasets_found: int


def normalize_license(value: Any) -> str | None:
    if isinstance(value, list | tuple):
        value = value[0] if value else None
    if value is None:
        return None
    normalized = str(value).strip().lower().replace(" ", "-")
    return normalized or None


def classify_license(value: Any) -> bool | None:
    """True when the license allows commercial use, False when it forbids it, None when nobody can tell."""
    license_name = normalize_license(value)
    if license_name is None or license_name in _UNKNOWN_LICENSES:
        return None
    if any(marker in license_name for marker in _RESTRICTIVE_MARKERS):
        return False
    if license_name in COMMERCIAL_LICENSES or license_name.startswith(_COMMERCIAL_PREFIXES):
        return True
    return None


def _hugging_face_license(dataset: Any) -> str | None:
    for tag in getattr(dataset, "tags", None) or []:
        if str(tag).startswith("license:"):
            return str(tag).split(":", 1)[1]
    return None


def search_hugging_face(client: Any, queries: Iterable[str], limit: int = 40) -> list[DatasetHit]:
    hits = []
    for query in queries:
        for dataset in client.list_datasets(search=query, limit=limit, full=True):
            hits.append(
                DatasetHit(
                    source="hugging_face",
                    dataset_id=str(dataset.id),
                    license=_hugging_face_license(dataset),
                    downloads=getattr(dataset, "downloads", None),
                    query=query,
                    url=f"https://huggingface.co/datasets/{dataset.id}",
                )
            )
    return hits


def search_kaggle(client: Any, queries: Iterable[str]) -> list[DatasetHit]:
    hits = []
    for query in queries:
        for dataset in client.dataset_list(search=query) or []:
            dataset_id = str(getattr(dataset, "ref", None) or dataset)
            hits.append(
                DatasetHit(
                    source="kaggle",
                    dataset_id=dataset_id,
                    license=getattr(dataset, "licenseName", None) or getattr(dataset, "license_name", None),
                    downloads=getattr(dataset, "downloadCount", None) or getattr(dataset, "download_count", None),
                    query=query,
                    url=f"https://www.kaggle.com/datasets/{dataset_id}",
                )
            )
    return hits


def _license_rank(commercial_ok: bool | None) -> int:
    return {True: 0, None: 1, False: 2}[commercial_ok]


def summarize_hits(hits: Sequence[DatasetHit]) -> pd.DataFrame:
    """One row per dataset: commercially usable first, then unknown, then restricted; most downloaded first."""
    if not hits:
        return pd.DataFrame(columns=list(REPORT_COLUMNS))
    frame = pd.DataFrame(
        [
            {
                "source": hit.source,
                "dataset_id": hit.dataset_id,
                "license": hit.license,
                "downloads": hit.downloads,
                "query": hit.query,
                "url": hit.url,
            }
            for hit in hits
        ]
    )
    frame["downloads"] = pd.to_numeric(frame["downloads"], errors="coerce")
    summary = frame.groupby(["source", "dataset_id"], as_index=False, sort=False).agg(
        license=("license", "first"),
        downloads=("downloads", "max"),
        queries=("query", lambda values: ", ".join(sorted(set(values)))),
        url=("url", "first"),
    )
    summary["commercial_ok"] = [classify_license(value) for value in summary["license"]]
    summary["rank"] = [_license_rank(value) for value in summary["commercial_ok"]]
    summary = summary.sort_values(
        ["rank", "downloads"],
        ascending=[True, False],
        na_position="last",
        kind="stable",
    )
    return summary[list(REPORT_COLUMNS)].reset_index(drop=True)


def _hugging_face_client() -> Any:
    from huggingface_hub import HfApi

    return HfApi(token=os.environ.get("HF_TOKEN"))


def _kaggle_client() -> Any:
    try:
        from google.colab import userdata
    except ImportError:
        pass
    else:
        for name in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
            try:
                os.environ[name] = userdata.get(name)
            except (userdata.SecretNotFoundError, userdata.NotebookAccessError):
                logger.info("Colab secret %s is not set or not shared with this notebook", name)
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def run_discovery(workspace: Workspace, queries: Sequence[str] = SEARCH_QUERIES) -> DiscoveryResult:
    hubs: tuple[
        tuple[str, Callable[[], Any], Callable[[Any, Sequence[str]], list[DatasetHit]]],
        ...,
    ] = (
        ("hugging_face", _hugging_face_client, search_hugging_face),
        ("kaggle", _kaggle_client, search_kaggle),
    )
    hits: list[DatasetHit] = []
    status_by_hub: dict[str, str] = {}
    for hub, build_client, search in hubs:
        try:
            found = search(build_client(), queries)
        except Exception as exc:  # noqa: BLE001 - a missing library, credential or network must not stop the run
            status_by_hub[hub] = f"skipped ({type(exc).__name__}: {str(exc)[:100]})"
            logger.warning("discovery: %s %s", hub, status_by_hub[hub])
            continue
        hits.extend(found)
        status_by_hub[hub] = f"{len(found)} hits"
        logger.info("discovery: %s returned %d hits", hub, len(found))
    table = summarize_hits(hits)
    report_path = write_text_atomic(workspace.reports_dir / REPORT_FILE, table.to_csv(index=False))
    logger.info(
        "discovery: %d datasets listed in %s (nothing downloaded)",
        len(table),
        report_path,
    )
    return DiscoveryResult(report_path=report_path, status_by_hub=status_by_hub, datasets_found=len(table))
