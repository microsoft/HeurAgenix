"""Pinned first-party releases for the initial real-network collection."""

from dataclasses import dataclass
from pathlib import Path


# Source-checkout default; callers may supply an explicit data_dir elsewhere.
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "real"


@dataclass(frozen=True, slots=True)
class DatasetSource:
    name: str
    domain: str
    source_url: str
    download_url: str
    filename: str
    file_format: str
    sha256: str
    citation: str
    published_nodes: int
    published_edges: int
    directed: bool = False
    weighted: bool = False
    license_note: str = (
        "No explicit redistribution license verified on the dataset page. "
        "Retain source attribution; check upstream terms before redistributing. "
        "This dataset does not inherit the project's code license."
    )


_SOURCES = {
    "facebook": DatasetSource(
        name="facebook", domain="social",
        source_url="https://snap.stanford.edu/data/ego-Facebook.html",
        download_url="https://snap.stanford.edu/data/facebook_combined.txt.gz",
        filename="facebook_combined.txt.gz", file_format="edgelist-gzip",
        sha256="125e84db872eeba443d270c70315c256b0af43a502fcfe51f50621166ad035d7",
        citation="J. McAuley and J. Leskovec. Learning to Discover Social Circles in Ego Networks. NIPS, 2012.",
        published_nodes=4039, published_edges=88234,
    ),
    "ca-grqc": DatasetSource(
        name="ca-grqc", domain="collaboration",
        source_url="https://snap.stanford.edu/data/ca-GrQc.html",
        download_url="https://snap.stanford.edu/data/ca-GrQc.txt.gz",
        filename="ca-GrQc.txt.gz", file_format="edgelist-gzip",
        sha256="a254442cdf5d684712578b630c2e0d7543518ab154ef2341cabb607572ce7230",
        citation="J. Leskovec, J. Kleinberg and C. Faloutsos. Graph Evolution: Densification and Shrinking Diameters. ACM TKDD, 2007.",
        published_nodes=5242, published_edges=14496,
    ),
    "power": DatasetSource(
        name="power", domain="infrastructure",
        source_url="https://websites.umich.edu/~mejn/netdata/",
        download_url="https://websites.umich.edu/~mejn/netdata/power.zip",
        filename="power.zip", file_format="power-gml-zip",
        sha256="866f2301d113928973f40c8e1ca34be84efa2ad8209cd9d0cf7c16e972167b7d",
        citation="D. J. Watts and S. H. Strogatz. Collective dynamics of small-world networks. Nature 393, 440–442, 1998.",
        published_nodes=4941, published_edges=6594,
    ),
    "human-ppi": DatasetSource(
        name="human-ppi", domain="biology",
        source_url="https://snap.stanford.edu/biodata/datasets/10000/10000-PP-Pathways.html",
        download_url="https://snap.stanford.edu/biodata/datasets/10000/files/PP-Pathways_ppi.csv.gz",
        filename="PP-Pathways_ppi.csv.gz", file_format="csv-gzip",
        sha256="a33d7236b90b6f6cfcec11d0e2c5be41867a0bb547a3c4341abf5f9091e62342",
        citation="M. Agrawal, M. Zitnik and J. Leskovec. Large-scale analysis of disease pathways in the human interactome. PSB, 2018.",
        published_nodes=21557, published_edges=342353,
    ),
}


def available_datasets() -> tuple[str, ...]:
    return tuple(sorted(_SOURCES))


def get_source(name: str) -> DatasetSource:
    """Return an exact registered ID; do not guess between dataset versions."""
    if not isinstance(name, str) or name not in _SOURCES:
        raise ValueError(f"Unknown dataset {name!r}; choose from {available_datasets()}")
    return _SOURCES[name]