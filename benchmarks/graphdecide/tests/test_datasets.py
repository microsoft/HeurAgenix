"""Offline graph-reader and downloader tests using tiny synthetic fixtures.

No downloaded datasets, network, API credentials or model calls are required.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import httpx
import networkx as nx

from src.datasets import available_datasets, get_source, load_graph
from src.datasets import download, sources
from src.datasets.real_graphs import _load_source, _normalize, verify_file


class GraphReaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self, content: bytes, name: str = "facebook"):
        source = replace(get_source(name), sha256=hashlib.sha256(content).hexdigest())
        path = self.root / "raw" / source.filename
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(content)
        return source, path

    def test_catalog_is_explicit_and_pinned(self) -> None:
        self.assertEqual(available_datasets(), ("ca-grqc", "facebook", "human-ppi", "power"))
        for name in available_datasets():
            source = get_source(name)
            self.assertRegex(source.sha256, r"^[0-9a-f]{64}$")
            self.assertTrue(source.download_url.startswith("https://"))
            self.assertFalse(source.directed)
            self.assertFalse(source.weighted)
        for name in ("unknown", "", None):
            with self.subTest(name=name), self.assertRaises(ValueError):
                get_source(name)

    def test_undirected_normalization_counts_and_ids(self) -> None:
        # Source comment is not a reliable direction declaration (as in ca-GrQc).
        source, path = self.fixture(gzip.compress(b"# Directed graph\n9 2\n2 9\n9 2\n50 50\n\n"))
        data = _load_source(source, path)
        self.assertEqual(list(data.graph), [2, 9, 50])
        self.assertEqual(list(data.graph.edges()), [(2, 9)])
        self.assertEqual(data.stats.raw_edge_records, 4)
        self.assertEqual(data.stats.duplicate_edges_removed, 2)
        self.assertEqual(data.stats.self_loops_removed, 1)
        self.assertEqual(data.stats.isolated_nodes, 1)
        self.assertEqual(data.stats.connected_components, 2)
        self.assertEqual(data.raw_sha256, source.sha256)
        self.assertEqual(data.source, source)
        self.assertTrue(nx.is_frozen(data.graph))
        with self.assertRaises(nx.NetworkXError):
            data.graph.add_node(99)
        data.graph.copy().add_node(99)
        self.assertNotIn(99, data.graph)

    def test_csv_has_two_integer_columns_and_no_header(self) -> None:
        source, path = self.fixture(gzip.compress(b"1394,2778\n2778,1394\n7,7\n"), "human-ppi")
        data = _load_source(source, path)
        self.assertEqual(list(data.graph), [7, 1394, 2778])
        self.assertEqual(data.stats.edges, 1)
        self.assertEqual(data.stats.duplicate_edges_removed, 1)
        self.assertEqual(data.stats.self_loops_removed, 1)

    def test_malformed_edge_rows_are_not_silently_skipped(self) -> None:
        for name, row in (
            ("facebook", b"1 2 3"), ("facebook", b"1"),
            ("facebook", b"-1 3"), ("facebook", b"1.5 3"),
            ("human-ppi", b"source,target"), ("human-ppi", b"1,2,3"),
        ):
            with self.subTest(name=name, row=row):
                source, path = self.fixture(gzip.compress(row + b"\n"), name)
                with self.assertRaisesRegex(ValueError, "invalid edge row"):
                    _load_source(source, path)

    def test_gml_retains_explicit_isolates_without_extracting_zip(self) -> None:
        graph = nx.Graph()
        graph.add_nodes_from([0, 1, 2])
        graph.add_edge(0, 1)
        content = io.BytesIO()
        with ZipFile(content, "w") as archive:
            archive.writestr("power.gml", "\n".join(nx.generate_gml(graph)))
            archive.writestr("../should-not-extract.txt", "untrusted extra member")
        source, path = self.fixture(content.getvalue(), "power")
        data = _load_source(source, path)
        self.assertEqual(list(data.graph), [0, 1, 2])
        self.assertEqual(list(data.graph.edges()), [(0, 1)])
        self.assertEqual(data.stats.isolated_nodes, 1)
        self.assertEqual(list(self.root.rglob("*.txt")), [])

    def test_gml_direction_and_weights_are_checked(self) -> None:
        for directed, weighted in ((True, False), (False, True)):
            graph = nx.DiGraph() if directed else nx.Graph()
            graph.add_edge(0, 1, **({"weight": 3} if weighted else {}))
            content = io.BytesIO()
            with ZipFile(content, "w") as archive:
                archive.writestr("power.gml", "\n".join(nx.generate_gml(graph)))
            source, path = self.fixture(content.getvalue(), "power")
            with self.subTest(directed=directed), self.assertRaises(ValueError):
                _load_source(source, path)

    def test_directed_normalizer_does_not_merge_reverse_edges(self) -> None:
        graph, stats = _normalize([(1, 2), (2, 1), (1, 2)], directed=True)
        self.assertTrue(graph.is_directed())
        self.assertEqual(list(graph.edges()), [(1, 2), (2, 1)])
        self.assertEqual(stats.duplicate_edges_removed, 1)

    def test_order_is_independent_of_raw_edge_order(self) -> None:
        rows = [(9, 1), (4, 2), (1, 2), (8, 8)]
        first, _ = _normalize(rows)
        second, _ = _normalize(reversed(rows))
        self.assertEqual(list(first), list(second))
        self.assertEqual(list(first.edges()), list(second.edges()))

    def test_empty_graph_is_supported(self) -> None:
        graph, stats = _normalize([])
        self.assertEqual(len(graph), 0)
        self.assertEqual(stats.connected_components, 0)

    def test_public_loader_accepts_custom_root_without_network(self) -> None:
        source, _ = self.fixture(gzip.compress(b"0 1\n"))
        with patch.dict(sources._SOURCES, {"facebook": source}), patch.object(httpx.Client, "send", side_effect=AssertionError("network")):
            data = load_graph("facebook", data_dir=self.root)
        self.assertEqual(data.stats.edges, 1)

    def test_missing_and_corrupt_files_fail_before_parsing(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_graph("facebook", data_dir=self.root)
        path = self.root / "corrupt.gz"
        path.write_bytes(b"not a gzip archive")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            _load_source(get_source("facebook"), path)

    def test_unsupported_format_and_weights_fail_explicitly(self) -> None:
        source, path = self.fixture(gzip.compress(b"0 1\n"))
        for changed in (replace(source, weighted=True), replace(source, file_format="unknown")):
            with self.subTest(source=changed), self.assertRaises(ValueError):
                _load_source(changed, path)


class DownloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        self.content = gzip.compress(b"0 1\n1 2\n")
        self.source = replace(get_source("facebook"), sha256=hashlib.sha256(self.content).hexdigest())

    def client(self, handler):
        client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(client.close)
        return client

    def test_download_then_verified_cache_does_not_refetch(self) -> None:
        calls = []
        def handler(request):
            calls.append(request)
            return httpx.Response(200, content=self.content)
        client = self.client(handler)
        first = download.download_file(self.source, self.raw, client)
        second = download.download_file(self.source, self.raw, client)
        self.assertEqual(first["status"], "downloaded")
        self.assertIn("downloaded_at_utc", first)
        self.assertEqual(second["status"], "verified-cache")
        self.assertEqual(len(calls), 1)
        self.assertNotIn("authorization", calls[0].headers)
        self.assertEqual(verify_file(self.raw / self.source.filename, self.source.sha256), self.source.sha256)
        self.assertEqual(len(list(self.raw.iterdir())), 1)

    def test_bad_checksum_does_not_install_or_leave_partial(self) -> None:
        client = self.client(lambda _: httpx.Response(200, content=b"wrong content"))
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            download.download_file(self.source, self.raw, client)
        self.assertEqual(list(self.raw.iterdir()), [])

    def test_corrupt_existing_file_is_not_overwritten(self) -> None:
        self.raw.mkdir()
        path = self.raw / self.source.filename
        path.write_bytes(b"existing bad file")
        client = self.client(lambda _: self.fail("Should not download over existing file"))
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            download.download_file(self.source, self.raw, client)
        self.assertEqual(path.read_bytes(), b"existing bad file")

    def test_http_error_has_no_retries_or_partial_file(self) -> None:
        calls = []
        def handler(request):
            calls.append(request)
            return httpx.Response(503)
        with self.assertRaises(httpx.HTTPStatusError):
            download.download_file(self.source, self.raw, self.client(handler))
        self.assertEqual(len(calls), 1)
        self.assertEqual(list(self.raw.iterdir()), [])

    def test_https_redirect_is_recorded(self) -> None:
        def handler(request):
            if request.url.path != "/archive":
                return httpx.Response(302, headers={"location": "https://example.org/archive"})
            return httpx.Response(200, content=self.content)
        receipt = download.download_file(self.source, self.raw, self.client(handler))
        self.assertEqual(receipt["resolved_download_url"], "https://example.org/archive")

    def test_http_downgrade_is_rejected_before_following(self) -> None:
        calls = []
        def handler(request):
            calls.append(request)
            return httpx.Response(302, headers={"location": "http://example.org/archive"})
        with self.assertRaisesRegex(ValueError, "non-HTTPS"):
            download.download_file(self.source, self.raw, self.client(handler))
        self.assertEqual(len(calls), 1)
        self.assertEqual(list(self.raw.iterdir()), [])

    def test_redirect_loop_is_bounded(self) -> None:
        calls = []
        def handler(request):
            calls.append(request)
            return httpx.Response(302, headers={"location": str(request.url)})
        with self.assertRaisesRegex(ValueError, "Too many"):
            download.download_file(self.source, self.raw, self.client(handler))
        self.assertEqual(len(calls), 6)

    def test_size_limit_cleans_partial_file(self) -> None:
        client = self.client(lambda _: httpx.Response(200, content=self.content))
        with patch.object(download, "MAX_DOWNLOAD_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "size limit"):
                download.download_file(self.source, self.raw, client)
        self.assertEqual(list(self.raw.iterdir()), [])

    def test_interrupted_stream_cleans_partial_file(self) -> None:
        class BrokenStream(httpx.SyncByteStream):
            def __iter__(self):
                yield b"partial"
                raise httpx.ReadError("fixture disconnect")
        client = self.client(lambda _: httpx.Response(200, stream=BrokenStream()))
        with self.assertRaises(httpx.ReadError):
            download.download_file(self.source, self.raw, client)
        self.assertEqual(list(self.raw.iterdir()), [])

    def test_cli_manifest_and_verify_only_are_reproducible(self) -> None:
        client = self.client(lambda _: httpx.Response(200, content=self.content))
        with patch.dict(sources._SOURCES, {"facebook": self.source}), contextlib.redirect_stdout(io.StringIO()):
            with patch.object(download.httpx, "Client", return_value=client):
                download.main(["--datasets", "facebook", "--data-dir", str(self.root)])
            path = self.root / "manifest.json"
            before = path.read_bytes()
            record = json.loads(before)["datasets"]["facebook"]
            self.assertEqual(record["sha256"], self.source.sha256)
            self.assertEqual(record["stats"]["nodes"], 3)
            self.assertEqual(record["raw_file"], "raw/facebook_combined.txt.gz")
            with patch.object(httpx.Client, "send", side_effect=AssertionError("network")):
                download.main(["--datasets", "facebook", "--data-dir", str(self.root), "--verify-only"])
                self.assertEqual(path.read_bytes(), before)
                download.main(["--datasets", "facebook", "--data-dir", str(self.root)])
                self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()