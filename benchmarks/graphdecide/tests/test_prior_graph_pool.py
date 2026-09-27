import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import unittest
from unittest import mock
import uuid


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "collect_prior_graphs.py"
SPEC = importlib.util.spec_from_file_location("prior_graph_pool", SCRIPT)
pool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pool)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def graph(edges=None, **extra):
    return {"nodes": list(range(8)), "edges": edges or [], **extra}


def check_renderers_deduplicate_and_record_every_source_row(workspace):
    scan = workspace / "source"
    edges = [[0, 1], [1, 2]]
    adjacency = {str(i): [] for i in range(8)}
    adjacency["0"] = [1]
    adjacency["1"] = [2, 0]
    adjacency["2"] = [1]
    rows = [
        {"state": graph(edges, vertex=1, history=["not graph data"])},
        {"request": {"state": {"nodes": list(range(8)), "adjacency": adjacency}}},
        {
            "state": graph(
                "Node 1 and node 0 are joined by an undirected edge.\n"
                "Node 2 and node 1 are joined by an undirected edge."
            )
        },
    ]
    source = scan / "cycle" / "requests.jsonl"
    write_rows(source, rows)
    output = scan / "fresh"
    manifest = pool.collect_prior_graphs(scan, output)
    assert manifest["status"] == "completed"
    assert manifest["unique_graphs"] == 1
    assert manifest["coverage"]["accepted_rows"] == 3
    assert len(next(iter(manifest["provenance"].values()))) == 3
    record = json.loads((output / "graph-pool.jsonl").read_text())
    assert set(record) == {"id", "task", "state"}
    assert set(record["state"]) == {"nodes", "edges", "directed"}
    assert manifest["sources"][0]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def check_size_filter_weighted_directed_and_scalar_remapping(workspace):
    scan = workspace / "source"
    names = [f"node{i}" for i in range(8)]
    write_rows(
        scan / "instances.jsonl",
        [
            {"state": {"nodes": names, "edges": [["node7", "node0"]]}},
            {"state": graph([[0, 1, 2.5]], points=[[1, 2]] * 8, metric="x")},
            {"state": graph([[0, 1]], directed=True, threshold={"numerator": 1})},
            {"state": {"nodes": list(range(7)), "edges": []}},
            {"state": {"nodes": list(range(13)), "edges": []}},
        ],
    )
    output = workspace / "pool"
    manifest = pool.collect_prior_graphs(scan, output)
    assert manifest["unique_graphs"] == 3
    assert manifest["coverage"]["outside_size_rows"] == 2
    records = [json.loads(line) for line in (output / "graph-pool.jsonl").read_text().splitlines()]
    by_task = {record["task"]: record for record in records}
    assert by_task["degree_exact"]["state"]["edges"] == [[0, 7]]
    assert by_task["tsp_construct"]["state"]["edges"] == [[0, 1, 2.5]]
    assert by_task["lt_influence_construct"]["state"]["directed"] is True


INVALID_STATES = [
        graph([[0, 0]]),
        graph([[0, 1], [1, 0]]),
        graph([[0, 9]]),
        graph([[0, 1, float("nan")]]),
        graph([[0, 1], [1, 2, 3]]),
        graph("0 is connected to 1"),
        {"nodes": [0] * 8, "edges": []},
        {"nodes": list(range(8)), "adjacency": {"0": [1]}},
        {"nodes": list(range(8)), "adjacency": {str(i): [1] if i == 0 else [] for i in range(8)}},
        {"nodes": list(range(8)), "edges": [], "adjacency": {}},
    ]


def check_invalid_candidate_is_explicit_and_never_published(workspace, state):
    scan = workspace / "source"
    write_rows(scan / "requests.jsonl", [{"state": state}])
    output = workspace / "pool"
    manifest = pool.collect_prior_graphs(scan, output)
    assert manifest["status"] == "rejected_unsupported_rows"
    assert manifest["coverage"]["unsupported_rows"] == 1
    assert manifest["unsupported"][0]["line"] == 1
    assert not (output / "graph-pool.jsonl").exists()


def check_unknown_graph_size_and_invalid_json_are_not_silently_skipped(workspace):
    scan = workspace / "source"
    write_rows(scan / "requests.jsonl", [{"state": {"edges": [[0, 1]]}}])
    with (scan / "requests.jsonl").open("a") as stream:
        stream.write("{invalid\n")
    manifest = pool.collect_prior_graphs(scan, workspace / "pool")
    assert manifest["coverage"]["unsupported_rows"] == 2
    assert len(manifest["unsupported"]) == 2


def check_exact_labeled_dedup_does_not_replace_isomorphism_exclusion(workspace):
    scan = workspace / "source"
    write_rows(
        scan / "requests.jsonl",
        [{"state": graph([[0, 1]])}, {"state": graph([[0, 2]])}],
    )
    manifest = pool.collect_prior_graphs(scan, workspace / "pool")
    assert manifest["unique_graphs"] == 2


class PriorGraphPoolTests(unittest.TestCase):
    def setUp(self):
        self.workspace = Path.cwd() / f".prior-pool-test-{uuid.uuid4().hex}"
        self.workspace.mkdir()
        self.addCleanup(shutil.rmtree, self.workspace)

    def test_renderers_and_provenance(self):
        check_renderers_deduplicate_and_record_every_source_row(self.workspace)

    def test_weights_direction_and_sizes(self):
        check_size_filter_weighted_directed_and_scalar_remapping(self.workspace)

    def test_invalid_candidates_fail_closed(self):
        for index, state in enumerate(INVALID_STATES):
            with self.subTest(index=index):
                workspace = self.workspace / str(index)
                workspace.mkdir()
                check_invalid_candidate_is_explicit_and_never_published(workspace, state)

    def test_refuses_existing_target(self):
        scan = self.workspace / "source"
        scan.mkdir()
        with self.assertRaises(FileExistsError):
            pool.collect_prior_graphs(scan, scan)

    def test_changed_source_fails_before_publication(self):
        scan = self.workspace / "source"
        write_rows(scan / "requests.jsonl", [{"state": graph()}])
        output = self.workspace / "pool"
        with mock.patch.object(pool, "file_sha256", return_value="not-the-source-hash"):
            with self.assertRaisesRegex(RuntimeError, "source changed"):
                pool.collect_prior_graphs(scan, output)
        self.assertFalse(output.exists())

    def test_unreadable_candidates(self):
        check_unknown_graph_size_and_invalid_json_are_not_silently_skipped(self.workspace)

    def test_labeled_not_isomorphism_dedup(self):
        check_exact_labeled_dedup_does_not_replace_isomorphism_exclusion(self.workspace)


if __name__ == "__main__":
    unittest.main()
