from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from unittest import TestCase
from unittest.mock import patch

from scripts import build_result_release as builder
from scripts import graph_release_api as api
from scripts import portable_result_replay as replay
from test_structural_integration import fixture


def traces_for(instances):
    traces = []
    for instance in instances:
        actions, hashes = [], []
        while (request := api.tasks.next_request(instance, actions)) is not None:
            request = api.canonical_graph_request(instance["task"], request)
            hashes.append(replay.value_hash(asdict(request)))
            actions.append(request.options[0].id)
        traces.append({"id": instance["id"], "actions": actions, "request_sha256": hashes,
                       "score": api.tasks.score(instance, actions)})
    return traces


class PortableResultTests(TestCase):
    def setUp(self):
        self.instances, _ = fixture()
        self.traces = traces_for(self.instances)

    def test_all_ten_tasks_replay_without_clients_or_mutation(self):
        before = deepcopy(self.traces)
        with patch.object(builder.suite, "create_client", side_effect=AssertionError("inference")):
            result = replay.replay_lane(self.instances, self.traces, api)
        self.assertEqual(result["episodes"], 10)
        self.assertEqual(len(result["by_task"]), 10)
        self.assertEqual(result["decisions"], sum(len(t["actions"]) for t in self.traces))
        self.assertEqual(before, self.traces)

    def test_schema_and_scheduled_coverage_are_not_silently_reduced(self):
        with self.assertRaisesRegex(ValueError, "coverage"):
            replay.replay_lane(self.instances, self.traces[:-1], api)
        with self.assertRaisesRegex(ValueError, "coverage"):
            replay.replay_lane([], [], api)
        for change in ("duplicate", "extra"):
            rows = deepcopy(self.traces)
            if change == "duplicate":
                rows[1] = rows[0]
            else:
                rows[0]["provider_body"] = "must not be exported"
            with self.assertRaisesRegex(ValueError, "schema"):
                replay.replay_lane(self.instances, rows, api)

    def test_request_hash_and_score_tampering_refused(self):
        rows = deepcopy(self.traces)
        rows[0]["request_sha256"][0] = "0" * 64
        with self.assertRaisesRegex(ValueError, "request mismatch"):
            replay.replay_lane(self.instances, rows, api)
        rows = deepcopy(self.traces)
        rows[0]["score"]["feasible"] = False
        with self.assertRaisesRegex(ValueError, "score differs"):
            replay.replay_lane(self.instances, rows, api)

    def test_illegal_and_incomplete_actions_are_not_repaired(self):
        rows = deepcopy(self.traces)
        rows[0]["actions"][0] = "not-an-option"
        with self.assertRaisesRegex(ValueError, "Illegal"):
            replay.replay_lane(self.instances, rows, api)
        rows = deepcopy(self.traces)
        rows[0]["actions"].pop()
        rows[0]["request_sha256"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            replay.replay_lane(self.instances, rows, api)

    def test_extra_action_after_completion_refused(self):
        rows = deepcopy(self.traces)
        rows[0]["actions"].append(rows[0]["actions"][0])
        rows[0]["request_sha256"].append(rows[0]["request_sha256"][0])
        with self.assertRaisesRegex(ValueError, "after completion"):
            replay.replay_lane(self.instances, rows, api)

    def test_wrong_legal_answers_remain_in_denominators(self):
        instance = next(i for i in self.instances if i["kind"] == "exact")
        request = api.tasks.next_request(instance, [])
        wrong = next(o.id for o in request.options if o.id != str(instance["private"]["answer"]))
        trace = traces_for([instance])[0]
        trace["actions"] = [wrong]
        trace["score"] = api.tasks.score(instance, [wrong])
        cell = replay.replay_lane([instance], [trace], api)["by_task"][instance["task"]]
        self.assertEqual(cell["instances"], 1)
        self.assertEqual(cell["correct"], 0)
        self.assertEqual(cell["accuracy"], 0)

    def test_size_macro_is_not_pooled_accuracy(self):
        source = next(i for i in self.instances if i["task"] == "adjacency")
        large = deepcopy(source)
        large["id"] += "-larger"
        large["state"]["nodes"].append(len(large["state"]["nodes"]))
        small1, small2 = deepcopy(source), deepcopy(source)
        small2["id"] += "-second"
        instances = [small1, small2, large]
        rows = traces_for(instances)
        for instance, row, correct in zip(instances, rows, (True, True, False)):
            request = api.tasks.next_request(instance, [])
            row["actions"] = [next(o.id for o in request.options
                                   if (o.id == str(instance["private"]["answer"])) == correct)]
            row["score"] = api.tasks.score(instance, row["actions"])
        cell = replay.replay_lane(instances, rows, api)["by_task"][source["task"]]
        self.assertEqual(cell["accuracy"], 2 / 3)
        self.assertEqual(cell["macro_size_accuracy"], .5)

    def test_only_derived_floats_have_absolute_tolerance(self):
        self.assertTrue(replay.same_values(1.0, 1.0 + 1e-13))
        self.assertFalse(replay.same_values(1.0, 1.0 + 1e-10))
        self.assertFalse(replay.same_values(1e12, 1e12 + 1))
        self.assertFalse(replay.same_values(1, True))
        self.assertFalse(replay.same_values(float("nan"), float("nan")))

    def test_export_refuses_mock_and_failed_sources(self):
        for mock, failure in ((True, False), (False, True)):
            manifest = {"profile": "jev_action", "injected_client": mock}
            with patch.object(builder.matrix, "verified_run", return_value=(
                    manifest, {}, [{"status": "failure" if failure else "success"}], [])):
                with self.assertRaises(builder.suite.Refusal):
                    builder.collect_lane(Path("."), Path("."), "original", "development",
                                         "jev_action")

    def test_archive_bytes_are_deterministic_and_exclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bundle"
            root.mkdir()
            (root / "data.json").write_text(json.dumps({"value": 1}))
            a, b = root.parent / "a.tgz", root.parent / "b.tgz"
            builder.archive_bundle(root, a)
            builder.archive_bundle(root, b)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            with self.assertRaises(builder.suite.Refusal):
                builder.archive_bundle(root, a)
            with self.assertRaises(builder.suite.Refusal):
                builder.archive_bundle(root, root / "nested.tgz")

    def test_manifest_and_file_tampering_fail_before_scorer_import(self):
        for tamper in ("manifest", "file", "extra", "path"):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                payload = root / "payload.json"
                payload.write_text("{}")
                manifest = {"schema_version": 1, "protocol": "explicit_actions",
                            "evidence_kind": "verified_completed_real_actions", "lanes": [],
                            "sha256": {"payload.json": replay.digest(payload)}}
                if tamper == "path":
                    manifest["sha256"] = {"../outside.json": "0" * 64}
                path = root / "manifest.json"
                path.write_text(replay.canonical(manifest))
                (root / "manifest.sha256").write_text(json.dumps(replay.digest(path)))
                if tamper == "manifest":
                    path.write_text(path.read_text() + "\n")
                elif tamper == "file":
                    payload.write_text('{"changed":true}')
                elif tamper == "extra":
                    (root / "unregistered.json").write_text("{}")
                with patch.object(replay.importlib, "import_module",
                                  side_effect=AssertionError("premature import")):
                    with self.assertRaisesRegex(ValueError, "integrity|Unexpected"):
                        replay.verify(root)
