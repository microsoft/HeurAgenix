import asyncio
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from unittest import TestCase
from unittest.mock import AsyncMock
import uuid

from scripts import graph_invariance_controls as controls
from src.benchmark import extended_tasks as tasks
from src.clients.base import DecisionResponse, InvalidResponseError, ProviderHTTPError, TokenUsage


def fixture_bundle(root):
    root.mkdir()
    instances, metadata = [], {"cases": {}}
    for task in controls.TASKS:
        for n in controls.SIZES:
            if task == "degree_exact":
                candidates = []
                for degree in [k * (n - 1) // 3 for k in range(4)]:
                    for copy in ("a", "b"):
                        candidates.append((f"degree-{degree}-{copy}",
                                           {"edges": [[0, v] for v in range(1, degree + 1)], "vertex": 0},
                                           None))
            else:
                candidates = []
                for pair in ("a", "b", "c"):
                    edges = [[v, v + 1] for v in range(n - 1)]
                    if task == "adjacency":
                        states = [{"edges": edges, "pair": [0, 1]}, {"edges": edges, "pair": [0, 2]}]
                    elif task == "cycle_detection":
                        states = [{"edges": edges + [[0, 2]]}, {"edges": edges}]
                    elif task == "pair_connectivity":
                        states = [{"edges": [[0, 1]], "pair": [0, 1]},
                                  {"edges": [[0, 1]], "pair": [0, 2]}]
                    elif task == "distance_threshold":
                        states = [{"edges": edges, "pair": [0, 2], "threshold": 2},
                                  {"edges": edges, "pair": [0, 3], "threshold": 2}]
                    else:
                        states = [{"edges": edges, "vertex": 1}, {"edges": edges, "vertex": 0}]
                    candidates.extend((f"pair-{pair}-{index}", state, f"{task}-n{n}-pair-{pair}")
                                      for index, state in enumerate(states))
            for suffix, state, pair in candidates:
                state = {"nodes": list(range(n)), "directed": False, **state}
                state["edges"] = sorted(state["edges"])
                instance = tasks._record(task, 0, state)
                instance["id"] = f"{task}-n{n}-{suffix}"
                instances.append(instance)
                graph_key = json.dumps({"nodes": state["nodes"], "edges": state["edges"]}, sort_keys=True)
                metadata["cases"][instance["id"]] = {
                    "cluster_id": controls.hashlib.sha256(graph_key.encode()).hexdigest(),
                    "pair_id": pair,
                }
    for directory in ("public", "references", "analysis"):
        (root / directory).mkdir()
    split = controls.SPLIT
    controls.write_rows(root / f"public/{split}.jsonl",
                        [{k: i[k] for k in controls.release.PUBLIC_KEYS} for i in instances])
    controls.write_rows(root / f"references/{split}.jsonl",
                        [{"id": i["id"], "private": i["private"]} for i in instances])
    controls.suite.write_json(root / f"analysis/{split}.json", metadata)
    manifest = {"schema_version": 1, "protocols": ["explicit_actions"],
                "splits": {split: {"validation": tasks.validate_instances(instances)}},
                "sha256": {f"{folder}/{split}.{extension}": controls.suite.digest(
                    root / f"{folder}/{split}.{extension}")
                    for folder, extension in (("public", "jsonl"), ("references", "jsonl"), ("analysis", "json"))}}
    controls.suite.write_json(root / "manifest.json", manifest)
    controls.suite.write_json(root / "manifest.sha256", controls.suite.digest(root / "manifest.json"))
    return instances, metadata


class FakeClient:
    def __init__(self, model):
        self.model = model
        self.calls = 0

    async def initialize(self):
        pass

    async def aclose(self):
        pass

    async def predict(self, request):
        self.calls += 1
        return DecisionResponse(request.request_id, request.options[0].id, self.model)


class InvarianceTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = controls.DEFAULT_ROOT / "test-artifacts" / uuid.uuid4().hex
        cls.base.mkdir(parents=True)
        cls.bundle = cls.base / "bundle"
        try:
            cls.instances, cls.metadata = fixture_bundle(cls.bundle)
        except BaseException:
            shutil.rmtree(cls.base)
            raise
        cls.public = [{k: i[k] for k in controls.release.PUBLIC_KEYS} for i in cls.instances]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.base)

    def setUp(self):
        self.root = self.base / self._testMethodName

    def prepared(self):
        controls.prepare(self.root, self.bundle)
        return self.root

    def execute(self, client=FakeClient, profile="qwen4_grammar", lock=None):
        self.prepared()
        return asyncio.run(controls.run_profile(
            self.root, profile, client_factory=lambda _, **kw: client(kw["model"]),
            preflight=AsyncMock(), lock_factory=lock or (lambda _: nullcontext())))

    def test_selection_is_balanced_complete_pairs_and_opaque_id_first(self):
        selected = controls.select_cases(self.public, self.metadata)
        self.assertEqual(len(selected), 120)
        self.assertEqual(selected, controls.select_cases(list(reversed(self.public)), self.metadata))
        for task in controls.TASKS:
            for n in controls.SIZES:
                group = [i for i in selected if i["task"] == task and len(i["state"]["nodes"]) == n]
                self.assertEqual(len(group), 4)
                if task == "degree_exact":
                    self.assertEqual([controls.truth(task, i["state"]) for i in group],
                                     [str(k * (n - 1) // 3) for k in range(4)])
                    self.assertTrue(all(i["id"].endswith("-a") for i in group))
                else:
                    self.assertEqual({self.metadata["cases"][i["id"]]["pair_id"] for i in group},
                                     {f"{task}-n{n}-pair-a", f"{task}-n{n}-pair-b"})
                    self.assertEqual(sorted(controls.truth(task, i["state"]) for i in group),
                                     ["no", "no", "yes", "yes"])

    def test_selection_refuses_incomplete_pairs_instead_of_replenishing(self):
        pool = [i for i in self.public if not (i["task"] == "adjacency"
                and len(i["state"]["nodes"]) == 8 and not i["id"].endswith("pair-a-0"))]
        with self.assertRaisesRegex(ValueError, "complete binary pairs"):
            controls.select_cases(pool, self.metadata)

    def test_all_transformed_truth_labels_equal_references(self):
        cases, requests, _ = controls.plan_inputs(self.bundle)
        labels = {c["instance_id"]: c["label"] for c in cases}
        tasks_by_id = {c["instance_id"]: c["task"] for c in cases}
        self.assertEqual(len(requests), 600)
        for row in requests:
            self.assertEqual(controls.truth(tasks_by_id[row["instance_id"]], row["request"]["state"]),
                             labels[row["instance_id"]])

    def test_node_mapping_preserves_edge_positions_and_query_semantics(self):
        for task in controls.TASKS:
            instance = next(i for i in self.instances if i["task"] == task)
            request, details = controls.transform(instance, "shared-root", "node_relabel")
            mapping = {int(k): v for k, v in details["node_mapping"].items()}
            self.assertEqual(request.state["edges"],
                             [[mapping[u], mapping[v]] for u, v in instance["state"]["edges"]])
            self.assertEqual(request.state["nodes"], list(range(len(mapping))))
            if "pair" in instance["state"]:
                self.assertEqual(request.state["pair"], [mapping[v] for v in instance["state"]["pair"]])
            if "vertex" in instance["state"]:
                self.assertEqual(request.state["vertex"], mapping[instance["state"]["vertex"]])
            original = controls.transform(instance, "shared-root", "original")[0]
            self.assertEqual(request.options, original.options)
            self.assertEqual(request.question, original.question)

    def test_shared_graph_queries_share_exact_transformations(self):
        pair = [i for i in self.instances if i["task"] == "adjacency" and "-n8-pair-a-" in i["id"]]
        for arm in ("node_relabel", "edge_shuffle"):
            left, ld = controls.transform(pair[0], "graph", arm)
            right, rd = controls.transform(pair[1], "graph", arm)
            self.assertEqual(ld, rd)
            self.assertEqual(left.state["edges"], right.state["edges"])

    def test_shuffle_changes_only_edge_position_order(self):
        instance = self.instances[0]
        original = controls.transform(instance, "graph", "original")[0]
        shuffled, details = controls.transform(instance, "graph", "edge_shuffle")
        self.assertEqual(shuffled.state["edges"],
                         [original.state["edges"][i] for i in details["edge_index_order"]])
        self.assertEqual({k: v for k, v in shuffled.state.items() if k != "edges"},
                         {k: v for k, v in original.state.items() if k != "edges"})

    def test_noop_interventions_are_retained_not_resampled(self):
        instance = deepcopy(self.instances[0])
        instance["task"] = "cycle_detection"
        instance["state"] = {"nodes": list(range(8)), "edges": [], "directed": False}
        original = controls.transform(instance, "root", "original")[0]
        self.assertEqual(original, controls.transform(instance, "root", "node_relabel")[0])
        self.assertEqual(original, controls.transform(instance, "root", "edge_shuffle")[0])
        instance["state"]["edges"] = [[0, 1]]
        self.assertEqual(controls.transform(instance, "root", "original")[0],
                         controls.transform(instance, "root", "edge_shuffle")[0])
        _, requests, _ = controls.plan_inputs(self.bundle)
        self.assertTrue(any(r["arm"] == "edge_shuffle" and not r["input_changed"] for r in requests))

    def test_reversed_options_keep_semantic_ids_and_descriptions(self):
        for task in ("degree_exact", "adjacency"):
            instance = next(i for i in self.instances if i["task"] == task)
            original = controls.transform(instance, "root", "original")[0]
            reversed_request = controls.transform(instance, "root", "reversed_options")[0]
            self.assertEqual(original.options[::-1], reversed_request.options)
            self.assertEqual(original.state, reversed_request.state)
            self.assertEqual(original.question, reversed_request.question)
            self.assertEqual(original.request_id, reversed_request.request_id)

    def test_identical_repeat_and_private_separation(self):
        instance = deepcopy(self.instances[0])
        instance["private"] = {"answer": "SECRET_LABEL", "reference": "DO_NOT_SEND"}
        instance["state"]["secret_private_oracle"] = "DO_NOT_SEND"
        original = controls.transform(instance, "root", "original")[0]
        repeat = controls.transform(instance, "root", "identical_repeat")[0]
        self.assertEqual(asdict(original), asdict(repeat))
        serialized = json.dumps(asdict(original))
        self.assertNotIn("DO_NOT_SEND", serialized)
        self.assertNotIn("SECRET_LABEL", serialized)
        self.assertNotIn("secret_private_oracle", serialized)

    def test_explicit_dependency_closure_does_not_pin_unrelated_helpers(self):
        hashes = controls.source_hashes()
        for name in ("scripts/graph_invariance_controls.py", "scripts/evaluate_graph_release.py",
                     "scripts/semif_vllm_bridge.py", "src/clients/typesafe.py",
                     "src/clients/vllm_client.py", "src/clients/_llm_choice.py"):
            self.assertIn(name, hashes)
        self.assertNotIn("scripts/run_paper_gpt_anchor.py", hashes)
        self.assertNotIn("tests/test_graph_invariance_controls.py", hashes)

    def test_prepare_verify_and_input_tamper_refusal(self):
        self.prepared()
        registration, cases, requests = controls.verify(self.root)
        self.assertEqual(registration["total_calls"], 2400)
        self.assertEqual(len(cases), 120)
        self.assertEqual(len(requests), 600)
        with (self.root / "requests.jsonl").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "input hash drift"):
            controls.verify(self.root)

    def test_recomputed_plan_rejects_forged_request_even_after_rehash(self):
        self.prepared()
        requests = controls.suite.read_rows(self.root / "requests.jsonl")
        requests[1]["request"]["question"] += " SECRET"
        (self.root / "requests.jsonl").write_text("".join(
            controls.suite.canonical(r) + "\n" for r in requests))
        registration = controls.suite.read_json(self.root / "registration.json")
        registration["input_sha256"]["requests.jsonl"] = controls.suite.digest(self.root / "requests.jsonl")
        (self.root / "registration.json").chmod(0o644)
        (self.root / "registration.json").write_text(controls.suite.canonical(registration) + "\n")
        (self.root / "registration.sha256").write_text(json.dumps(
            controls.suite.digest(self.root / "registration.json")) + "\n")
        with self.assertRaisesRegex(ValueError, "replay differs"):
            controls.verify(self.root)

    def test_mock_run_replays_all_calls_and_cannot_be_paper_results(self):
        result = self.execute()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["calls"], 600)
        _, _, scores, summary = controls.verify_run(self.root, "qwen4_grammar", allow_injected=True)
        self.assertEqual(len(scores), 600)
        originals = [r for r in summary["accuracy"] if r["node_count"] == "all" and r["arm"] == "original"]
        self.assertTrue(all(r["scheduled"] == r["valid"] == 20 for r in originals))
        repeats = [r for r in summary["paired_choice_changes"]
                   if r["arm"] == "identical_repeat" and r["actual_input_changed"] == "all"]
        self.assertTrue(all(r["choice_flips"] == 0 and r["both_valid"] == 20 for r in repeats))
        self.assertTrue(any(r["invariant_wrong"] for r in repeats))
        self.assertTrue(any(r["robust_correct"] for r in repeats))
        with self.assertRaisesRegex(ValueError, "not paper results"):
            controls.verify_run(self.root, "qwen4_grammar")

    def test_fatal_failure_records_all_unattempted_denominators(self):
        class Fatal(FakeClient):
            async def predict(self, request):
                raise ProviderHTTPError(401)
        result = self.execute(Fatal)
        self.assertEqual(result["status"], "aborted")
        self.assertEqual(result["calls"], 1)
        _, _, scores, summary = controls.verify_run(self.root, "qwen4_grammar", allow_injected=True)
        self.assertEqual(sum(s["attempted"] for s in scores), 1)
        self.assertEqual(sum(s["valid"] for s in scores), 0)
        self.assertTrue(all(r["both_valid"] == 0 and r["flip_rate_both_valid"] is None
                            for r in summary["paired_choice_changes"]))

    def test_nonfatal_failure_does_not_retry_and_excludes_pair_denominator(self):
        class Once(FakeClient):
            async def predict(self, request):
                if self.calls == 0:
                    self.calls += 1
                    raise ProviderHTTPError(503)
                return await super().predict(request)
        result = self.execute(Once)
        self.assertEqual(result["status"], "completed_with_failures")
        self.assertEqual(result["calls"], 600)
        _, _, scores, summary = controls.verify_run(self.root, "qwen4_grammar", allow_injected=True)
        self.assertEqual(sum(s["valid"] for s in scores), 599)
        repeat = next(r for r in summary["paired_choice_changes"] if
                      r["task"] == "adjacency" and r["arm"] == "identical_repeat"
                      and r["actual_input_changed"] == "all")
        self.assertEqual(repeat["both_valid"], 19)
        self.assertEqual(repeat["not_both_valid"], 1)

    def test_native_conformance_separate_from_supplied_action(self):
        class Native(FakeClient):
            async def predict(self, request):
                ids = [o.id for o in request.options]
                body = {"model": self.model, "secret": "NEVER_EXPORT",
                        "answers": {"decision": {"type": "choice", "choice": ids[0],
                                    "probabilities": {i: 1 / len(ids) + .01 for i in ids},
                                    "confidence": .5}}, "usage": {}}
                raise InvalidResponseError("sum", diagnostic_code="invalid_probability_sum", raw_output=body)
        result = self.execute(Native, "jev_action")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["strict_native_failures"], 600)
        _, _, scores, _ = controls.verify_run(self.root, "jev_action", allow_injected=True)
        self.assertTrue(all(s["valid"] for s in scores))
        self.assertNotIn("NEVER_EXPORT", (self.root / "jev_action/attempts.jsonl").read_text())

    def test_token_profile_does_not_acquire_second_outer_lock(self):
        class Tokens(FakeClient):
            async def predict(self, request):
                ids = [o.id for o in request.options]
                data = {"selected_option_id": ids[0], "candidate_count": len(ids),
                        "candidate_prompts": 1, "generated_tokens": 1,
                        "generated_tokens_used_for_decision": False, "scoring_mode": "requested",
                        "shared_prefix_tokens": 17, "prefill_tokens": 17, "requested_max_tokens": 1,
                        "scoring_rule": "conditional argmax", "prompt_sha256": "a" * 64,
                        "candidate_logprobs": [-1.] * len(ids),
                        "probabilities": {i: 1 / len(ids) for i in ids}, "option_ids": ids}
                return DecisionResponse(request.request_id, ids[0], self.model,
                                        usage=TokenUsage(input_tokens=17, output_tokens=1),
                                        raw_output={"bridge": data})
        def fail_lock(_):
            self.fail("Token bridge already owns qwen4 lock")
        result = self.execute(Tokens, "qwen4_token_scores", fail_lock)
        self.assertEqual(result["status"], "completed")
        controls.verify_run(self.root, "qwen4_token_scores", allow_injected=True)

    def test_attempt_request_tamper_fails_replay(self):
        self.execute()
        output = self.root / "qwen4_grammar"
        attempts = controls.suite.read_rows(output / "attempts.jsonl")
        attempts[0]["request"]["state"]["pair"] = [0, 7]
        with self.assertRaisesRegex(Exception, "request replay mismatch"):
            controls.scores_for(controls.suite.read_rows(self.root / "cases.jsonl"),
                                controls.suite.read_rows(self.root / "requests.jsonl"),
                                attempts, "qwen4_grammar")

    def test_startup_failure_is_sealed_with_all_scheduled_denominators(self):
        class InitializationFailure(FakeClient):
            async def initialize(self):
                raise ProviderHTTPError(503)
        result = self.execute(InitializationFailure)
        self.assertEqual(result["status"], "aborted")
        self.assertEqual(result["calls"], 0)
        _, _, scores, _ = controls.verify_run(self.root, "qwen4_grammar", allow_injected=True)
        self.assertEqual(len(scores), 600)
        self.assertTrue(all(s["failure_code"] == "not_attempted" for s in scores))

    def test_flip_and_five_arm_patterns_distinguish_robust_from_wrong(self):
        rows = []
        choices = {"correct": ["0"] * 5, "wrong": ["1"] * 5,
                   "unstable": ["1", "1", "2", "0", None]}
        for ident, supplied in choices.items():
            for arm, choice in zip(controls.ARMS, supplied):
                rows.append({"instance_id": ident, "task": "degree_exact", "node_count": 8,
                             "arm": arm, "valid": choice is not None, "attempted": True,
                             "correct": choice == "0", "selected_option_id": choice,
                             "input_changed": arm not in ("original", "identical_repeat")})
        summary = controls.metrics(rows)
        paired = {r["arm"]: r for r in summary["paired_choice_changes"]
                  if r["actual_input_changed"] == "all"}
        self.assertEqual(paired["node_relabel"]["wrong_to_other_wrong"], 1)
        self.assertEqual(paired["edge_shuffle"]["wrong_to_correct"], 1)
        self.assertEqual(paired["reversed_options"]["both_valid"], 2)
        self.assertEqual(paired["reversed_options"]["not_both_valid"], 1)
        self.assertEqual(paired["identical_repeat"]["invariant_wrong"], 2)
        joint = summary["five_arm_patterns"][0]
        self.assertEqual(joint["all_five_valid"], 2)
        self.assertEqual(joint["all_five_correct"], 1)
        self.assertEqual(joint["all_five_invariant_wrong"], 1)
