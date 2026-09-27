"""Offline structural-bank generation, independent truth, and VF2 regressions."""

from collections import Counter, defaultdict
from copy import deepcopy
from fractions import Fraction
from itertools import combinations, permutations, product
import json
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

import networkx as nx

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.benchmark import extended_tasks as tasks
from src.benchmark import structural_tasks as structural
from scripts.audit_task_shortcuts import audit


def graph(record):
    return structural._graph(record["state"], weighted=record["task"] == "tsp_construct")


class StructuralTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.small, cls.small_metadata = structural.build_structural_bank(node_counts=(6,), repetitions=1)
        cls.bank, cls.metadata = structural.build_structural_bank(node_counts=(8,), repetitions=1)

    def test_full_small_coverage_schema_and_budgets(self):
        for n, records, metadata in ((6, self.small, self.small_metadata), (8, self.bank, self.metadata)):
            with self.subTest(n=n):
                self.assertEqual(len(records), n + 14)
                self.assertEqual(sum(r["query_budget"] for r in records), 4 * n + 8)
                self.assertEqual(metadata["requested_instances"], len(records))
                self.assertEqual(metadata["shortfall_instances"], 0)
                self.assertEqual(metadata["query_pair_protocol"], "strict_degree_matched_v2")
                self.assertEqual(metadata["evaluation_scope"], "development_no_prior_pool")
                self.assertFalse(metadata["prior_exclusion_applied"])
                self.assertEqual(metadata["planned_queries_per_model"], 4 * n + 8)
                expected = {task: n if task == "degree_exact" else
                            2 if task in structural.BINARY_TASKS else 1 for task in structural.TASKS}
                self.assertEqual(Counter(r["task"] for r in records), expected)
                for record in records:
                    self.assertEqual(set(record), {
                        "id", "task", "kind", "state", "query_budget", "private"})
                    self.assertRegex(record["id"], r"^s-[0-9a-f]{32}$")
                    self.assertEqual(set(record["state"]), tasks._STATE_KEYS[record["task"]])
                    self.assertNotIn("family", record["state"])
                    self.assertNotIn("label", record["state"])
        self.assertEqual(Counter(r["private"]["answer"] for r in self.bank
                                 if r["task"] == "degree_exact"), Counter(range(8)))

    def test_reproducible_local_rng_shuffled_order_and_json(self):
        random.seed(831)
        before = random.getstate()
        repeated, metadata = structural.build_structural_bank(node_counts=(8,), repetitions=1)
        self.assertEqual(random.getstate(), before)
        self.assertEqual(repeated, self.bank)
        self.assertEqual(metadata, self.metadata)
        self.assertEqual(json.loads(json.dumps(metadata, allow_nan=False)), metadata)
        order = [r["task"] for r in repeated]
        self.assertNotEqual(order, sorted(order, key=structural.TASKS.index))
        alternate, _ = structural.build_structural_bank(seed=20261001, node_counts=(6,), repetitions=1)
        self.assertNotEqual(alternate, self.small)

    def test_independent_exact_answers_and_public_requests(self):
        for record in self.bank:
            if record["kind"] != "exact":
                continue
            g, task, state = graph(record), record["task"], record["state"]
            if task == "degree_exact":
                expected = g.degree(state["vertex"])
            elif task == "cycle_detection":
                expected = "yes" if nx.cycle_basis(g) else "no"
            elif task == "adjacency":
                expected = "yes" if g.has_edge(*state["pair"]) else "no"
            elif task == "pair_connectivity":
                expected = "yes" if nx.has_path(g, *state["pair"]) else "no"
            elif task == "distance_threshold":
                expected = "yes" if nx.has_path(g, *state["pair"]) and (
                    nx.shortest_path_length(g, *state["pair"]) <= state["threshold"]) else "no"
            else:
                before = nx.number_connected_components(g)
                g.remove_node(state["vertex"])
                expected = "yes" if nx.number_connected_components(g) > before else "no"
            with self.subTest(task=task):
                self.assertEqual(record["private"]["answer"], expected)
                request = tasks.next_request(record, [])
                self.assertNotIn("private", request.state)
                self.assertNotIn("family", request.state)
                self.assertNotIn("label", request.state)
                self.assertTrue(request.request_id.startswith(record["id"] + ":"))
                self.assertTrue(tasks.score(record, [str(expected)])["correct"])

    def test_balanced_pairs_share_graph_except_matched_cycle_switch(self):
        index = {r["id"]: r for r in self.bank}
        pairs = defaultdict(list)
        for row in self.metadata["records"]:
            if row["pair_id"]:
                pairs[row["pair_id"]].append(row)
        self.assertEqual(len(pairs), 5)
        for rows in pairs.values():
            self.assertEqual(len(rows), 2)
            records = [index[row["instance_id"]] for row in rows]
            self.assertEqual(Counter(r["private"]["answer"] for r in records), {"yes": 1, "no": 1})
            left, right = map(graph, records)
            if records[0]["task"] == "cycle_detection":
                self.assertNotEqual(rows[0]["cluster_id"], rows[1]["cluster_id"])
                self.assertEqual(len(left), len(right))
                self.assertEqual(left.number_of_edges(), right.number_of_edges())
                self.assertEqual(sorted(dict(left.degree()).values()), sorted(dict(right.degree()).values()))
                self.assertEqual(rows[0]["density"], rows[1]["density"])
                self.assertNotEqual(nx.is_forest(left), nx.is_forest(right))
            else:
                self.assertEqual(records[0]["state"]["edges"], records[1]["state"]["edges"])
                self.assertEqual(rows[0]["cluster_id"], rows[1]["cluster_id"])
                self.assertTrue(structural._isomorphic(left, right))
            if records[0]["task"] == "distance_threshold":
                threshold = records[0]["state"]["threshold"]
                self.assertEqual(records[1]["state"]["threshold"], threshold)
                self.assertEqual(sorted(row["task_parameters"]["shortest_distance"] for row in rows),
                                 [threshold, threshold + 1])

    def test_general_forest_switches_include_branching_and_disconnected_forests(self):
        pairs = [structural._forest_switch(random.Random(seed), 10) for seed in range(30)]
        pairs = [pair for pair in pairs if pair is not None]
        self.assertTrue(any(max(dict(structural._graph(pair[0][0]).degree()).values()) >= 3
                            for pair in pairs))
        self.assertTrue(any(nx.number_connected_components(structural._graph(pair[0][0])) > 1
                            for pair in pairs))
        for pair in pairs:
            before, after = (structural._graph(item[0]) for item in pair)
            self.assertEqual(sorted(before.degree(v) for v in before),
                             sorted(after.degree(v) for v in after))
            self.assertTrue(nx.is_forest(before))
            self.assertFalse(nx.is_forest(after))
            self.assertEqual(len(set(before.edges()) ^ set(after.edges())), 4)

    def test_metadata_density_parameters_and_exact_clusters(self):
        records = {record["id"]: record for record in self.bank}
        by_cluster = defaultdict(list)
        for row in self.metadata["records"]:
            record = records[row["instance_id"]]
            g = graph(record)
            self.assertEqual(row["n"], len(g))
            self.assertEqual(row["m"], g.number_of_edges())
            self.assertEqual(row["node_count"], len(g))
            self.assertEqual(row["edge_count"], g.number_of_edges())
            self.assertEqual(self.metadata["cases"][record["id"]], row)
            self.assertEqual(row["density"], nx.density(g))
            self.assertTrue(row["family_parameters"])
            self.assertTrue(row["task_parameters"])
            self.assertEqual(row["label"], record["private"].get("answer"))
            by_cluster[row["cluster_id"]].append(record)
        for cluster in self.metadata["clusters"]:
            members = by_cluster[cluster["cluster_id"]]
            self.assertEqual(set(cluster["instance_ids"]), {r["id"] for r in members})
            self.assertEqual(set(cluster["tasks"]), {r["task"] for r in members})
            for record in members:
                self.assertTrue(structural._isomorphic(graph(members[0]), graph(record)))
                self.assertEqual(cluster["fingerprint"], structural._fingerprint(graph(record)))

    def test_query_pair_controls_match_available_degree_signatures(self):
        examples = {
            "adjacency": nx.cycle_graph(6),
            "pair_connectivity": nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(3)),
            "distance_threshold": nx.path_graph(7),
            "articulation_point": nx.Graph([
                (0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 6), (6, 4)]),
        }
        for task, g in examples.items():
            with self.subTest(task=task), patch.object(
                    structural, "_query_base", return_value=(g, "test-family", {"test": True})):
                candidate = structural._candidate(task, len(g), random.Random(17))
                self.assertEqual(len(candidate), 2)
                control = candidate[0][2]["_degree_matching"]
                self.assertTrue(control["matched"])
                self.assertEqual(control["positive_signature"], control["negative_signature"])
                self.assertEqual(control["reason"], "matched")
                if task == "articulation_point":
                    self.assertEqual(candidate[0][0]["vertex"], 3)
                    self.assertEqual(control["positive_signature"], 2)
                if task == "distance_threshold":
                    for offset, (state, _, _) in enumerate(candidate):
                        self.assertEqual(nx.shortest_path_length(g, *state["pair"]),
                                         state["threshold"] + offset)

    def test_unmatched_query_proposal_is_unavailable_without_fallback(self):
        star = nx.star_graph(4)
        with patch.object(structural, "_query_base",
                          return_value=(star, "test-star", {"test": True})) as generate:
            candidate = structural._candidate("articulation_point", 5, random.Random(17))
        generate.assert_called_once()
        self.assertIsNone(candidate)

    def test_articulation_generation_and_case_metadata_controls(self):
        index = {r["id"]: r for r in self.bank}
        articulation = [row for row in self.metadata["cases"].values()
                        if row["task"] == "articulation_point"]
        self.assertTrue(all(row["degree_matching"]["positive_signature"] >= 2 for row in articulation))
        for row in self.metadata["cases"].values():
            if row["degree_matching"] is None:
                continue
            control, state = row["degree_matching"], index[row["instance_id"]]["state"]
            self.assertTrue(control["matched"])
            g = structural._graph(state)
            signature = g.degree(state["vertex"]) if row["task"] == "articulation_point" \
                else sorted(g.degree(v) for v in state["pair"])
            field = "positive_signature" if row["label"] == "yes" else "negative_signature"
            self.assertEqual(signature, control[field])
            self.assertEqual(control["matched"],
                             control["positive_signature"] == control["negative_signature"])
            if row["task"] == "distance_threshold":
                self.assertEqual(row["task_parameters"]["distance_kind"], "finite")
        for cell in self.metadata["coverage"]:
            self.assertEqual(cell["emitted"], cell["generated"])
            self.assertEqual(cell["reason"], "complete" if not cell["shortfall"]
                             else "bounded_attempts_exhausted")

    def test_dedup_is_per_task_and_shared_query_pairs_are_intentional(self):
        metadata = {row["instance_id"]: row for row in self.metadata["records"]}
        for task in structural.TASKS:
            records = [record for record in self.bank if record["task"] == task]
            for left, right in combinations(records, 2):
                if structural._isomorphic(graph(left), graph(right)):
                    self.assertIn(task, structural.BINARY_TASKS - {"cycle_detection"})
                    self.assertEqual(metadata[left["id"]]["pair_id"], metadata[right["id"]]["pair_id"])
                    self.assertEqual(metadata[left["id"]]["cluster_id"], metadata[right["id"]]["cluster_id"])

    def test_prior_exclusion_checks_all_task_graphs_not_only_matching_task(self):
        fresh, metadata = structural.build_structural_bank(
            node_counts=(8,), repetitions=1, prior_instances=self.bank)
        self.assertGreater(metadata["rejections"].get("prior_isomorphism", 0), 0)
        self.assertEqual(metadata["isomorphism"]["prior_records"], len(self.bank))
        self.assertEqual(metadata["evaluation_scope"], "prior_pool_relative")
        self.assertTrue(metadata["prior_exclusion_applied"])
        for record in fresh:
            candidate = graph(record)
            for prior in self.bank:
                previous = graph(prior) if record["task"] == "tsp_construct" else \
                    structural._graph(prior["state"])
                self.assertFalse(structural._isomorphic(candidate, previous))
        self.assertEqual(metadata["generated_instances"] + metadata["shortfall_instances"],
                         metadata["requested_instances"])

    def test_compact_prior_pool_needs_no_private_oracles_or_tsp_coordinates(self):
        compact = [{
            "id": record["id"],
            "task": ("tsp_construct" if record["task"] == "tsp_construct" else
                     "lt_influence_construct" if record["state"]["directed"] else "degree_exact"),
            "state": {key: deepcopy(record["state"][key]) for key in ("nodes", "edges", "directed")},
        } for record in self.bank]
        self.assertTrue(all("private" not in record and "points" not in record["state"]
                            for record in compact))
        from_full = structural.build_structural_bank(
            node_counts=(8,), repetitions=1, prior_instances=self.bank)
        from_compact = structural.build_structural_bank(
            node_counts=(8,), repetitions=1, prior_instances=compact)
        self.assertEqual(from_compact, from_full)
        self.assertGreater(from_compact[1]["rejections"].get("prior_isomorphism", 0), 0)

    def test_genuine_wl_collision_uses_vf2_not_hash_equality(self):
        cycle = structural._graph(structural._state(nx.cycle_graph(6)))
        triangles = structural._graph(structural._state(
            nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(3))))
        self.assertEqual(structural._fingerprint(cycle), structural._fingerprint(triangles))
        index = structural._IsoIndex()
        self.assertEqual(index.add(cycle, "cycle"), "cycle")
        self.assertIsNone(index.find(triangles))
        self.assertEqual(index.add(triangles, "triangles"), "triangles")
        self.assertEqual(index.size, 2)
        renamed = nx.relabel_nodes(cycle, {v: 100 - v for v in cycle})
        self.assertEqual(structural._fingerprint(cycle), structural._fingerprint(renamed))
        self.assertEqual(index.find(renamed), "cycle")
        self.assertGreater(index.comparisons, 0)

    def test_forced_hash_collisions_preserve_directed_and_weighted_identity(self):
        directed = structural._graph({"nodes": list(range(4)), "edges": [[0, 1], [0, 2], [0, 3]],
                                      "directed": True})
        reversed_graph = directed.reverse(copy=True)
        weighted = structural._graph({
            "nodes": list(range(4)), "edges": [[u, v, 1 + u + v] for u, v in combinations(range(4), 2)],
            "directed": False}, weighted=True)
        changed = weighted.copy()
        changed[0][1]["weight"] += 1
        with patch.object(structural, "_fingerprint", return_value="forced-collision"):
            index = structural._IsoIndex()
            index.add(directed, "directed")
            self.assertIsNone(index.find(reversed_graph))
            index.add(weighted, "weighted")
            self.assertIsNone(index.find(changed))
            renamed = nx.relabel_nodes(weighted, {v: 10 + v for v in weighted})
            self.assertEqual(index.find(renamed), "weighted")
            unweighted = structural._graph(structural._state(nx.complete_graph(4)))
            self.assertIsNone(index.find(unweighted))
        self.assertEqual(weighted.number_of_edges(), changed.number_of_edges())

    def test_exhausted_cells_are_bounded_explicit_and_never_filled(self):
        with patch.object(structural, "MAX_ATTEMPTS_PER_CELL", 3), \
                patch.object(structural, "_candidate", return_value=None) as candidate:
            bank, metadata = structural.build_structural_bank(node_counts=(6,), repetitions=1)
        self.assertEqual(bank, [])
        self.assertEqual(metadata["requested_instances"], 20)
        self.assertEqual(metadata["shortfall_instances"], 20)
        self.assertEqual(metadata["planned_queries_per_model"], 0)
        self.assertEqual(metadata["requested_queries_per_model"], 32)
        self.assertEqual(candidate.call_count, (6 + 9) * 3)
        self.assertTrue(all(c["attempts"] == 3 and c["generated"] == 0 for c in metadata["coverage"]))
        degree_cells = [c for c in metadata["coverage"] if c["task"] == "degree_exact"]
        self.assertEqual({c["target_degree"] for c in degree_cells}, set(range(6)))
        self.assertTrue(all(c["shortfall"] == 2 for c in metadata["coverage"]
                            if c["task"] in structural.BINARY_TASKS))

    def test_prior_rejection_omits_entire_cycle_pair(self):
        cycle_records = [r for r in self.bank if r["task"] == "cycle_detection"]
        prior = next(r for r in cycle_records if r["private"]["answer"] == "no")
        original = structural._candidate
        def fixed_cycle(task, n, rng, target):
            if task == "cycle_detection":
                return [(deepcopy(r["state"]), "test-cycle", {}) for r in cycle_records]
            return original(task, n, rng, target)
        with patch.object(structural, "MAX_ATTEMPTS_PER_CELL", 3), \
                patch.object(structural, "_candidate", side_effect=fixed_cycle):
            generated, metadata = structural.build_structural_bank(
                node_counts=(8,), repetitions=1, prior_instances=[prior])
        self.assertFalse(any(r["task"] == "cycle_detection" for r in generated))
        cell = next(c for c in metadata["coverage"] if c["task"] == "cycle_detection")
        self.assertEqual(cell["shortfall"], 2)
        self.assertEqual(cell["rejections"]["prior_isomorphism"], 3)

    def test_real_small_cycle_support_exhaustion_is_explicit(self):
        bank, metadata = structural.build_structural_bank(node_counts=(5,), repetitions=2)
        cycle_cells = [cell for cell in metadata["coverage"] if cell["task"] == "cycle_detection"]
        # At n=5, the switch's length-four forest path has only one topology.
        self.assertEqual(sum(cell["generated"] for cell in cycle_cells), 2)
        self.assertEqual(sum(cell["shortfall"] for cell in cycle_cells), 2)
        exhausted = next(cell for cell in cycle_cells if cell["shortfall"])
        self.assertEqual(exhausted["attempts"], structural.MAX_ATTEMPTS_PER_CELL)
        self.assertGreater(exhausted["rejections"].get("within_task_isomorphism", 0), 0)
        self.assertEqual(Counter(record["private"]["answer"] for record in bank
                                 if record["task"] == "cycle_detection"), {"no": 1, "yes": 1})

    def test_continuous_development_sizes_keep_bounded_coverage_not_ood_claims(self):
        sizes = tuple(range(5, 13))
        bank, metadata = structural.build_structural_bank(seed=20261001, node_counts=sizes, repetitions=2)
        self.assertEqual(metadata["node_counts"], list(sizes))
        self.assertEqual(metadata["requested_instances"], 360)
        self.assertEqual(metadata["requested_queries_per_model"], 672)
        self.assertEqual(metadata["evaluation_scope"], "development_no_prior_pool")
        self.assertFalse(metadata["prior_exclusion_applied"])
        self.assertEqual({(record["task"], len(record["state"]["nodes"])) for record in bank},
                         {(task, n) for task in structural.TASKS for n in sizes})
        cycle_n5 = [cell for cell in metadata["coverage"]
                    if cell["task"] == "cycle_detection" and cell["n"] == 5]
        self.assertEqual(sum(cell["emitted"] for cell in cycle_n5), 2)
        self.assertEqual(sum(cell["shortfall"] for cell in cycle_n5), 2)
        self.assertEqual(metadata["generated_instances"] + metadata["shortfall_instances"], 360)
        self.assertEqual(metadata["generated_instances"], 356)
        self.assertEqual(metadata["planned_queries_per_model"], 668)
        result = audit(bank)
        for task in structural.BINARY_TASKS:
            with self.subTest(task=task):
                self.assertEqual(result["by_task"][task]["empirical_lookup_ceiling"], 0.5)
                self.assertEqual(result["by_task"][task]["mixed_label_instance_fraction"], 1.0)

    def test_all_binary_paired_instances_have_chance_cheap_feature_ceiling(self):
        for bank in (self.small, self.bank):
            result = audit(bank)
            for task in structural.BINARY_TASKS:
                with self.subTest(task=task, n=len(bank[0]["state"]["nodes"])):
                    self.assertEqual(result["by_task"][task]["empirical_lookup_ceiling"], 0.5)
                    self.assertEqual(result["by_task"][task]["mixed_label_instance_fraction"], 1.0)

    def test_optimization_families_and_independent_small_optima(self):
        for record in self.small:
            if record["kind"] != "optimization":
                continue
            g, task, state = graph(record), record["task"], record["state"]
            if task == "tsp_construct":
                values = [sum(g[u][v]["weight"] for u, v in zip(tour, tour[1:]))
                          for tail in permutations(range(1, len(g)))
                          for tour in ([0, *tail, 0],)]
                optimum = min(values)
            elif task == "lt_influence_construct":
                self.assertTrue(nx.is_directed_acyclic_graph(g))
                self.assertEqual(state["seed_count"], 2)
                values = []
                for seeds in combinations(g, 2):
                    active = set(seeds)
                    while True:
                        added = {v for v in g if v not in active and g.in_degree(v)
                                 and sum((Fraction(1, g.in_degree(v)) for u in g.predecessors(v)
                                          if u in active), Fraction(0)) >= Fraction(1, 2)}
                        if not added:
                            break
                        active |= added
                    values.append(len(active))
                optimum = max(values)
            else:
                values = []
                for tail in product((0, 1), repeat=len(g) - 1):
                    sides = (0, *tail)
                    groups = [{v for v in g if sides[v] == side} for side in (0, 1)]
                    values.append(nx.cut_size(g, *groups) if task == "maxcut_construct" else
                                  nx.community.modularity(g, [group for group in groups if group]))
                optimum = max(values)
            self.assertAlmostEqual(record["private"]["objective"], optimum, places=12)
        for task, families in (
            ("maxcut_construct", {"small_world", "switched_regular"}),
            ("tsp_construct", {"clustered_manhattan", "ring_manhattan"}),
        ):
            proposals = [structural._candidate(task, 8, random.Random(seed)) for seed in range(20)]
            self.assertEqual({proposal[0][1] for proposal in proposals if proposal}, families)

    def test_public_validator_called_and_rejects_corrupt_truth(self):
        with patch.object(tasks, "validate_instances", wraps=tasks.validate_instances) as validate:
            records, metadata = structural.build_structural_bank(node_counts=(6,), repetitions=1)
        validate.assert_called_once_with(records)
        self.assertEqual(metadata["validation"]["instances"], len(records))
        corrupted = deepcopy(records)
        exact = next(record for record in corrupted if record["task"] == "adjacency")
        exact["private"]["answer"] = "no" if exact["private"]["answer"] == "yes" else "yes"
        with self.assertRaises(ValueError):
            tasks.validate_instances(corrupted)

    def test_invalid_arguments_and_prior_records(self):
        for kwargs in ({"seed": True}, {"repetitions": 0}, {"repetitions": 1.0},
                       {"node_counts": ()}, {"node_counts": (4,)}, {"node_counts": (13,)},
                       {"node_counts": (8, 8)}, {"node_counts": (True,)},
                       {"prior_instances": [{}]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                structural.build_structural_bank(**kwargs)


if __name__ == "__main__":
    unittest.main()
