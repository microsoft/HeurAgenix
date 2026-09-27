import copy
from pathlib import Path
import unittest

from scripts import stronger_local_preflight_v1 as p


class StrongerLocalAdmissionTests(unittest.TestCase):
    def mapping(self):
        services, inventory = {}, {"gpus": [], "processes": []}
        for lane, (gpu, _, _) in p.LANES.items():
            pid = 100 + gpu
            services[lane] = {
                "cuda_environment": {"CUDA_VISIBLE_DEVICES": str(gpu)},
                "process_tree": [{"pid": pid}],
            }
            inventory["gpus"].append([str(gpu), f"GPU-{gpu}"])
            inventory["processes"].append([str(pid), f"GPU-{gpu}", "100"])
        return services, inventory

    def test_all_four_originals_required(self):
        services, inventory = self.mapping()
        self.assertEqual(p.four_gpu_mapping(services, inventory),
                         [f"GPU-{gpu}" for gpu in range(4)])
        del services["qwen9"]
        with self.assertRaisesRegex(p.suite.shared.Refusal, "Missing original"):
            p.four_gpu_mapping(services, inventory)

    def test_unknown_occupants_rejected_on_every_gpu(self):
        for gpu in range(4):
            services, inventory = self.mapping()
            inventory["processes"].append(["999", f"GPU-{gpu}", "1"])
            with self.assertRaisesRegex(p.suite.shared.Refusal, "Unexpected GPU"):
                p.four_gpu_mapping(services, inventory)

    def test_wrong_mapping_and_missing_engine_rejected(self):
        services, inventory = self.mapping()
        inventory["processes"][3][1] = "GPU-2"
        with self.assertRaisesRegex(p.suite.shared.Refusal, "mapping differs"):
            p.four_gpu_mapping(services, inventory)
        services, inventory = self.mapping()
        inventory["processes"].pop()
        with self.assertRaisesRegex(p.suite.shared.Refusal, "mapping differs"):
            p.four_gpu_mapping(services, inventory)

    def test_start_ticks_not_historical_pid_are_identity(self):
        identity = {"pid": 100, "start_ticks": "123", "argv": ["vllm"],
                    "cuda_environment": {"CUDA_VISIBLE_DEVICES": "0"}, "cwd": str(p.REPO)}
        self.assertFalse(p.resources.same_identity(
            identity, {**identity, "start_ticks": "124"}))

    def metadata(self):
        spec = p.MODELS[0]
        info = {"sha": spec["revision"], "siblings": [{
            "rfilename": "model-1.safetensors", "size": 123,
            "lfs": {"sha256": "a" * 64, "size": 123}}]}
        config = {"architectures": [spec["architecture"]], "torch_dtype": "bfloat16"}
        index = {"weight_map": {"weight": "model-1.safetensors"}}
        return spec, info, config, index

    def test_metadata_is_not_payload_verification(self):
        result = p.validate_metadata(*self.metadata())
        self.assertEqual(result["payload_bytes"], 123)
        self.assertFalse(result["local_payload_sha_verified"])
        self.assertFalse(result["safetensors_header_dtypes_verified"])
        self.assertFalse(result["tokenizer_rendering_verified"])

    def test_revision_dtype_quantization_and_sha_fail_closed(self):
        for change in ("revision", "dtype", "quantization", "sha"):
            spec, info, config, index = copy.deepcopy(self.metadata())
            if change == "revision":
                info["sha"] = "wrong"
            elif change == "dtype":
                config["torch_dtype"] = "float16"
            elif change == "quantization":
                config["quantization_config"] = {"quant_method": "awq"}
            else:
                info["siblings"][0]["lfs"]["sha256"] = "missing"
            with self.assertRaises(p.suite.shared.Refusal):
                p.validate_metadata(spec, info, config, index)

    def test_exact_storage_headroom_boundary(self):
        data = [{"repository_bytes": 200, "payload_bytes": 190}]
        self.assertFalse(p.storage_admission(299, data, headroom=100)["sufficient"])
        self.assertTrue(p.storage_admission(300, data, headroom=100)["sufficient"])

    def test_lock_namespace_never_relocated_or_opened(self):
        evidence = {"leases": {"qwen08": {"path": "/outside/project/one.lock"},
                               "qwen2": {"path": "/outside/project/two.lock"}}}
        result = p.lock_evidence(evidence, Path("/project"))
        self.assertFalse(result["admissible"])
        self.assertFalse(result["access_attempted"])
        self.assertFalse(result["relocated"])
        self.assertEqual(set(result["standard_paths"]), set(p.LANES))
        self.assertTrue(all(path.startswith("/outside/project/")
                            for path in result["standard_paths"].values()))


if __name__ == "__main__":
    unittest.main()
