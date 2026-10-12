from accelerator_bench.hub_report import review_profile


def test_placement_review_requires_evidence_and_does_not_infer_fusion():
    assert not review_profile({})["all_reported_entries_on_npu"]
    profile = {"execution_detail": [{"name": "Conv", "compute_unit": "NPU", "execution_cycles": 0}],
               "execution_summary": {"estimated_inference_time": 10, "all_inference_times": [10, 20, 100]}}
    reviewed = review_profile(profile)
    assert reviewed["all_reported_entries_on_npu"]
    assert reviewed["all_samples_median_us"] == 20
    assert reviewed["hub_estimated_inference_us"] == 10
    assert not reviewed["fusion_verified"]
    profile["execution_detail"].append({"compute_unit": "CPU"})
    assert not review_profile(profile)["all_reported_entries_on_npu"]
