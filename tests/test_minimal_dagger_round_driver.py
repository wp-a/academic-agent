import importlib
import unittest
from argparse import Namespace
from pathlib import Path


class MinimalDaggerRoundDriverTest(unittest.TestCase):
    def test_build_round_paths_uses_fixed_layout(self):
        module = importlib.import_module("train_agent.scripts.run_minimal_dagger_round")
        paths = module.build_round_paths(Path("outputs/dagger_round_test"))

        self.assertEqual(paths["off_policy_dir"], Path("outputs/dagger_round_test/off_policy"))
        self.assertEqual(paths["relabel_dir"], Path("outputs/dagger_round_test/relabel"))
        self.assertEqual(paths["mixed_train_dir"], Path("outputs/dagger_round_test/mixed_train"))
        self.assertEqual(paths["smoke_compare_dir"], Path("outputs/dagger_round_test/smoke_compare"))
        self.assertEqual(paths["round_summary_path"], Path("outputs/dagger_round_test/round_summary.json"))

    def test_summarize_joint_comparison_reports_metric_deltas(self):
        module = importlib.import_module("train_agent.scripts.run_minimal_dagger_round")
        base = {
            "action_agreement": 0.895710,
            "stop_recall": 0.818841,
            "success_rate": 0.878698,
            "early_stop_rate": 0.103550,
            "quote_evidence_hit_rate": 0.878698,
        }
        mixed = {
            "action_agreement": 0.907028,
            "stop_recall": 0.845070,
            "success_rate": 0.890533,
            "early_stop_rate": 0.091716,
            "quote_evidence_hit_rate": 0.857550,
        }

        summary = module.summarize_joint_comparison(base=base, mixed=mixed)

        self.assertEqual(summary["base"]["success_rate"], 0.878698)
        self.assertEqual(summary["mixed"]["success_rate"], 0.890533)
        self.assertAlmostEqual(summary["delta"]["action_agreement"], 0.011318, places=6)
        self.assertAlmostEqual(summary["delta"]["stop_recall"], 0.026229, places=6)
        self.assertAlmostEqual(summary["delta"]["success_rate"], 0.011835, places=6)
        self.assertAlmostEqual(summary["delta"]["early_stop_rate"], -0.011834, places=6)
        self.assertAlmostEqual(summary["delta"]["quote_evidence_hit_rate"], -0.021148, places=6)

    def test_apply_preset_sets_export_relabel_mix_only_mode(self):
        module = importlib.import_module("train_agent.scripts.run_minimal_dagger_round")
        args = Namespace(
            preset="export_relabel_mix_only",
            skip_smoke_compare=False,
        )

        updated = module.apply_preset(args)

        self.assertEqual(updated.preset, "export_relabel_mix_only")
        self.assertTrue(updated.skip_smoke_compare)


if __name__ == "__main__":
    unittest.main()
