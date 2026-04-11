import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "curriculum" / "plan.json"


class WorkspaceLayoutTests(unittest.TestCase):
    def test_core_workspace_paths_exist(self) -> None:
        required_paths = [
            "README.md",
            "curriculum/plan.json",
            "notes/README.md",
            "notes/templates/weekly-review.md",
            "docs/setup/wsl-ubuntu-cuda.md",
            "docs/resources.md",
            "labs/cuda",
            "labs/triton",
            "benchmarks/README.md",
            "scripts/verify_windows_gpu.ps1",
            "scripts/verify_wsl_gpu.sh",
        ]
        for relative_path in required_paths:
            with self.subTest(relative_path=relative_path):
                self.assertTrue((ROOT / relative_path).exists(), relative_path)

    def test_curriculum_contains_12_weeks(self) -> None:
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        weeks = plan["weeks"]
        self.assertEqual(len(weeks), 12)
        self.assertEqual([week["week"] for week in weeks], list(range(1, 13)))

    def test_curriculum_references_existing_artifacts(self) -> None:
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        for week in plan["weeks"]:
            with self.subTest(week=week["week"]):
                self.assertTrue((ROOT / week["note"]).is_file(), week["note"])
                for artifact in week["artifacts"]:
                    self.assertTrue((ROOT / artifact).exists(), artifact)

    def test_acceptance_criteria_match_major_milestones(self) -> None:
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        milestone_weeks = sorted(int(week) for week in plan["acceptance_criteria"])
        self.assertEqual(milestone_weeks, [3, 6, 8, 10, 12])

    def test_verification_scripts_include_key_commands(self) -> None:
        windows_script = (ROOT / "scripts" / "verify_windows_gpu.ps1").read_text(
            encoding="utf-8"
        )
        wsl_script = (ROOT / "scripts" / "verify_wsl_gpu.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("nvidia-smi", windows_script)
        self.assertIn("nvidia-smi", wsl_script)
        self.assertIn("nvcc --version", wsl_script)


if __name__ == "__main__":
    unittest.main()
