"""Open the saved dissertation evidence without changing models or data."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    print("KneeAssist XAI - saved research evidence")
    print("=" * 72)
    print("Primary study: ResNet-18 versus EfficientNet-B0 on the fixed MRNet protocol.")
    print("Supplementary evidence: five-model target-routing portfolio for the live research interface.")
    print("No training, checkpoint replacement, case scoring or data modification is performed.\n")
    evidence_dir = ROOT / "docs" / "academic" / "audit_20261015"
    files = [
        evidence_dir / "PROJECT_AUDIT.md",
        evidence_dir / "EXPERIMENTAL_EVIDENCE_REPORT.md",
        evidence_dir / "OBJECTIVE_ALIGNMENT_MATRIX.md",
    ]
    missing = [path for path in files if not path.is_file()]
    if missing:
        print("Missing saved evidence:")
        for path in missing:
            print("-", path)
        return 1
    for path in files:
        print("\n" + "=" * 72)
        print(path.name)
        print("=" * 72)
        print(path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
