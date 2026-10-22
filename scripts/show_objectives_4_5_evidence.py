"""Legacy filename for compatibility.

The primary evidence is now the saved formal ResNet-18 versus EfficientNet-B0
comparison. This module delegates to the read-only research-evidence viewer and
never trains, scores an upload, or changes files.
"""
from __future__ import annotations

from show_research_evidence import main

if __name__ == "__main__":
    raise SystemExit(main())
