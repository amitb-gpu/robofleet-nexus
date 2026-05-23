from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock Isaac simulation runner")
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()

    result = {
        "job_id": args.job_id,
        "backend": "mock",
        "status": "completed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "note": "Mock simulation completed without launching Isaac Sim.",
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
