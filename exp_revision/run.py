"""One task entry point: resume the reviewed local phases or report saved status."""
import argparse
import json
import sys
from .common import ROOT


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task"); parser.add_argument("--dataset", default="tt"); parser.add_argument("--smoke", action="store_true")
    args, unknown = parser.parse_known_args()
    if args.task not in ("T1", "T2", "T3", "T5", "T5a", "T5b", "T6", "T10"):
        from .runner import main as legacy_main
        return legacy_main()
    if "--prepare-fold" in unknown or any(value.startswith("--prepare-fold=") for value in unknown):
        from .runner import main as legacy_main
        return legacy_main()
    parser.add_argument("-h", "--help", action="help", help="Resume local frozen phases or show collected task status; --smoke uses existing smoke jobs.")
    # Reject unsupported arguments before importing scheduling modules or taking actions.
    args = parser.parse_args()
    if args.dataset not in ("tt", "aiops"): parser.error("dataset must be tt or aiops")
    from . import remaining_stages as stages
    if args.task == "T10":
        from . import report_queue
        path = report_queue.create(args.smoke)
        monitor = stages.adaptive_queue.check()
        print(json.dumps(dict(task="T10", backend="local", manifest=str(path.relative_to(ROOT)) if path else None,
            action=monitor["action"], report="reports/summary.md" if (ROOT / "reports/summary.md").exists() else None)))
    else:
        monitor = stages.check(); result = stages.status(args.task, args.dataset, args.smoke)
        result["action"] = monitor.get("queue", {}).get("action", monitor["status"])
        print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()
