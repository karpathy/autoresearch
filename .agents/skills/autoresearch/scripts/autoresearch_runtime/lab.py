"""Split runtime facade. Behavior lives in the sibling modules."""
from __future__ import annotations

from .errors import LabError
from .store import _read_json, _sha256, _write_json
from .gitutil import _git, _git_bytes, _restore_candidate
from .config import LabConfig, MAX_UNCLASSIFIED_CANDIDATE_BYTES, SERVICE_PARTS, _relative_file, _service_file, _validate_fresh_known_case, load_config
from .snapshot import _discover_source, init_lab, setup_project
from .state import _check_candidate_unchanged, _check_external_frozen, _check_frozen, _finalize_setup, _fingerprints, _refresh_dashboard, _runs, _save_manifest, load_state
from .evaluator import _apply_net_stats, _attach_net_stats, _budget_overrun, _evaluate_once, _remaining, _spent, _usage_from_file, _validate_result
from .baseline import run_baseline
from .trial import _worst, assess_candidate, recover_lab, run_trial, status_lab
from .vault import _store_vault, restore_vault
from .replay import _GAP_STATUSES, _input_blockers, _replay_checkout, _tree_hashes, replay_plan, run_replay
from .promote import export_best, promote_in_place, promote_to_branch
from .cli import _print_console_safe, main


if __name__ == '__main__':
    raise SystemExit(main())
