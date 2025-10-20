"""Tool integrations exposed by the agent runtime."""

from .coverage_map import CoverageMap
from .gates import GateReport, StaticCheck, StaticCheckResult, describe_policy_capsule, run_policy_and_static
from .patch import PatchError, PatchResult, apply_patch
from .phase_logs import PhaseLogEntry, load_phase_log
from .phase_replay import PhaseReplayConfig, PhaseReplayWorkspace, prepare_replay_workspace
from .pytest_runner import PytestResult, PytestStatus, run_pytest
from .vcs import GitCheckpoint, GitError, GitRepository
from .workspace_state import WorkspaceState, apply_workspace_state, capture_workspace_state

__all__ = [
    "CoverageMap",
    "GateReport",
    "GitCheckpoint",
    "GitError",
    "GitRepository",
    "PatchError",
    "PatchResult",
    "PytestResult",
    "PytestStatus",
    "StaticCheck",
    "StaticCheckResult",
    "apply_patch",
    "PhaseLogEntry",
    "load_phase_log",
    "PhaseReplayConfig",
    "PhaseReplayWorkspace",
    "prepare_replay_workspace",
    "describe_policy_capsule",
    "run_policy_and_static",
    "run_pytest",
    "WorkspaceState",
    "apply_workspace_state",
    "capture_workspace_state",
]
