"""Curated Terminal-Bench (classic, local-Docker) env with a dense partial-credit reward.

STATUS (2026-05-29): does NOT run on the bundled verifiers 0.1.14 — this forks the older
`ibrahim` API, where `rollout` was overridable; in 0.1.14 `rollout`/`is_completed` are @final.
Kept for the reward/curation logic + as a porting reference. See
docs/terminal-bench-verifiers-compat.md for the fast paths (no-Docker env / official v1 Harbor).

Forked from the community `ibrahim/terminal-bench` verifiers wrapper (MIT). Two changes
vs upstream, both marked `# LAGUNA:`
  1. `_run_tests_and_score` stores per-test `tests_passed`/`tests_total` in state (upstream
     only kept the binary `terminalbench_is_resolved`), enabling a dense reward.
  2. The reward is a shaped partial-credit signal (test-pass fraction + efficiency) instead
     of pass/fail, and tasks can be curated to a difficulty-appropriate subset via `task_ids`.

Self-contained (vendors the reward math from src/laguna_finetune/rewards.py) so it installs
and pushes to the Hub with only its declared deps.
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from datasets import Dataset as _HFDS
from terminal_bench.dataset.dataset import Dataset as TBDataset
from terminal_bench.agents.terminus_2.terminus_json_plain_parser import (
    TerminusJSONPlainParser,
)
from terminal_bench.handlers.trial_handler import TrialHandler
from terminal_bench.parsers.pytest_parser import UnitTestStatus
from terminal_bench.terminal.docker_compose_manager import DockerComposeManager
from terminal_bench.terminal.tmux_session import TmuxSession

import verifiers as vf


# --- vendored reward math (mirror of src/laguna_finetune/rewards.py; kept inline for self-containment) ---
@dataclass(frozen=True)  # no slots= : robust when the env is loaded via a non-standard importer
class RolloutState:
    tests_passed: int
    tests_total: int
    turns: int
    max_turns: int
    succeeded: bool


def _partial_credit(s: RolloutState) -> float:
    return s.tests_passed / s.tests_total if s.tests_total > 0 else 0.0


def _efficiency_bonus(s: RolloutState) -> float:
    if not s.succeeded or s.max_turns <= 0:
        return 0.0
    return max(0.0, 1.0 - s.turns / s.max_turns)


def _shaped(s: RolloutState, efficiency_weight: float) -> float:
    return _partial_credit(s) + efficiency_weight * _efficiency_bonus(s)


def _rollout_state(state, max_turns: int) -> RolloutState:
    return RolloutState(
        tests_passed=int(state.get("tests_passed", 0)),
        tests_total=int(state.get("tests_total", 0)),
        turns=int(state.get("turn", 0)),
        max_turns=max_turns,
        succeeded=bool(state.get("terminalbench_is_resolved", False)),  # LAGUNA: real upstream key
    )


PROMPT_TEMPLATE = (
    "You are an AI assistant tasked with solving command-line tasks in a Linux environment. You will be given a task description and the output from previously executed commands. Your goal is to solve the task by providing batches of shell commands.\n\n"
    "Format your response as JSON with the following structure:\n\n"
    "{{\n"
    '  "analysis": "Analyze the current state based on the terminal output provided.",\n'
    '  "plan": "Describe your plan for the next steps.",\n'
    '  "commands": [\n'
    '    {{"keystrokes": "ls -la\\n", "duration": 0.1}}\n'
    "  ],\n"
    '  "task_complete": true\n'
    "}}\n\n"
    "Required fields: analysis, plan, commands. Optional: task_complete (defaults false).\n"
    "Command object: keystrokes (exact, usually end with \\n), duration (seconds to wait, default 1.0).\n"
    "Use tmux escape sequences for special keys (C-c, C-d). Never wait longer than 60 seconds; poll instead.\n\n"
    "Task Description:\n{instruction}\n\n"
    "Current terminal state:\n{terminal_state}\n"
)


class TerminalBenchCuratedEnv(vf.MultiTurnEnv):
    def __init__(
        self,
        *,
        dataset_name: str,
        dataset_version: str,
        registry_url: str | None,
        task_ids: List[str] | None,
        exclude_task_ids: List[str] | None = None,
        n_tasks: int | None = None,
        runs_dir: str | None = None,
        timeout_multiplier: float | None = None,
        agent_timeout_sec: float | None = None,
        test_timeout_sec: float | None = None,
        no_rebuild: bool = False,
        cleanup: bool = False,
        max_turns: int = 50,
        fn: str = "shaped",
        efficiency_weight: float = 0.1,
    ) -> None:
        if task_ids:
            effective_task_ids = list(task_ids)
        else:
            ds_all = TBDataset(name=dataset_name, version=dataset_version, registry_url=registry_url)
            effective_task_ids = [p.name for p in ds_all.tasks]

        if exclude_task_ids:
            patterns = [re.compile(re.escape(p)) for p in exclude_task_ids]
            effective_task_ids = [t for t in effective_task_ids if not any(c.search(t) for c in patterns)]
        if isinstance(n_tasks, int) and n_tasks > 0:
            effective_task_ids = effective_task_ids[:n_tasks]

        rows = [{"question": "", "answer": "", "task": t, "info": {"task_id": t}} for t in effective_task_ids]
        ds = _HFDS.from_list(rows or [{"question": "", "answer": "", "task": "default", "info": {}}])

        # LAGUNA: shaped partial-credit reward + a 0-weight binary success metric.
        def shaped_reward(**kwargs) -> float:
            s = _rollout_state(kwargs.get("state", {}) or {}, max_turns)
            return float(s.succeeded) if fn == "binary" else _shaped(s, efficiency_weight)

        def success_metric(**kwargs) -> float:
            return 1.0 if (kwargs.get("state", {}) or {}).get("terminalbench_is_resolved") else 0.0

        rubric = vf.Rubric(funcs=[shaped_reward, success_metric], weights=[1.0, 0.0])
        super().__init__(dataset=ds, eval_dataset=ds, rubric=rubric, max_turns=max_turns, message_type="chat")

        self._dataset_name = dataset_name
        self._dataset_version = dataset_version
        self._registry_url = registry_url
        self._parser = TerminusJSONPlainParser()
        self._runs_root = Path(runs_dir or "runs").resolve()
        self._timeout_mult = float(timeout_multiplier) if timeout_multiplier is not None else 1.0
        self._agent_timeout_override = agent_timeout_sec
        self._test_timeout_override = test_timeout_sec
        self._no_rebuild = bool(no_rebuild)
        self._cleanup = bool(cleanup)

    async def setup_state(self, state, **kwargs):
        state.setdefault("pending_confirm", False)
        state.setdefault("terminalbench_done", False)
        if (sess := kwargs.get("tb_session")) is not None:
            state["tb_session"] = sess
        if "agent_deadline" not in state and "agent_deadline" in kwargs:
            state["agent_deadline"] = kwargs["agent_deadline"]
        return state

    async def is_completed(self, messages, state, **kwargs) -> bool:
        if state.get("terminalbench_done"):
            return True
        deadline = state.get("agent_deadline")
        if isinstance(deadline, (int, float)) and deadline > 0 and time.time() >= float(deadline):
            state["terminalbench_done"] = True
            return True
        return state.get("turn", 0) >= self.max_turns

    async def env_response(self, messages, state, **kwargs):
        last = messages[-1]
        content = last.get("content") if isinstance(last, dict) else ""
        parse_result = self._parser.parse_response(content if isinstance(content, str) else "")
        session: TmuxSession = kwargs.get("tb_session") or state.get("tb_session")
        if session is None:
            raise RuntimeError("tb_session missing in env_response context")

        if parse_result.error:
            return [{"role": "user", "content": f"Parsing error: {parse_result.error}. Provide valid JSON."}], state

        for cmd in parse_result.commands:
            try:
                session.send_keys(cmd.keystrokes, block=False,
                                  min_timeout_sec=min(float(getattr(cmd, "duration", 1.0)), 60.0))
            except TimeoutError:
                screen = self._limit_output_length(session.capture_pane(capture_entire=False))
                return [{"role": "user", "content": f"[TIMEOUT] {cmd.keystrokes}\n{screen}"}], state

        if parse_result.is_task_complete:
            if state.get("pending_confirm"):
                state["terminalbench_done"] = True
                return [], state
            state["pending_confirm"] = True
            screen = self._limit_output_length(session.capture_pane(capture_entire=False))
            return [{"role": "user", "content": f"{screen}\n\nConfirm complete? Re-send \"task_complete\": true."}], state
        state["pending_confirm"] = False
        return [{"role": "user", "content": self._limit_output_length(session.capture_pane(capture_entire=False))}], state

    def _resolve_task(self, task_id: str) -> TrialHandler:
        ds = TBDataset(name=self._dataset_name, version=self._dataset_version, registry_url=self._registry_url)
        paths = [p for p in ds.tasks if p.name == task_id]
        if not paths:
            raise ValueError(f"Task '{task_id}' not found in {self._dataset_name}=={self._dataset_version}")
        return TrialHandler(trial_name=f"vf-{uuid.uuid4().hex[:8]}", input_path=paths[0], output_path=self._runs_root)

    def _start_container(self, th: TrialHandler) -> Tuple[DockerComposeManager, TmuxSession]:
        dcm = DockerComposeManager(
            client_container_name=th.client_container_name,
            client_image_name=th.client_image_name,
            docker_compose_path=th.task_paths.docker_compose_path,
            docker_image_name_prefix=th.docker_image_name_prefix,
            no_rebuild=self._no_rebuild,
            cleanup=self._cleanup,
            sessions_logs_path=th.trial_paths.sessions_path,
            agent_logs_path=th.trial_paths.agent_logging_dir,
        )
        container = dcm.start()
        session = TmuxSession(
            session_name=th.client_container_name,
            container=container,
            commands_path=th.trial_paths.commands_path,
            disable_recording=bool(getattr(th.task, "disable_asciinema", False)),
        )
        session.start()
        return dcm, session

    def _run_tests_and_score(self, th: TrialHandler, session: TmuxSession, state: dict) -> None:
        session.copy_to_container(paths=[th.task_paths.run_tests_path],
                                  container_dir=str(DockerComposeManager.CONTAINER_TEST_DIR))
        if th.task_paths.test_dir.exists():
            session.copy_to_container(paths=[th.task_paths.test_dir],
                                      container_dir=str(DockerComposeManager.CONTAINER_TEST_DIR))
        test_timeout = (float(self._test_timeout_override) if self._test_timeout_override is not None
                        else float(getattr(th.task, "max_test_timeout_sec", 60.0)) * self._timeout_mult)
        cmd = ["timeout", f"{int(test_timeout)}s", "bash",
               str(DockerComposeManager.CONTAINER_TEST_DIR / th.task_paths.run_tests_path.name)]
        try:
            result = session.container.exec_run(cmd)
            post_test = result.output.decode(errors="replace") if hasattr(result, "output") else ""
            results = th.parser.parse(post_test) or {}
            # LAGUNA: keep the per-test breakdown for a dense reward, not just the binary verdict.
            passed = sum(1 for v in results.values() if v == UnitTestStatus.PASSED)
            state["tests_passed"] = passed
            state["tests_total"] = len(results)
            state["terminalbench_is_resolved"] = bool(results) and passed == len(results)
        except Exception:
            state["tests_passed"] = 0
            state["tests_total"] = state.get("tests_total", 0)
            state["terminalbench_is_resolved"] = False

    def _limit_output_length(self, output: str, max_bytes: int = 20000) -> str:
        if not isinstance(output, str):
            return ""
        b = output.encode("utf-8", errors="ignore")
        if len(b) <= max_bytes:
            return output
        half = max_bytes // 2
        return (f"{b[:half].decode('utf-8', errors='ignore')}\n"
                f"[... {len(b) - max_bytes} bytes omitted ...]\n{b[-half:].decode('utf-8', errors='ignore')}")

    async def rollout(self, client, model: str, prompt: List[dict], answer: str = "",
                      task: str = "default", info: dict | None = None,
                      sampling_args: dict | None = None, **kwargs) -> Tuple[List[dict], dict]:
        if not info or not info.get("task_id"):
            raise ValueError("rollout requires info['task_id']")
        th = self._resolve_task(str(info["task_id"]))
        dcm, session = self._start_container(th)
        try:
            pre = self._limit_output_length(session.capture_pane(capture_entire=False))
            initial = PROMPT_TEMPLATE.format(instruction=th.instruction, terminal_state=pre)
            agent_timeout = (float(self._agent_timeout_override) if self._agent_timeout_override is not None
                             else float(getattr(th.task, "max_agent_timeout_sec", 360.0)) * self._timeout_mult)
            sampling = {"temperature": 0.7, "top_p": 1.0, **(sampling_args or {})}
            messages, state = await super().rollout(
                client=client, model=model, prompt=[{"role": "user", "content": initial}],
                answer=answer, task=task, info=info, sampling_args=sampling,
                tb_session=session, agent_deadline=time.time() + agent_timeout,
            )
            state = dict(state or {})
            state["tb_session"] = session
            self._run_tests_and_score(th, session, state)
            return messages, state
        finally:
            for closer in (session.stop, dcm.stop):
                try:
                    closer()
                except Exception:
                    pass


def load_environment(
    *,
    dataset: str = "terminal-bench-core",
    dataset_version: str = "head",
    registry_url: str | None = None,
    task_ids: List[str] | None = None,        # LAGUNA: curate to a difficulty-appropriate subset
    exclude_task_ids: List[str] | None = None,
    n_tasks: int | None = None,
    runs_dir: str | None = None,
    timeout_multiplier: float | None = None,
    no_rebuild: bool = False,
    cleanup: bool = False,
    max_turns: int = 50,
    fn: str = "shaped",                        # LAGUNA: "shaped" (partial credit) | "binary"
    efficiency_weight: float = 0.1,
    **kwargs,
) -> vf.MultiTurnEnv:
    if dataset and "==" in dataset and dataset_version == "head":
        dataset, dataset_version = (s.strip() for s in dataset.split("==", 1))
    return TerminalBenchCuratedEnv(
        dataset_name=dataset, dataset_version=dataset_version, registry_url=registry_url,
        task_ids=task_ids, exclude_task_ids=exclude_task_ids, n_tasks=n_tasks, runs_dir=runs_dir,
        timeout_multiplier=timeout_multiplier, no_rebuild=no_rebuild, cleanup=cleanup,
        max_turns=max_turns, fn=fn, efficiency_weight=efficiency_weight,
    )
