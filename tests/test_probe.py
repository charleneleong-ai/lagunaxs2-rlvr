import json
from pathlib import Path

from laguna_finetune.probe import build_eval_command, normalize_records


class TestBuildEvalCommand:
    def _cmd(self, **overrides) -> list[str]:
        kw = dict(env="e", model="m", provider="anthropic", num_examples=5,
                  rollouts_per_example=2, max_tokens=1024, temperature=0.7,
                  output_dir=Path("/tmp/x"))
        kw.update(overrides)
        return build_eval_command(**kw)

    def test_forwards_temperature(self):
        cmd = self._cmd()
        assert cmd[cmd.index("--temperature") + 1] == "0.7"

    def test_forwards_env_args_as_json(self):
        cmd = self._cmd(env_args={"split": "curated_easy", "max_turns": 50})
        assert json.loads(cmd[cmd.index("--env-args") + 1]) == {"split": "curated_easy", "max_turns": 50}

    def test_omits_env_args_when_empty(self):
        assert "--env-args" not in self._cmd(env_args=None)


def test_normalize_records_projects_onto_success_reward():
    raw = [{"ok": True, "r": 1.0, "extra": 9}, {"ok": False, "r": 0.0}]
    assert normalize_records(raw, "ok", "r") == [
        {"success": True, "reward": 1.0},
        {"success": False, "reward": 0.0},
    ]
