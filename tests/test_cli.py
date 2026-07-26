"""Every CLI command honours the three graduated smoke levels.

The suite runs with no dataset and no weights, so this is what establishes that
the operational surface actually works end to end. The uniformity is asserted
rather than assumed: a suite where some tools printed a contract report and exited
0 while others fell back to default paths and exited non-zero would leave an exit
code meaning something different from one command to the next, and the
convention was applied inconsistently -- five tools printed a contract report and
exited 0 while three fell back to default paths and exited non-zero -- so the
uniformity is asserted rather than assumed.
"""

from __future__ import annotations

import json

import pytest

from koa_multimodal.cli.__main__ import main
from koa_multimodal.training.stages import STAGE_NAMES

#: Commands whose default (no-flag) invocation must degrade to a contract report
#: rather than demanding inputs this checkout does not have.
ALL_COMMANDS = [
    ["check", "env"],
    ["check", "data", "--allow-empty"],
    ["check", "pipeline", "--synthetic"],
    ["train", "rckf"],
    ["ensemble", "stack"],
    ["ensemble", "oracle"],
    ["ensemble", "cmodes"],
    ["qat", "train"],
    ["qat", "export"],
    ["records", "validate"],
    ["pipeline", "list"],
    ["serve", "--check"],
]

#: The subset that also accepts ``--dry-run``. ``check env``, ``pipeline list``
#: and ``serve --check`` are already read-only reports, so a dry run of them
#: would be the same call twice.
DRY_RUNNABLE = [
    ["check", "data", "--allow-empty"],
    ["train", "rckf"],
    ["ensemble", "stack"],
    ["ensemble", "oracle"],
    ["ensemble", "cmodes"],
    ["qat", "train"],
    ["qat", "export"],
    ["records", "validate"],
]


def run(argv):
    return main(argv)


@pytest.mark.parametrize("argv", DRY_RUNNABLE, ids=lambda a: " ".join(a))
def test_dry_run_exits_zero_and_emits_json(argv, capsys):
    assert run(argv + ["--dry-run"]) == 0
    assert isinstance(json.loads(capsys.readouterr().out), dict)


@pytest.mark.parametrize("argv", ALL_COMMANDS, ids=lambda a: " ".join(a))
def test_contract_check_exits_zero_and_emits_json(argv, capsys):
    assert run(argv) == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, dict)


@pytest.mark.parametrize("stage", STAGE_NAMES)
def test_every_stage_resolves(stage, capsys):
    assert run(["train", stage, "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)["plan"]
    assert plan["stage"]["stage"] == stage
    assert plan["stage"]["batchSize"] >= 1
    assert plan["stage"]["effectiveBatchSize"] >= plan["stage"]["batchSize"]


def test_qat_stage_resolves_to_the_fusion_modality(capsys):
    """No modality is ever 'qat', so a batch-size branch keyed on one would be
    unreachable. The stage registry resolves it to FUSION instead."""

    run(["train", "qat", "--dry-run"])
    plan = json.loads(capsys.readouterr().out)["plan"]
    assert plan["stage"]["modality"] == "fusion"
    assert plan["stage"]["distills"] is True


def test_pipeline_list_finds_all_six_sections(capsys):
    assert run(["pipeline", "list"]) == 0
    pipelines = json.loads(capsys.readouterr().out)["pipelines"]
    assert len(pipelines) == 6
    assert {entry["name"] for entry in pipelines} >= {"01_data", "06_deployment"}


@pytest.mark.parametrize(
    "name", ["01_data", "02_unimodal", "03_fusion", "04_oof", "05_ensemble", "06_deployment"]
)
def test_every_pipeline_dry_runs(name, capsys):
    assert run(["pipeline", "run", name, "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["steps"], f"{name} resolved no steps"
    assert payload["level"] == "dry-run"


def test_synthetic_pipeline_check_forwards_every_route(capsys):
    assert run(["check", "pipeline", "--synthetic"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["routes"]) == 5
    assert all(route["probabilitiesSumToOne"] for route in payload["routes"])
    assert all(route["thresholdLogits"][1] == 4 for route in payload["routes"])
    assert payload["router"]["isPosteriorRoute"] is True
    assert payload["router"]["selectorInputDim"] == 16


def test_records_validate_reports_pending(capsys):
    """The record must be honestly pending: no training has run in this checkout."""

    assert run(["records", "validate", "--all"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["allValid"] is True
    assert any(record["status"] == "pending" for record in payload["records"])


def test_every_manifest_step_names_a_real_command():
    """A manifest naming a command that does not exist is a broken run book.

    ``koa pipeline run`` only *describes* steps, so a bad target would never fail
    at dry-run time and the manifest would quietly document a workflow nobody can
    execute.
    """

    from koa_multimodal.cli.__main__ import build_parser
    from koa_multimodal.cli.commands.pipeline import _list_pipelines, _load_manifest

    actions = build_parser()._subparsers._group_actions[0]
    known = {
        name: set(sub._subparsers._group_actions[0].choices)
        if sub._subparsers is not None
        else set()
        for name, sub in actions.choices.items()
    }

    problems = []
    for entry in _list_pipelines():
        for step in _load_manifest(entry["name"]).get("step", []):
            command = step.get("command")
            if command not in known:
                problems.append(f"{entry['name']}: unknown command {command!r}")
                continue
            operation = step.get("operation") or step.get("stage")
            if known[command] and operation and operation not in known[command]:
                # 'train' takes a stage positionally rather than a subcommand.
                if command != "train":
                    problems.append(
                        f"{entry['name']}: {command} has no {operation!r} "
                        f"(has {sorted(known[command])})"
                    )
    assert not problems, "pipeline manifests reference missing CLI targets:\n  " + "\n  ".join(problems)


def test_unknown_stage_is_rejected():
    with pytest.raises(SystemExit):
        run(["train", "not_a_stage", "--dry-run"])


def test_pool_files_load(capsys):
    for pool in ("xray", "fusion"):
        assert run(["ensemble", "stack", "--pool", pool, "--dry-run"]) == 0
        plan = json.loads(capsys.readouterr().out)["plan"]
        assert plan["pool"]["size"] == 4
        assert plan["requireOof"] is True
        assert plan["pool"]["defaultCandidateId"] in [
            member["candidateId"] for member in plan["pool"]["members"]
        ]
