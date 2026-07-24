#!/usr/bin/env python3
"""Shared four-arm ADM strategy definitions for CCW and FERMAT rollout."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class StrategySpec:
    name: str
    deadline_months: int
    mode: str  # "init_by" or "no_init_through"


STRATEGY_SPECS = (
    StrategySpec("INIT_WITHIN_3M", 3, "init_by"),
    StrategySpec("INIT_WITHIN_6M", 6, "init_by"),
    StrategySpec("INIT_WITHIN_12M", 12, "init_by"),
    StrategySpec("NO_INIT_WITHIN_12M", 12, "no_init_through"),
)
STRATEGY_BY_NAME = {item.name: item for item in STRATEGY_SPECS}


def strategy_action(strategy, candidate_day, deadline_day, adm_already_seen):
    """Return the constraint to apply to the next generated event.

    `force_adm_at_deadline` means that the next unconstrained event would occur
    after the grace period while the initiation requirement remains unmet.
    `suppress_adm` means ADM tokens are forbidden before/at the no-init deadline.
    """
    spec = STRATEGY_BY_NAME[str(strategy)]
    candidate_day = float(candidate_day)
    deadline_day = float(deadline_day)
    if candidate_day < 0:
        return {"force_adm_at_deadline": False, "suppress_adm": False}
    if spec.mode == "init_by":
        return {
            "force_adm_at_deadline": bool(
                not adm_already_seen and candidate_day > deadline_day
            ),
            "suppress_adm": False,
        }
    if spec.mode == "no_init_through":
        return {
            "force_adm_at_deadline": False,
            "suppress_adm": bool(candidate_day <= deadline_day),
        }
    raise ValueError(spec.mode)


def self_test():
    assert strategy_action("INIT_WITHIN_3M", 91, 90, False)["force_adm_at_deadline"]
    assert not strategy_action("INIT_WITHIN_3M", 89, 90, False)["force_adm_at_deadline"]
    assert not strategy_action("INIT_WITHIN_3M", 100, 90, True)["force_adm_at_deadline"]
    assert strategy_action("NO_INIT_WITHIN_12M", 365, 365, False)["suppress_adm"]
    assert not strategy_action("NO_INIT_WITHIN_12M", 366, 365, False)["suppress_adm"]
    print("SELF_TEST_OK four_strategy_state_machine")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("use --self-test")
    self_test()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
