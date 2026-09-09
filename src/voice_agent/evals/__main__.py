from __future__ import annotations

import asyncio
import sys

from .runner import EvalRunner, build_offline_agent


async def _amain() -> int:
    agent, _metrics = build_offline_agent()
    scores = await EvalRunner(agent).run_all()
    failed = 0
    for score in scores:
        mark = "PASS" if score.passed else "FAIL"
        print(f"{mark} {score.scenario} tools={score.tools} cosine={score.cosine}")
        for reason in score.reasons:
            print(f"  - {reason}")
        failed += int(not score.passed)
    print(f"{len(scores) - failed}/{len(scores)} passed")
    return 1 if failed else 0


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
