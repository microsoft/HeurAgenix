"""Console-only progress; never write model payloads or exception messages."""

from __future__ import annotations

import sys

from tqdm import tqdm


class EvaluationProgress:
    """Keep display counters independent of persisted evaluation artifacts."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.stream = sys.stderr
        self.bar = None
        self.attempts = 0
        self.correct = 0
        self.failures = 0

    def preparing(self, planned_graphs: int) -> None:
        if self.enabled:
            print(f"Preparing up to {planned_graphs} graphs: loading, sampling and verifying labels...",
                  file=self.stream, flush=True)

    def start(self, graphs: int, questions: int, repetitions: int) -> None:
        if not self.enabled:
            return
        print(f"Prepared {graphs} graphs, {questions} questions x {repetitions} repetitions.",
              file=self.stream, flush=True)
        self.bar = tqdm(total=questions * repetitions, desc="Evaluating", unit="call",
                        file=self.stream, disable=False, mininterval=1.0,
                        dynamic_ncols=True, ascii=True,
                        postfix={"accuracy": "n/a", "failures": 0})

    def advance(self, *, correct: bool, status: str) -> None:
        if not self.enabled:
            return
        self.attempts += 1
        self.correct += int(correct)
        self.failures += int(status != "success")
        if self.bar is not None:
            self.bar.set_postfix(accuracy=f"{self.correct / self.attempts:.2%}",
                                 failures=self.failures, refresh=False)
            self.bar.update(1)

    def finish(self, status: str) -> None:
        if not self.enabled:
            return
        if self.bar is not None:
            self.bar.close()
        accuracy = f"{self.correct / self.attempts:.2%}" if self.attempts else "n/a"
        print(f"Run {status}: {self.attempts} attempts, accuracy={accuracy}, failures={self.failures}.",
              file=self.stream, flush=True)