"""Tests for concurrent lake tool state persistence."""

import json
import os
import tempfile
import threading
import unittest

from src.opencode_lake_tool import (
    _load_state_payload,
    _state_path,
    init_state,
    load_state,
    save_state,
)


class TestLakeToolStatePersistence(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.workdir = self.tmp

    def test_concurrent_save_state_stays_valid_json(self) -> None:
        errors: list[BaseException] = []

        def worker(idx: int) -> None:
            try:
                state = load_state(self.workdir)
                retrieved = list(state.get("retrieved_passage_ids") or [])
                retrieved.append(f"P{idx}")
                state["retrieved_passage_ids"] = retrieved
                state["attempts_this_step"] = int(state.get("attempts_this_step") or 0) + 1
                save_state(self.workdir, state)
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        path = _state_path(self.workdir)
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertEqual(len(payload["retrieved_passage_ids"]), 20)
        self.assertEqual(payload["attempts_this_step"], 20)

    def test_load_state_recovers_first_json_object(self) -> None:
        path = _state_path(self.workdir)
        os.makedirs(self.workdir, exist_ok=True)
        good = init_state()
        good["step"] = 3
        bad = json.dumps(good, indent=2) + '\n  "sql_attempts": {}\n}\n'
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(bad)

        recovered = _load_state_payload(path)
        self.assertEqual(recovered["step"], 3)

        save_state(self.workdir, recovered)
        with open(path, encoding="utf-8") as handle:
            json.load(handle)


if __name__ == "__main__":
    unittest.main()
