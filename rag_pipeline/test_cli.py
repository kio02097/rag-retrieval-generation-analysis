"""Fast CLI checks; no model downloads, GPU or external service required."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from .__main__ import main, parse_args


class CLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "questions.csv"
        self.dataset.write_text("id,body,type,documents,exact_answer\n", encoding="utf-8")
        self.prompt = self.root / "prompt.txt"
        self.prompt.write_text("{question}", encoding="utf-8")
        # A marker is sufficient for input-only dry-run; it is not a valid index.
        (self.root / "chroma.sqlite3").touch()
        self.args = ["--dataset", str(self.dataset), "--prompt", str(self.prompt),
                     "--chroma-dir", str(self.root), "--collections", "bge_C"]

    def test_dry_run_has_no_output_directory_side_effect(self):
        out = self.root / "outputs"
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            main(self.args + ["--dry-run", "--output-dir", str(out)])
        self.assertIn("bge_C", captured.getvalue())
        self.assertFalse(out.exists())

    def test_invalid_k_and_collection_fail_before_model_loading(self):
        for extra in (["--k", "0"], ["--collections", "bge_CM"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    parse_args(self.args + extra)
                self.assertEqual(raised.exception.code, 2)

    def test_missing_columns_fail(self):
        self.dataset.write_text("id,body\n", encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(self.args)


if __name__ == "__main__":
    unittest.main()
