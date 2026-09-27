from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1] / "paper"


class PaperStructureTests(unittest.TestCase):
    def setUp(self):
        self.main = (ROOT / "main.tex").read_text()
        self.body, self.appendices = self.main.split("\\appendix", 1)
        self.body = self.body.replace("\\input{main-text}", (ROOT / "main-text.tex").read_text())

    def test_title_and_supplied_author_order(self):
        self.assertIn(r"\newcommand{\bench}{GraphDecide}", self.main)
        self.assertIn(r"\title{\bench: Benchmarking Decision Models on Graphs}", self.main)
        author = self.main.split(r"\author{", 1)[1].split(r"\hypersetup", 1)[0]
        self.assertLess(author.index("Xianliang Yang"), author.index("Yapu Zhang"))
        self.assertLess(author.index("Yapu Zhang"), author.index("Li Zhao"))
        for email in ("xianya@microsoft.com", "zhangyapu@bjut.edu.cn", "lizo@microsoft.com"):
            self.assertIn(f"mailto:{email}", author)
        self.assertIn(r"Li Zhao\textsuperscript{1}\thanks{Corresponding author:", author)
        self.assertIn(r"Xianliang Yang\textsuperscript{1}", author)
        self.assertIn(r"Yapu Zhang\textsuperscript{2}", author)
        self.assertIn(r"\textsuperscript{1}Microsoft", author)
        self.assertIn(r"\textsuperscript{2}Beijing University of Technology", author)
        self.assertIn("pdfauthor={Xianliang Yang, Yapu Zhang, Li Zhao}", self.main)
        self.assertNotIn("author information to be supplied", self.main)

    def test_complete_matrix_precedes_capability_interpretation(self):
        self.assertLess(
            self.body.index("\\label{sec:full-matrix}"),
            self.body.index("\\label{sec:discussion}"),
        )
        for filename in ("coverage", "exact", "gaps"):
            self.assertEqual(
                self.body.count(f"\\input{{generated-twelve-model/{filename}.tex}}"), 1
            )

    def test_gap_is_primary_and_optimum_counts_are_supplementary(self):
        directive = "\\input{generated-twelve-model/optimal.tex}"
        self.assertNotIn(directive, self.body)
        self.assertEqual(self.appendices.count(directive), 1)
        self.assertIn("primary construction-quality metric is mean additive", self.body)
        self.assertIn("Primary construction outcome: mean additive", self.body)
        self.assertIn("not establish better mean quality", self.body)

    def test_historical_evidence_is_mandatory_appendix_content(self):
        for filename in (
            "original-results", "confirmation-details", "interface-controls",
            "protocol-details", "related_work", "capability-analysis"
        ):
            directive = f"\\input{{{filename}}}"
            self.assertNotIn(directive, self.body)
            self.assertEqual(self.appendices.count(directive), 1)
        self.assertIn(
            "\\input{generated-gpt/anchor.tex}",
            (ROOT / "interface-controls.tex").read_text(),
        )

    def test_all_mandatory_inputs_exist_and_labels_are_unique(self):
        texts = []

        def visit(path, ancestors):
            self.assertNotIn(path, ancestors, f"Cyclic input: {path}")
            text = path.read_text()
            texts.append(text)
            for name in re.findall(r"\\input\{([^}]+)\}", text):
                target = ROOT / name
                if not target.suffix:
                    target = target.with_suffix(".tex")
                self.assertTrue(target.is_file(), f"Missing input: {target}")
                visit(target, (*ancestors, path))

        visit(ROOT / "main.tex", ())
        combined = "\n".join(texts)
        labels = re.findall(r"\\label\{([^}]+)\}", combined)
        labels.extend(re.findall(r"legend to name=([^\],\s]+)", combined))
        self.assertEqual(len(labels), len(set(labels)), "Duplicate LaTeX labels")
        references = re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", combined)
        self.assertFalse(set(references) - set(labels), "Undefined internal references")

    def test_scope_and_research_questions_remain_explicit(self):
        for question in ("RQ1:", "RQ2:", "RQ3:"):
            self.assertIn(question, self.body)
        abstract = self.body.split("\\begin{abstract}", 1)[1].split(
            "\\end{abstract}", 1
        )[0]
        self.assertIn("upper reference points", abstract)
        self.assertNotIn("Thinking", abstract)
        capability = (ROOT / "capability-analysis.tex").read_text()
        self.assertIn("not identified", capability)
        self.assertIn("not all twelve", capability)
        self.assertIn("fitted reliability bands", capability)

    def test_larger_models_do_not_imply_parameter_or_latency_equivalence(self):
        for phrase in (
            "11,520 scheduled episode evaluations",
            "Qwen3.8-27B", "Qwen2.5-72B", "older-generation",
            "not a scale-only", "No equal-hardware latency comparison",
            "720 task/size/configuration cells",
        ):
            self.assertIn(phrase, self.body)

    def test_main_text_has_explicit_page_boundary_before_references(self):
        self.assertIn("\\input{main-text}\n\\clearpage\n\\bibliographystyle", self.main)
        self.assertIn("\\clearpage\n\\appendix", self.main)
        self.assertIn("\\label{sec:main-end}", self.body)
        self.assertIn("\\documentclass[11pt]{article}", self.main)
        self.assertIn("\\usepackage[margin=1in]{geometry}", self.main)

    def test_code_release_points_to_requested_repository_and_branch(self):
        self.assertIn(r"\url{https://github.com/microsoft/HeurAgenix.git}", self.body)
        self.assertIn(r"\texttt{graph\_benchmark}", self.body)
        self.assertIn(r"\texttt{benchmarks/graphdecide}", self.body)
        self.assertIn("not included in this code release", self.body)
