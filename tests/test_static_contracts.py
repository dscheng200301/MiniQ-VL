import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def source(path):
    return (ROOT / path).read_text(encoding="utf-8")


def top_level_function(path, name):
    tree = ast.parse(source(path))
    return next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


class TrainingContractTests(unittest.TestCase):
    def test_model_wrapper_is_present_and_exposes_required_interface(self):
        text = source("model/qwen_vl.py")
        self.assertIn("class QwenVLMConfig", text)
        self.assertIn("class QwenVLM", text)
        self.assertIn("def forward", text)
        self.assertIn("def generate", text)

    def test_grpo_samples_group_with_repeated_generate_calls(self):
        node = top_level_function("trainer/train_grpo.py", "sample_group")
        calls = [
            n for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "generate"
        ]
        loops = [
            n for n in ast.walk(node)
            if isinstance(n, ast.For) and ast.unparse(n.iter) == "range(group_size)"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(loops), 1)

    def test_dpo_uses_reference_model_and_response_masks(self):
        node = top_level_function("trainer/train_dpo.py", "compute_dpo_loss")
        args = [arg.arg for arg in node.args.args]
        self.assertIn("ref_model", args)
        self.assertIn("chosen_response_mask", args)
        self.assertIn("rejected_response_mask", args)

    def test_lora_is_injected_before_ddp_wrap(self):
        text = source("trainer/train_sft.py")
        self.assertIn("get_peft_model(inner.model", text)
        self.assertLess(text.index("get_peft_model(inner.model"), text.index("DDP must wrap"))

    def test_launchers_resolve_project_directory(self):
        for path in ("scripts/train_sft.sh", "scripts/train_dpo.sh", "scripts/train_grpo.sh"):
            self.assertIn('dirname "${BASH_SOURCE[0]}"', source(path))

    def test_sft_resume_skips_consumed_batches(self):
        text = source("trainer/train_sft.py")
        self.assertIn("itertools.islice(loader, start_step, None)", text)


if __name__ == "__main__":
    unittest.main()
