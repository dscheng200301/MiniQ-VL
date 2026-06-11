import unittest
from types import SimpleNamespace

import torch

from trainer.grpo_utils import compute_advantages
from trainer.train_dpo import compute_dpo_loss
from trainer.train_grpo import sample_group


class DummyTokenizer:
    pad_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        return ",".join(str(int(token)) for token in ids)


class DummyProcessor:
    tokenizer = DummyTokenizer()


class DummyGenerator:
    def __init__(self):
        self.next_token = 10

    def generate(self, input_ids, **kwargs):
        suffix = torch.tensor([[self.next_token]])
        self.next_token += 1
        return torch.cat([input_ids, suffix], dim=1)


class PreferenceModel(torch.nn.Module):
    def __init__(self, chosen_bias):
        super().__init__()
        self.chosen_bias = chosen_bias

    def forward(self, input_ids, **kwargs):
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, 8)
        logits[:, 1, 3] = self.chosen_bias
        logits[:, 1, 4] = -self.chosen_bias
        return SimpleNamespace(logits=logits)


class TrainingLogicTests(unittest.TestCase):
    def test_grpo_samples_independent_group_members(self):
        prompt = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        samples = sample_group(
            DummyGenerator(), DummyProcessor(), prompt, group_size=4,
            gen_kwargs={}, device="cpu",
        )

        self.assertEqual(len(samples), 4)
        self.assertEqual(len({sample["text"] for sample in samples}), 4)

    def test_group_advantages_have_signal_for_distinct_rewards(self):
        advantages = compute_advantages(torch.tensor([1.0, 2.0, 3.0, 4.0]), 4)
        self.assertGreater(float(advantages.abs().sum()), 0.0)
        self.assertAlmostEqual(float(advantages.mean()), 0.0, places=6)

    def test_group_advantages_are_finite_for_single_sample_group(self):
        advantages = compute_advantages(torch.tensor([1.0]), 1)
        self.assertTrue(bool(torch.isfinite(advantages).all()))

    def test_dpo_prefers_policy_that_improves_over_reference(self):
        chosen_ids = torch.tensor([[1, 2, 3]])
        rejected_ids = torch.tensor([[1, 2, 4]])
        attention_mask = torch.ones_like(chosen_ids)
        response_mask = torch.tensor([[0, 0, 1]])
        reference = PreferenceModel(chosen_bias=0.0)

        good_loss, _, _ = compute_dpo_loss(
            PreferenceModel(chosen_bias=2.0), reference,
            chosen_ids, attention_mask, rejected_ids, attention_mask,
            response_mask, response_mask,
        )
        bad_loss, _, _ = compute_dpo_loss(
            PreferenceModel(chosen_bias=-2.0), reference,
            chosen_ids, attention_mask, rejected_ids, attention_mask,
            response_mask, response_mask,
        )

        self.assertLess(float(good_loss), float(bad_loss))


if __name__ == "__main__":
    unittest.main()
