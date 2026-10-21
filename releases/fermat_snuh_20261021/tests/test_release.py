from pathlib import Path
import json
import unittest
import torch
from fermat_release import load_release, Tokenizer, embedding, next_distribution, generate_consistent, load_next_record_head
from fermat_release.inference import draw_next

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = load_release(ROOT)
        cls.tokenizer = Tokenizer(ROOT)
        cls.events = json.loads((ROOT / 'examples/synthetic_patient.json').read_text())['events']
        cls.idx, cls.age, cls.typ, _ = cls.tokenizer.encode(cls.events)

    def test_parameter_and_token_identity(self):
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 65221122)
        self.assertIs(self.model.transformer.wte.weight, self.model.lm_head.weight)
        self.assertEqual(self.tokenizer.by_key['SPECIAL:DEATH'], (7, 6))
        self.assertEqual(self.tokenizer.by_key['SPECIAL:UNK_DX'], (3, 1))
        self.assertEqual(self.tokenizer.vocab_size, self.model.config.vocab_size)

    def test_unknown_and_lab_boundary(self):
        self.assertEqual(self.tokenizer.lookup('DX:999999999999'), (3, 1, True))
        with self.assertRaises(ValueError):
            self.tokenizer.lookup('CUSTOM:123')
        pair, cuts = next(iter(self.tokenizer.cutpoints.items()))
        val = cuts[0]
        expected = 1 + sum(val > x for x in cuts)
        self.assertTrue(self.tokenizer.numeric_lab_key(*pair, val).endswith(f'Q{expected:02d}'))
        with self.assertRaises(ValueError):
            self.tokenizer.encode([dict(age_days=-1, token_key='SEX:M')])

    def test_padding_and_embedding(self):
        expected = embedding(self.model, self.idx, self.age, self.typ)
        padded = embedding(self.model, torch.nn.functional.pad(self.idx, (0, 2)),
            torch.nn.functional.pad(self.age, (0, 2), value=-10000), torch.nn.functional.pad(self.typ, (0, 2)))
        torch.testing.assert_close(expected, padded, rtol=0, atol=0)
        self.assertEqual(tuple(expected.shape), (1, 640))
        self.assertTrue(torch.isfinite(expected).all())

    def test_joint_distribution(self):
        d = next_distribution(self.model, self.idx, self.age, self.typ, self.tokenizer.allowed_outputs())
        torch.testing.assert_close(d.event_probability.sum(-1), torch.ones(1))
        self.assertEqual(int((d.event_probability[:, ~self.tokenizer.allowed_outputs()] != 0).sum()), 0)
        # Force each branch using controlled uniforms. Check the event was drawn
        # from that branch, rather than independently from the mixed distribution.
        for u in (0., 0.999999):
            uniform = torch.tensor([[u, .5, .3]])
            token, gap, same = draw_next(d, uniform)
            conditional = d.same_day_events if bool(same[0]) else d.different_day_events
            expected = (conditional.cumsum(-1) <= .3).sum(-1)
            torch.testing.assert_close(token, expected)
            self.assertEqual(float(gap[0]) == 0, bool(same[0]))

    def test_terminal_death_and_cap(self):
        death = self.tokenizer.by_key['SPECIAL:DEATH'][0]
        args = dict(token_type_lookup=self.tokenizer.types, allowed_token_mask=self.tokenizer.allowed_outputs(), termination_tokens=[death])
        idx, age, typ, _ = self.tokenizer.encode([dict(age_days=100, token_key='SPECIAL:DEATH')])
        out, _, _, _ = generate_consistent(self.model, idx, age, typ, max_new_tokens=3, **args)
        self.assertEqual(out.shape[1], 1)
        out, _, _, _ = generate_consistent(self.model, self.idx, self.age, self.typ, max_new_tokens=0, **args)
        torch.testing.assert_close(out, self.idx)

    def test_head_output(self):
        head = load_next_record_head(ROOT)
        vector = embedding(self.model, self.idx, self.age, self.typ)
        with torch.inference_mode():
            logits = head(vector)
            top = head.top_tokens(vector)
        self.assertEqual(tuple(logits.shape), (1, 17460))
        self.assertEqual(len(set(top[0].tolist())), 10)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(all(t in self.tokenizer.types for t in top[0].tolist()))


if __name__ == '__main__':
    unittest.main()
