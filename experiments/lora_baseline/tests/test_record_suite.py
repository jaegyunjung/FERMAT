import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_record_suite import final_contrasts, plan


class RecordSuiteTests(unittest.TestCase):
    def test_fixed_order_and_matched_schedule(self):
        jobs = plan()
        self.assertEqual([j['name'] for j in jobs],
                         [f'{group}_seed{seed}' for group in ('A', 'C', 'lora') for seed in (1, 2, 3)])
        for c, lora in zip(jobs[3:6], jobs[6:9]):
            for field in ('batch', 'epochs', 'patience', 'warmup', 'clip', 'seed'):
                self.assertEqual(c[field], lora[field])
        self.assertEqual(jobs[0]['batch'], 256)
        self.assertEqual(jobs[0]['epochs'], 60)

    def test_secondary_pairing_uses_ids_not_row_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for job in plan():
                folder = root / job['name']
                folder.mkdir()
                rows = '1,10\n2,20\n' if job['group'] == 'lora' else '2,15\n1,5\n'
                (folder / 'test_patient_metrics.csv').write_text('patient_id,recall_at_10\n' + rows)
            final_contrasts(root)
            result = json.loads((root / 'secondary_contrasts.json').read_text())
            for contrast in result['comparisons'].values():
                self.assertEqual(contrast['point'], 5.)
                self.assertEqual(contrast['ci_low'], 5.)
                self.assertEqual(contrast['ci_high'], 5.)
                self.assertEqual(contrast['n_patients'], 2)
            (root / 'C_seed3/test_patient_metrics.csv').write_text('patient_id,recall_at_10\n9,15\n1,5\n')
            with self.assertRaisesRegex(ValueError, 'patient sets differ'):
                final_contrasts(root)
