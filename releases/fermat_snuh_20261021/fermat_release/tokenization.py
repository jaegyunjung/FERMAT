"""Frozen vocabulary adapter for already harmonized, single-patient events.

This is not an automatic mapping from Korean insurance codes to OMOP concepts.
Numeric labs must first be unit-harmonized and aggregated to daily medians.
"""
import csv
import json
import math
from pathlib import Path
import torch


class Tokenizer:
    def __init__(self, root):
        root = Path(root)
        with (root / 'artifacts/token_registry.csv').open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        self.by_key = {r['token_key']: (int(r['token_id']) + 1, int(r['token_type_id'])) for r in rows}
        self.labels = {idx: key for key, (idx, _) in self.by_key.items()}
        self.types = {idx: typ for idx, typ in self.by_key.values()}
        self.vocab_size = max(self.types) + 1
        self.cutpoints = {(int(r['measurement_concept_id']), int(r['unit_concept_id'])): r['cutpoints']
                          for r in json.loads((root / 'artifacts/lab_cutpoints.json').read_text())}

    def numeric_lab_key(self, concept_id, unit_concept_id, value):
        if not math.isfinite(float(value)):
            raise ValueError('Lab value must be finite')
        concept_id, unit_concept_id = int(concept_id), int(unit_concept_id)
        cuts = self.cutpoints.get((concept_id, unit_concept_id))
        if cuts is None:
            return f'LAB_TEST:{concept_id}'
        quantile = 1 + sum(float(value) > c for c in cuts)
        return f'LAB:{concept_id}:{unit_concept_id}:Q{quantile:02d}'

    def lookup(self, key):
        if key in self.by_key:
            return (*self.by_key[key], False)
        domain = key.split(':', 1)[0]
        fallback = {'DX': 'DX', 'RX': 'RX', 'PX': 'PX', 'LAB': 'LAB',
                    'LAB_TEST': 'LAB', 'LAB_CAT': 'LAB'}.get(domain)
        if fallback is None:
            raise ValueError(f'Unsupported token key: {key}')
        return (*self.by_key[f'SPECIAL:UNK_{fallback}'], True)

    def encode(self, events, device='cpu', truncate=True):
        """Return [1,T] tensors plus a coverage report. Retain repeated events.

        Each event has integer age_days and either token_key, or the numeric lab
        fields measurement_concept_id, unit_concept_id, value_as_number.
        """
        rows, unknown, count = [], 0, 0
        for event in events:
            age = float(event['age_days'])
            if not math.isfinite(age) or age < 0 or not age.is_integer():
                raise ValueError('Input age_days must be a nonnegative integer')
            key = event.get('token_key')
            if key is None:
                key = self.numeric_lab_key(event['measurement_concept_id'],
                                           event.get('unit_concept_id', 0), event['value_as_number'])
            idx, typ, missed = self.lookup(key)
            if typ in (0, 8):
                raise ValueError('PAD and NO_EVENT are not observed clinical events')
            unknown += int(missed or key.startswith('SPECIAL:UNK_'))
            count += 1
            rows.append((int(age), 0 if typ == 7 else typ, idx, typ))
        if not rows:
            raise ValueError('At least one observed event is required')
        rows.sort(key=lambda row: row[:3])
        used = rows[-512:] if truncate else rows
        idx = torch.tensor([[r[2] for r in used]], dtype=torch.long, device=device)
        age = torch.tensor([[r[0] for r in used]], dtype=torch.float32, device=device)
        typ = torch.tensor([[r[3] for r in used]], dtype=torch.long, device=device)
        return idx, age, typ, dict(input_events=count, retained_events=len(used),
                                   unknown_events=unknown, unknown_fraction=unknown / count)

    def allowed_outputs(self, device='cpu'):
        mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=device)
        for idx, typ in self.types.items():
            if typ in (1, 2, 3, 6):
                mask[idx] = True
        return mask
