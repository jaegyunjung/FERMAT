"""One real optimizer update on fabricated events; never modifies shipped weights."""
import argparse
import json
from pathlib import Path
import torch
from fermat_release import load_release, Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--release', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--device', default='cpu')
args = parser.parse_args()
torch.set_num_threads(2)
torch.manual_seed(42)
model = load_release(args.release, args.device).train()
events = json.loads((args.release / 'examples/synthetic_patient.json').read_text())['events']
idx, age, typ, _ = Tokenizer(args.release).encode(events, args.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
_, losses, _ = model(idx[:, :-1], age[:, :-1], typ[:, :-1],
    targets=idx[:, 1:], targets_age=age[:, 1:], target_token_type=typ[:, 1:], return_attention=False)
loss = losses['loss_ce'] + 0.3 * (losses['loss_same_day'] + losses['loss_dt'])
assert torch.isfinite(loss)
loss.backward()
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
optimizer.step()
print(json.dumps(dict(loss=float(loss.detach()), gradient_norm=float(grad_norm),
                     optimizer_steps=1, saved_weights_modified=False)))
