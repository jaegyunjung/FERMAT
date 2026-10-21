import argparse
import json
from pathlib import Path
import torch
from fermat_release import (load_release, Tokenizer, embedding, next_distribution,
                            generate_consistent, load_next_record_head)

parser = argparse.ArgumentParser()
parser.add_argument('--release', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--patient', type=Path)
parser.add_argument('--device', default='cpu')
parser.add_argument('--new-events', type=int, default=8)
args = parser.parse_args()
torch.set_num_threads(2)
torch.manual_seed(42)
model = load_release(args.release, args.device)
tokenizer = Tokenizer(args.release)
events = json.loads((args.patient or args.release / 'examples/synthetic_patient.json').read_text())['events']
idx, age, typ, coverage = tokenizer.encode(events, args.device)
with torch.inference_mode():
    vector = embedding(model, idx, age, typ)
    distribution = next_distribution(model, idx, age, typ, tokenizer.allowed_outputs(args.device))
    probability, tokens = distribution.event_probability.topk(5)
    head = load_next_record_head(args.release, args.device)
    ranked = head.top_tokens(vector)[0].tolist()
    result, ages, types, _ = generate_consistent(
        model, idx, age, typ, max_new_tokens=args.new_events, max_age=float(age.max()) + 365.25 * 5,
        token_type_lookup=tokenizer.types, allowed_token_mask=tokenizer.allowed_outputs(args.device),
        termination_tokens=[tokenizer.by_key['SPECIAL:DEATH'][0]])
new = [(int(t), float(a), int(y)) for t, a, y in zip(result[0, idx.shape[1]:], ages[0, idx.shape[1]:], types[0, idx.shape[1]:]) if int(t) > 0]
print(json.dumps(dict(coverage=coverage, embedding_shape=list(vector.shape),
    same_day_probability=float(distribution.same_day_probability[0]),
    next_event_top5=[dict(token=tokenizer.labels[t], probability=p) for t, p in zip(tokens[0].tolist(), probability[0].tolist())],
    next_record_head_top10=[tokenizer.labels[t] for t in ranked],
    generated=[dict(token=tokenizer.labels[t], age_days=a) for t, a, _ in new],
    stopped_at_token_cap=len(new) == args.new_events and bool(new) and new[-1][2] != 6), indent=2))
