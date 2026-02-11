"""
SAE Feature Evaluation & Interpretability

Flow:
  1. Load trained SAE + LM
  2. Run text through LM → hook activations → SAE encode
  3. Store the full activation matrix + token IDs to /home/haoming/SAELens/my_project/activations/
  4. Extract top-10 tokens per feature from the stored matrix
  5. Serve a local HTML dashboard to search and browse features
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
from tqdm import tqdm
from http.server import HTTPServer, SimpleHTTPRequestHandler
import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.activations import get_activations
from utils.dataset import text_batch_iterator
from models.simple_top_k import SparseAutoencoder

SAVE_DIR = Path("/home/haoming/SAELens/my_project/activations")


# ============ 1. Load SAE from checkpoint ============

def load_sae(checkpoint_path, d_in, d_hidden, top_k, device):
    sae = SparseAutoencoder(d_in, d_hidden, top_k)
    sae.load_state_dict(torch.load(checkpoint_path, map_location=device))
    sae.to(device)
    sae.eval()
    return sae


# ============ 2. Build and save activation matrix ============

@torch.no_grad()
def build_and_save_activations(model, tokenizer, sae, texts, hook_layer, seq_len, device):
    """
    Run all text through LM → SAE, store the full sparse activation matrix,
    token IDs, and per-sequence IDs to disk.
    Only stores activations for REAL tokens (padding is excluded).
    """
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    all_z = []
    all_token_ids = []       # flat list of token IDs aligned with all_z rows
    all_input_ids = []       # (n_sequences, seq_len) for context windows
    all_seq_indices = []     # which sequence each row came from
    all_tok_positions = []   # which position in the sequence
    total_mse = 0.0
    total_exp_var = 0.0
    total_l0 = 0.0
    n_batches = 0
    seq_counter = 0

    for text_batch in tqdm(texts, desc="Building activation matrix"):
        # Single tokenization — get_activations returns both acts and tokens
        acts, tokens = get_activations(model, tokenizer, text_batch, hook_layer, seq_len, device)
        input_ids = tokens["input_ids"]         # (batch, seq_len)
        attn_mask = tokens["attention_mask"]     # (batch, seq_len)

        acts = acts.to(dtype=next(sae.parameters()).dtype)
        batch_size = acts.shape[0]

        # Flatten to (batch*seq, d_model), filter padding
        acts_flat = acts.reshape(-1, acts.shape[-1])
        ids_flat = input_ids.reshape(-1).cpu()
        mask_flat = attn_mask.bool().reshape(-1)

        # Build sequence index and position arrays
        seq_ids = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len).reshape(-1) + seq_counter
        tok_pos = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).reshape(-1)

        # Keep only real (non-padding) tokens
        acts_real = acts_flat[mask_flat]
        ids_real = ids_flat[mask_flat.cpu()]
        seq_ids_real = seq_ids[mask_flat.cpu()]
        tok_pos_real = tok_pos[mask_flat.cpu()]

        # Drop NaN/Inf rows
        valid = torch.isfinite(acts_real).all(dim=-1)
        if not valid.all():
            n_bad = (~valid).sum().item()
            print(f"  Dropped {n_bad}/{acts_real.shape[0]} NaN/Inf activation vectors")
            acts_real = acts_real[valid]
            ids_real = ids_real[valid.cpu()]
            seq_ids_real = seq_ids_real[valid.cpu()]
            tok_pos_real = tok_pos_real[valid.cpu()]

        if acts_real.shape[0] == 0:
            seq_counter += batch_size
            continue

        # SAE encode
        sae_out = sae(acts_real)
        z = torch.relu(sae.encode(acts_real))
        z_sparse = sae.topk_gating(z)

        total_mse += (sae_out - acts_real).pow(2).sum(dim=-1).mean().item()
        total_exp_var += (1.0 - (acts_real - sae_out).var() / acts_real.var()).item()
        total_l0 += (z_sparse > 0).float().sum(dim=-1).mean().item()
        n_batches += 1

        all_z.append(z_sparse.cpu())
        all_token_ids.append(ids_real)
        all_input_ids.append(input_ids.cpu())
        all_seq_indices.append(seq_ids_real)
        all_tok_positions.append(tok_pos_real)

        seq_counter += batch_size

    all_z = torch.cat(all_z, dim=0)
    all_token_ids = torch.cat(all_token_ids, dim=0)
    all_input_ids = torch.cat(all_input_ids, dim=0)
    all_seq_indices = torch.cat(all_seq_indices, dim=0)
    all_tok_positions = torch.cat(all_tok_positions, dim=0)

    # Save to disk
    torch.save(all_z, SAVE_DIR / "activation_matrix.pt")
    torch.save(all_token_ids, SAVE_DIR / "token_ids.pt")
    torch.save(all_input_ids, SAVE_DIR / "input_ids_per_seq.pt")
    torch.save(all_seq_indices, SAVE_DIR / "seq_indices.pt")
    torch.save(all_tok_positions, SAVE_DIR / "tok_positions.pt")

    metrics = {
        "mse": total_mse / max(n_batches, 1),
        "l0": total_l0 / max(n_batches, 1),
        "explained_var": total_exp_var / max(n_batches, 1),
        "n_tokens": int(all_z.shape[0]),
        "d_hidden": int(all_z.shape[1]),
        "seq_len": seq_len,
    }
    with open(SAVE_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved activation matrix {tuple(all_z.shape)} to {SAVE_DIR}")
    print(f"  activation_matrix.pt  ({all_z.nbytes / 1e9:.2f} GB)")
    print(f"  token_ids.pt  (real tokens only, no padding)")
    print(f"  input_ids_per_seq.pt")
    print(f"  metrics.json")
    return metrics


# ============ 3. Load from disk and extract top tokens ============

def load_and_extract(tokenizer, top_n=10):
    """Load saved activation matrix, extract top-N tokens per feature."""
    print(f"Loading activation matrix from {SAVE_DIR}...")
    all_z = torch.load(SAVE_DIR / "activation_matrix.pt", map_location="cpu")
    all_token_ids = torch.load(SAVE_DIR / "token_ids.pt", map_location="cpu")
    all_input_ids = torch.load(SAVE_DIR / "input_ids_per_seq.pt", map_location="cpu")
    all_seq_indices = torch.load(SAVE_DIR / "seq_indices.pt", map_location="cpu")
    all_tok_positions = torch.load(SAVE_DIR / "tok_positions.pt", map_location="cpu")
    with open(SAVE_DIR / "metrics.json") as f:
        metrics = json.load(f)

    seq_len = metrics["seq_len"]
    n_tokens = all_z.shape[0]
    d_hidden = all_z.shape[1]

    print(f"  {n_tokens} real tokens (padding excluded), {d_hidden} features")

    # Feature density
    feature_fire_counts = (all_z > 0).float().sum(dim=0)
    feature_density = (feature_fire_counts / n_tokens).tolist()
    n_dead = sum(1 for d in feature_density if d < 1e-6)
    metrics["n_alive_features"] = d_hidden - n_dead
    metrics["n_dead_features"] = n_dead
    metrics["feature_density"] = feature_density

    # Top-k per alive feature
    alive_indices = (feature_fire_counts > 0).nonzero(as_tuple=True)[0]
    feature_tops = {}

    for feat_idx in tqdm(alive_indices.tolist(), desc="Extracting top tokens"):
        col = all_z[:, feat_idx]
        k = min(top_n, int((col > 0).sum().item()))
        if k == 0:
            continue
        vals, positions = torch.topk(col, k)

        entries = []
        for v, pos in zip(vals.tolist(), positions.tolist()):
            seq_idx = all_seq_indices[pos].item()
            tok_idx = all_tok_positions[pos].item()

            # Decode the specific token
            token_str = tokenizer.decode([all_token_ids[pos].item()])

            # Context window: 5 tokens before and after
            row_ids = all_input_ids[seq_idx].tolist()
            start = max(0, tok_idx - 5)
            end = min(seq_len, tok_idx + 6)
            context = tokenizer.decode(row_ids[start:end], skip_special_tokens=False)
            entries.append((v, token_str, context))
        feature_tops[feat_idx] = entries

    return feature_tops, metrics


# ============ 3b. Query a single feature ============

def query_feature(tokenizer, feature_idx, top_n=10):
    """Look up a single feature by index and return its top-N activating tokens."""
    print(f"Loading activation matrix from {SAVE_DIR}...")
    all_z = torch.load(SAVE_DIR / "activation_matrix.pt", map_location="cpu")
    all_token_ids = torch.load(SAVE_DIR / "token_ids.pt", map_location="cpu")
    all_input_ids = torch.load(SAVE_DIR / "input_ids_per_seq.pt", map_location="cpu")
    all_seq_indices = torch.load(SAVE_DIR / "seq_indices.pt", map_location="cpu")
    all_tok_positions = torch.load(SAVE_DIR / "tok_positions.pt", map_location="cpu")
    with open(SAVE_DIR / "metrics.json") as f:
        metrics = json.load(f)

    seq_len = metrics["seq_len"]
    d_hidden = all_z.shape[1]

    if feature_idx < 0 or feature_idx >= d_hidden:
        print(f"Error: feature index {feature_idx} out of range [0, {d_hidden})")
        return

    col = all_z[:, feature_idx]
    n_firing = int((col > 0).sum().item())
    density = n_firing / all_z.shape[0]

    k = min(top_n, n_firing)
    if k == 0:
        print(f"Feature {feature_idx} is dead (never fires).")
        return

    vals, positions = torch.topk(col, k)

    print(f"\n=== Feature {feature_idx} ===")
    print(f"  Density:  {density:.6f} ({n_firing}/{all_z.shape[0]} tokens)")
    print(f"  Top {k} activating tokens:\n")

    for rank, (v, pos) in enumerate(zip(vals.tolist(), positions.tolist()), 1):
        seq_idx = all_seq_indices[pos].item()
        tok_idx = all_tok_positions[pos].item()
        token_str = tokenizer.decode([all_token_ids[pos].item()])
        row_ids = all_input_ids[seq_idx].tolist()
        start = max(0, tok_idx - 5)
        end = min(seq_len, tok_idx + 6)
        context = tokenizer.decode(row_ids[start:end], skip_special_tokens=False)
        print(f"  #{rank:2d}  act={v:.4f}  token='{token_str}'")
        print(f"        context: ...{context}...")
    print()


# ============ 3. Generate HTML dashboard ============

def generate_html(feature_tops, metrics, output_path):
    """Build a self-contained HTML file for browsing features."""

    # Prepare feature data as JSON
    features_json = {}
    for feat_idx, tops in sorted(feature_tops.items()):
        features_json[feat_idx] = [
            {"token": t, "activation": round(a, 4), "context": c}
            for a, t, c in tops
        ]

    density_list = [round(d, 8) for d in metrics['feature_density']]

    # Write JSON data to a separate file (avoids inline encoding issues)
    data_path = Path(output_path).parent / "feature_data.js"
    with open(data_path, "w", encoding="utf-8") as f:
        f.write("const DATA = ")
        json.dump(features_json, f, ensure_ascii=True)
        f.write(";\nconst DENSITY = ")
        json.dump(density_list, f, ensure_ascii=True)
        f.write(";\n")
    print(f"  feature_data.js ({data_path.stat().st_size / 1e6:.1f} MB)")

    # Write the HTML file (small, no embedded data)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>SAE Feature Browser</title>
<style>
  body {{ font-family: monospace; max-width: 900px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #eee; }}
  h1 {{ color: #e94560; }}
  .metrics {{ background: #16213e; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
  .metrics span {{ margin-right: 20px; }}
  .search {{ width: 100%; padding: 10px; font-size: 16px; margin-bottom: 15px; background: #0f3460; color: #eee; border: 1px solid #e94560; border-radius: 4px; }}
  .feature {{ background: #16213e; margin: 10px 0; padding: 12px; border-radius: 8px; cursor: pointer; }}
  .feature:hover {{ background: #0f3460; }}
  .feature-header {{ font-weight: bold; color: #e94560; }}
  .token-list {{ display: none; margin-top: 8px; }}
  .token-list.open {{ display: block; }}
  .token-entry {{ padding: 4px 0; border-bottom: 1px solid #1a1a2e; }}
  .tok {{ background: #e94560; color: white; padding: 2px 6px; border-radius: 3px; }}
  .act {{ color: #00d2ff; }}
  .ctx {{ color: #888; font-size: 12px; }}
  .dead {{ color: #555; }}
</style></head><body>
<h1>SAE Feature Browser</h1>
<div class="metrics">
  <span>MSE: {metrics['mse']:.4f}</span>
  <span>L0: {metrics['l0']:.1f}</span>
  <span>Explained Var: {metrics['explained_var']:.3f}</span>
  <span>Alive: {metrics['n_alive_features']}</span>
  <span>Dead: {metrics['n_dead_features']}</span>
  <span>Tokens: {metrics['n_tokens']}</span>
</div>
<input class="search" type="text" placeholder="Search by feature index..." oninput="filterFeatures(this.value)">
<div id="features"></div>
<script src="feature_data.js"></script>
<script>
function esc(s) {{ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }}

function renderFeatures(filter) {{
  const container = document.getElementById('features');
  container.innerHTML = '';
  const keys = Object.keys(DATA).map(Number).sort((a,b)=>a-b);
  let shown = 0;
  for (const idx of keys) {{
    if (filter !== '' && !String(idx).startsWith(filter)) continue;
    if (shown > 200) break;
    const tops = DATA[idx];
    const div = document.createElement('div');
    div.className = 'feature';
    const density = DENSITY[idx] ? DENSITY[idx].toExponential(2) : '0';
    const header = document.createElement('div');
    header.className = 'feature-header';
    header.textContent = 'Feature ' + idx + ' (density: ' + density + ', top act: ' + tops[0].activation + ')';
    const tokenList = document.createElement('div');
    tokenList.className = 'token-list';
    tops.forEach(function(t) {{
      const entry = document.createElement('div');
      entry.className = 'token-entry';
      entry.innerHTML = '<span class="tok">' + esc(t.token) + '</span> <span class="act">' + t.activation + '</span> <span class="ctx">' + esc(t.context) + '</span>';
      tokenList.appendChild(entry);
    }});
    header.onclick = function() {{ tokenList.classList.toggle('open'); }};
    div.appendChild(header);
    div.appendChild(tokenList);
    container.appendChild(div);
    shown++;
  }}
  if (shown === 0) container.innerHTML = '<div class="dead">No matching features.</div>';
}}
function filterFeatures(v) {{ renderFeatures(v); }}
renderFeatures('');
</script></body></html>"""

    Path(output_path).write_text(html, encoding="utf-8")
    print(f"Dashboard saved to {output_path}")
    return output_path


# ============ 4. Serve locally ============

def serve_dashboard(html_path, port=8765):
    import os
    os.chdir(str(Path(html_path).parent))
    filename = Path(html_path).name

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                self.path = f'/{filename}'
            return super().do_GET()

    server = HTTPServer(('0.0.0.0', port), Handler)
    print(f"\n  Open http://localhost:{port} to browse features\n")
    server.serve_forever()


# ============ Main ============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae_checkpoint", type=str, default="/home/haoming/SAELens/checkpoints/sae_epoch_4.pt",
                        help="Path to SAE .pt file (only needed if activations not yet saved)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--adapter_dir", type=str, default="/home/haoming/SAELens/checkpoints/checkpoint-263309")
    parser.add_argument("--d_in", type=int, default=4096)
    parser.add_argument("--d_hidden", type=int, default=4096 * 4)
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--hook_layer", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--n_batches", type=int, default=500)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild activations even if saved")
    parser.add_argument("--no_serve", action="store_true")
    parser.add_argument("--feature", type=int, default=None,
                        help="Query a single feature by index (skips dashboard)")
    parser.add_argument("--top_n", type=int, default=10,
                        help="Number of top tokens to show per feature")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tokenizer is always needed (for decoding token IDs back to text)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Phase 1: Build activation matrix (skip if already saved)
    required_files = ["activation_matrix.pt", "token_ids.pt", "input_ids_per_seq.pt",
                      "seq_indices.pt", "tok_positions.pt", "metrics.json"]
    saved = all((SAVE_DIR / f).exists() for f in required_files)
    if not saved or args.rebuild:
        if not args.sae_checkpoint:
            parser.error("--sae_checkpoint is required when activations haven't been saved yet")

        print("Loading SAE...")
        sae = load_sae(args.sae_checkpoint, args.d_in, args.d_hidden, args.top_k, device)

        print("Loading LM...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        model = PeftModel.from_pretrained(base_model, args.adapter_dir)
        model = model.merge_and_unload()
        model.resize_token_embeddings(len(tokenizer))
        model.to(device)
        model.eval()

        text_iter = text_batch_iterator("NeelNanda/pile-10k", "train", batch_size=4, format_fn=None)
        text_batches = []
        for i, batch in enumerate(text_iter):
            if i >= args.n_batches:
                break
            text_batches.append(batch)

        print(f"Building activation matrix from {len(text_batches)} batches...")
        build_and_save_activations(model, tokenizer, sae, text_batches,
                                   args.hook_layer, args.seq_len, device)

        # Free GPU memory — no longer needed
        del model, base_model, sae
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        print(f"Activations already saved at {SAVE_DIR}, skipping LM inference.")
        print("  (use --rebuild to recompute)")

    # If user asked for a single feature, query it and exit
    if args.feature is not None:
        query_feature(tokenizer, args.feature, top_n=args.top_n)
        return

    # Phase 2: Load matrix from disk, extract top tokens, launch dashboard
    feature_tops, metrics = load_and_extract(tokenizer, top_n=args.top_n)

    print(f"\n=== Metrics ===")
    print(f"  MSE:            {metrics['mse']:.4f}")
    print(f"  L0:             {metrics['l0']:.1f}")
    print(f"  Explained Var:  {metrics['explained_var']:.3f}")
    print(f"  Alive features: {metrics['n_alive_features']}")
    print(f"  Dead features:  {metrics['n_dead_features']}")

    # Generate HTML dashboard
    out_path = str(SAVE_DIR / "feature_dashboard.html")
    generate_html(feature_tops, metrics, out_path)

    if not args.no_serve:
        serve_dashboard(out_path, args.port)


if __name__ == "__main__":
    main()