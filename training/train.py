import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import yaml
from dataclasses import dataclass
from pathlib import Path
import torch
from torch.optim import AdamW
from tqdm import tqdm
from utils.activations import ActivationBuffer
from utils.dataset import text_batch_iterator
from models.simple_top_k import SparseAutoencoder

@dataclass 
class TrainConfig:
    """Training configuration."""
    model_name: str = "Qwen/Qwen3-8B"
    seq_len: int = 256
    
    # Dataset
    dataset_name: str = "NeelNanda/pile-10k"
    dataset_split: str = "train"
    
    # SAE dimensions — d_in must match model hidden_size (4096 for Qwen3-8B)
    d_in: int = 4096
    d_hidden: int = 4096 * 4   # 4x expansion
    top_k: int = 32
    hook_layer: int = 16       # middle layer of 36

    # Training
    seed: int = 42
    epochs: int = 20
    lr: float = 3e-4
    sae_batch_size: int = 256      # SAE training batch (num activation vectors)
    text_batch_size: int = 4       # how many texts to feed the LM at once
    buffer_size: int = 4096        # activation vectors held in buffer
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    steps_per_epoch: int = 1000
    log_steps: int = 10
    
    # Output
    out_dir: str = "/home/haoming/SAELens/checkpoints"
    adapter_dir: str = "/home/haoming/SAELens/checkpoints/checkpoint-263309"
    sae_dir: str = None
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def load_config(path: str) -> TrainConfig:
    """Load config from YAML file."""
    data = yaml.safe_load(Path(path).read_text())
    return TrainConfig(**data)


def train_sae(cfg, model, tokenizer):
    """Full SAE training: build buffer, create SAE, run training loop."""

    # --- SAE ---
    sae = SparseAutoencoder(cfg.d_in, cfg.d_hidden, cfg.top_k).to(cfg.device)
    if cfg.sae_dir is not None:
        sae.load_state_dict(torch.load(cfg.sae_dir, map_location=cfg.device))

    # --- Dataset iterator (streams formatted persona chat text) ---
    dataset_iter = text_batch_iterator(
        cfg.dataset_name, cfg.dataset_split, cfg.text_batch_size,
        format_fn=None,
    )

    # --- Activation buffer (handles LM inference + shuffling) ---
    model.eval()
    buffer = ActivationBuffer(
        model=model,
        tokenizer=tokenizer,
        dataset_iterator=dataset_iter,
        hook_layer=cfg.hook_layer,
        seq_len=cfg.seq_len,
        device=cfg.device,
        buffer_size=cfg.buffer_size,
        batch_size=cfg.sae_batch_size,
    )

    # --- Optimizer (no scheduler for now as you noted) ---
    optimizer = AdamW(sae.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # --- Training loop ---
    sae.train()
    for epoch in range(cfg.epochs):
        pbar = tqdm(range(cfg.steps_per_epoch), desc=f"Epoch {epoch}")

        for step in pbar:
            # Get a batch of activation vectors from the buffer
            # sae_in shape: (sae_batch_size, d_in)
            sae_in = next(buffer).to(cfg.device).float()

            # Forward: SAE reconstructs the activation
            sae_out = sae(sae_in)

            # Loss: MSE summed over d_in, averaged over batch
            # (this is how SAELens computes it — sum features, mean batch)
            recon_loss = (sae_out - sae_in).pow(2).sum(dim=-1).mean()
            loss = recon_loss

            if torch.isnan(loss):
                print(f"\n[!] NaN loss at epoch {epoch} step {step}")
                print(f"    sae_in  has NaN: {torch.isnan(sae_in).any().item()}, inf: {torch.isinf(sae_in).any().item()}")
                print(f"    sae_out has NaN: {torch.isnan(sae_out).any().item()}, inf: {torch.isinf(sae_out).any().item()}")
                print(f"    sae_in  range: [{sae_in.min().item():.4f}, {sae_in.max().item():.4f}]")
                raise RuntimeError("NaN loss detected — see diagnostics above.")

            # Backward + clip + step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg.max_grad_norm)
            optimizer.step()

            # Logging
            if step % cfg.log_steps == 0:
                with torch.no_grad():
                    # L0: average number of active features per sample
                    z = sae.encode(sae_in)
                    z = torch.relu(z)
                    z = sae.topk_gating(z)
                    l0 = (z > 0).float().sum(dim=-1).mean().item()

                    # Explained variance: how much of the input variance is captured
                    exp_var = 1.0 - (sae_in - sae_out).var() / sae_in.var()

                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    l0=f"{l0:.1f}",
                    exp_var=f"{exp_var.item():.3f}",
                )

        # Save SAE checkpoint each epoch
        save_path = Path(cfg.out_dir) / f"sae_epoch_{epoch}.pt"
        torch.save(sae.state_dict(), save_path)
        print(f"Saved SAE checkpoint: {save_path}")

    return sae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else TrainConfig()
    if args.epochs:
        cfg.epochs = args.epochs
    if args.lr:
        cfg.lr = args.lr

    print("\n=== Loading Model ===")

    # Load base Qwen3-8B, then apply the LoRA adapter on top
    # (the checkpoint is a LoRA adapter, not a full model)
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        dtype=torch.float16 if cfg.device == "cuda" else torch.float32,
    )
    model = PeftModel.from_pretrained(base_model, cfg.adapter_dir)
    model = model.merge_and_unload()  # merge LoRA weights into base for clean hooks
    model = model.to(cfg.device)
    model.eval()
    print(f"Model loaded: {cfg.model_name} + LoRA from {cfg.adapter_dir}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Train the SAE
    sae = train_sae(cfg, model, tokenizer)

main()
    