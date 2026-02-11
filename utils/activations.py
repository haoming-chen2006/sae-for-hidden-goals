import torch


@torch.no_grad()
def get_activations(model, tokenizer, text_batch, hook_layer, seq_len, device):
    """Extract activations from a transformer layer.
    
    Returns:
        acts: (batch * seq_len, d_model) — flattened activation vectors
        attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding
        input_ids: (batch, seq_len) — token IDs
    """
    tokens = tokenizer(
        text_batch, return_tensors="pt", padding="max_length",
        truncation=True, max_length=seq_len,
    ).to(device)

    captured = {}

    def hook_fn(module, input, output):
        # Some models return tuple (hidden_states, ...), some just hidden_states
        if isinstance(output, tuple):
            captured["acts"] = output[0].detach()
        else:
            captured["acts"] = output.detach()

    target_layer = model.model.layers[hook_layer]
    handle = target_layer.register_forward_hook(hook_fn)

    model(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])
    handle.remove()

    acts = captured["acts"].float()
    batch_size = tokens["input_ids"].shape[0]
    d_model = acts.shape[-1]

    # Reshape to (batch, seq_len, d_model) regardless of how the hook returns it
    if acts.dim() == 2:
        # (total_tokens, d_model) — need to infer batch structure
        total_tokens = acts.shape[0]
        actual_seq_len = total_tokens // batch_size
        if total_tokens == batch_size * seq_len:
            acts = acts.reshape(batch_size, seq_len, d_model)
        else:
            # Hook returned unexpected size — just return what we have
            print(f"  [Warning] Hook returned {acts.shape}, expected ({batch_size}, {seq_len}, {d_model})")
            acts = acts.reshape(batch_size, -1, d_model)
    elif acts.dim() == 3:
        # Already (batch, seq, d_model) — verify batch matches
        if acts.shape[0] != batch_size:
            print(f"  [Warning] Hook batch={acts.shape[0]} vs tokenizer batch={batch_size}")

    return acts, tokens

class ActivationBuffer:
    """Accumulates activations from the LM and serves shuffled mini-batches."""

    def __init__(self, model, tokenizer, dataset_iterator,
                 hook_layer, seq_len, device,
                 buffer_size=2048, batch_size=256, refresh_fraction=0.5):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_iter = dataset_iterator
        self.hook_layer = hook_layer
        self.seq_len = seq_len
        self.device = device
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.refresh_fraction = refresh_fraction
        self.buffer = None
        self.pointer = 0

    def _collect_activations(self, n_needed):
        """Pull text batches and extract activations until we have n_needed vectors."""
        chunks = []
        collected = 0
        while collected < n_needed:
            try:
                text_batch = next(self.dataset_iter)
            except StopIteration:
                break
            acts, tokens = get_activations(
                self.model, self.tokenizer, text_batch,
                self.hook_layer, self.seq_len, self.device,
            )
            # Flatten to (total_tokens, d_model)
            acts_flat = acts.reshape(-1, acts.shape[-1])

            # Mask out padding — build mask from attention_mask, trimmed to match acts
            attn_mask = tokens["attention_mask"]  # (batch, seq_len)
            mask_flat = attn_mask.bool().reshape(-1)  # (batch * seq_len,)

            # If acts has fewer tokens than mask (hook returned fewer), trim the mask
            if acts_flat.shape[0] < mask_flat.shape[0]:
                mask_flat = mask_flat[: acts_flat.shape[0]]
            elif acts_flat.shape[0] > mask_flat.shape[0]:
                # More acts than mask entries — no padding info, keep all
                mask_flat = torch.ones(acts_flat.shape[0], dtype=torch.bool, device=mask_flat.device)

            acts_flat = acts_flat[mask_flat]

            # Drop rows that contain NaN or Inf (fp16 instability on some inputs)
            valid_mask = torch.isfinite(acts_flat).all(dim=-1)
            if not valid_mask.all():
                n_bad = (~valid_mask).sum().item()
                print(f"  [ActivationBuffer] Dropped {n_bad}/{acts_flat.shape[0]} NaN/Inf activation vectors")
                acts_flat = acts_flat[valid_mask]
            if acts_flat.shape[0] == 0:
                continue
            chunks.append(acts_flat)
            collected += acts_flat.shape[0]
        if not chunks:
            raise StopIteration("Dataset exhausted.")
        return torch.cat(chunks, dim=0)

    def _fill_buffer(self):
        """Fill buffer from scratch, then shuffle."""
        new_acts = self._collect_activations(self.buffer_size)
        self.buffer = new_acts[: self.buffer_size]
        self.buffer = self.buffer[torch.randperm(self.buffer.shape[0])]
        self.pointer = 0

    def _maybe_refresh(self):
        """Keep refresh_fraction of old data, fill rest with new, re-shuffle."""
        n_keep = int(self.buffer.shape[0] * self.refresh_fraction)
        n_new = self.buffer.shape[0] - n_keep
        old_acts = self.buffer[:n_keep]
        new_acts = self._collect_activations(n_new)[:n_new]
        self.buffer = torch.cat([old_acts, new_acts], dim=0)
        self.buffer = self.buffer[torch.randperm(self.buffer.shape[0])]
        self.pointer = 0

    def __iter__(self):
        return self

    def __next__(self):
        """Yield one mini-batch of shape (batch_size, d_model)."""
        if self.buffer is None:
            self._fill_buffer()
        if self.pointer + self.batch_size > self.buffer.shape[0]:
            self._maybe_refresh()
        batch = self.buffer[self.pointer : self.pointer + self.batch_size]
        self.pointer += self.batch_size
        return batch