import re
from datasets import load_dataset


def format_persona_chat(row):
    """Format a Synthetic-Persona-Chat row with special tokens."""
    persona = row.get("user 1 personas", "")
    text = f"<persona>{persona}</persona>"

    conversation = row.get("Best Generated Conversation", "")
    turns = re.split(r'(?=User [12]:)', conversation)
    for turn in turns:
        turn = turn.strip()
        if not turn:
            continue
        match = re.match(r"User (\d+):\s*(.+)", turn, re.DOTALL)
        if not match:
            continue
        user_num, message = match.group(1), match.group(2).strip()
        if user_num == "1":
            text += f"<user1>{message}</user1>"
        else:
            text += f"<user2>{message}</user2>"
    return text


def text_batch_iterator(dataset_name, split, batch_size, format_fn=format_persona_chat):
    """Yields lists of formatted text strings from an HF dataset. Loops forever."""
    while True:
        ds = load_dataset(dataset_name, split=split, streaming=True)
        batch = []
        for row in ds:
            if format_fn:
                text = format_fn(row)
            else:
                text = row.get("text", "") if isinstance(row, dict) else str(row)
            batch.append(text)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
        print("  [Dataset] Reached end of dataset, looping from start.")