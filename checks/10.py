# Exercise 1
import torch

m = tiny_classifier(48, 3)
assert m(input_ids=torch.tensor([[1, 2]])).logits.shape == (1, 3)

# Exercise 2
from unittest.mock import patch

with (
    patch("transformers.AutoTokenizer.from_pretrained", return_value="t") as t,
    patch(
        "transformers.AutoModelForSequenceClassification.from_pretrained",
        return_value="m",
    ) as m,
):
    assert local_classifier("snap") == ("t", "m")
    for loader in (t, m):
        loader.assert_called_once_with(
            "snap", local_files_only=True, trust_remote_code=False
        )

# Exercise 3
from unittest.mock import patch

with patch("transformers.pipeline", return_value="pipe") as p:
    assert classification_pipeline("m", "t") == "pipe"
    p.assert_called_once_with(
        "text-classification", model="m", tokenizer="t", device=-1
    )

# Exercise 4
def tok(rows, **kw):
    assert kw == {"truncation": True, "max_length": 3, "padding": False}
    return {"input_ids": [[1]] * len(rows)}


assert prepare_batch({"text": ["a"], "label": [1]}, tok, 3)["labels"] == [1]

# Exercise 5
c = padding_collator("tokenizer", 4)
assert c.pad_to_multiple_of == 4 and c.return_tensors == "pt"
with raises(ValueError):
    padding_collator("t", 0)

# Exercise 6
import torch
from transformers import BertConfig, BertForSequenceClassification

m = BertForSequenceClassification(
    BertConfig(
        vocab_size=12,
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=1,
        intermediate_size=16,
    )
)
before = m.classifier.weight.detach().clone()
loss = fine_tune_step(
    m,
    {"input_ids": torch.tensor([[1, 2]]), "labels": torch.tensor([1])},
    torch.optim.SGD(m.parameters(), lr=0.1),
)
assert loss > 0 and not torch.equal(before, m.classifier.weight)

# Exercise 7
import torch

m = torch.nn.Module()
m.base_model = torch.nn.Linear(2, 2)
m.head = torch.nn.Linear(2, 1)
assert {id(p) for p in head_parameters(m)} == {id(p) for p in m.head.parameters()}

# Exercise 8
import torch
from types import SimpleNamespace as NS

m = NS(generate=lambda **kw: torch.tensor([[0, 4, 7]]))
t = NS(batch_decode=lambda x, skip_special_tokens: x.tolist())
assert completions(m, t, {"input_ids": torch.tensor([[0, 4]])}, 2) == [[7]]

# Exercise 9
import tempfile

with tempfile.TemporaryDirectory() as path:
    args = training_args(path, 73)
    assert (
        args.seed == args.data_seed == 73
        and args.load_best_model_at_end
        and args.eval_strategy == "epoch"
    )

# Exercise 10
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

with tempfile.TemporaryDirectory() as tmp:
    dest = Path(tmp) / "v1"
    export_snapshot(
        NS(save_pretrained=lambda p: (p / "model").touch()),
        NS(save_pretrained=lambda p: (p / "vocab").touch()),
        dest,
    )
    assert {f.name for f in dest.iterdir()} == {"model", "vocab"}
    with raises(FileExistsError):
        export_snapshot(None, None, dest)

from unittest.mock import Mock

with tempfile.TemporaryDirectory() as tmp:
    dest = Path(tmp) / "incomplete"
    model = NS(save_pretrained=lambda p: (p / "model").touch())
    with raises(OSError):
        export_snapshot(
            model, NS(save_pretrained=Mock(side_effect=OSError("disk full"))), dest
        )
    assert not dest.exists() and list(Path(tmp).iterdir()) == []
