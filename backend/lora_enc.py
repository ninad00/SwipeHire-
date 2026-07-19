

import os
from pathlib import Path
from typing import Optional
import modal
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model



APP_NAME = "resume-job-fit-trainer"
MODEL_NAME = "BAAI/bge-large-en-v1.5"
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
OUTPUT_DIR = Path("/models")

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "pandas",
        "numpy",
        "peft",
        "accelerate",
    )
)
volume = modal.Volume.from_name("resume-data", create_if_missing=True)
output_volume = modal.Volume.from_name("model-storage", create_if_missing=True)

def prepare_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(data_dir / "final_dataset.csv")
    df_test = pd.read_csv(data_dir / "final_dataset_test.csv")

    label_map = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 1}
    df["label"] = df["label"].replace(label_map)
    df_test["label"] = df_test["label"].replace(label_map)

    val_df = df_test.sample(frac=0.2, random_state=42)
    df_test = df_test.drop(val_df.index)

    df = df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    return df, val_df, df_test


def tokenize_dataframe(
    tokenizer: AutoTokenizer,
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    max_length: int = 512,
) -> dict[str, torch.Tensor]:
    return tokenizer(
        df[col_a].astype(str).tolist(),
        df[col_b].astype(str).tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


class MyModel(nn.Module):
    def __init__(self, base_model: nn.Module, classifier: nn.Module) -> None:
        super().__init__()
        self.base_model = base_model
        self.classifier = classifier

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if token_type_ids is None:
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        else:
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits


def predict_df(
    df: pd.DataFrame,
    model: nn.Module,
    encodings: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int = 16,
) -> pd.DataFrame:
    model.eval()

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    token_type_ids = encodings.get("token_type_ids")

    n = len(df)
    all_scores = np.empty(n, dtype=np.float32)
    all_preds = np.empty(n, dtype=np.int64)

    with torch.inference_mode():
        for i in range(0, n, batch_size):
            slc = slice(i, i + batch_size)

            batch_input_ids = input_ids[slc].to(device)
            batch_attention_mask = attention_mask[slc].to(device)
            batch_token_type_ids = token_type_ids[slc].to(device) if token_type_ids is not None else None

            logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
            probs = torch.sigmoid(logits).squeeze(-1)
            preds = (probs > 0.5).long()

            all_scores[slc] = probs.detach().cpu().numpy()
            all_preds[slc] = preds.detach().cpu().numpy()

    df = df.copy()
    df["score"] = all_scores
    df["prediction"] = all_preds
    return df


def train(
    model: nn.Module,
    encodings: dict[str, torch.Tensor],
    labels: list[int],
    val_df: pd.DataFrame,
    val_encodings: dict[str, torch.Tensor],
    val_label_col: str,
    device: torch.device,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 1e-5,
) -> nn.Module:
    model.to(device)

    if hasattr(model, "base_model") and hasattr(model.base_model, "gradient_checkpointing_enable"):
        model.base_model.gradient_checkpointing_enable()

    train_dataset = TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"],
        encodings.get("token_type_ids", torch.zeros_like(encodings["input_ids"])),
        torch.tensor(labels, dtype=torch.float32).unsqueeze(1),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(
        [
            {"params": [p for n, p in model.named_parameters() if "classifier" not in n], "lr": lr},
            {"params": [p for n, p in model.named_parameters() if "classifier" in n], "lr": lr * 2},
        ],
        weight_decay=0.02,
    )

    scaler = torch.amp.GradScaler('cuda')
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            input_ids, attention_mask, token_type_ids, batch_labels = [x.to(device) for x in batch]

            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                logits = model(input_ids, attention_mask, token_type_ids)
                if torch.isnan(logits).any(): 
                    print("NaN loss detected, stopping training")
                    break
                loss = criterion(logits, batch_labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            probs = torch.sigmoid(logits).squeeze(-1)
            preds = (probs > 0.5).long()

            correct += (preds == batch_labels.squeeze(1).long()).sum().item()
            total += batch_labels.size(0)

            if step % 50 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | Step {step}/{len(train_loader)} "
                    f"| Loss: {total_loss / (step + 1):.4f} | Acc: {correct / max(total, 1):.4f}"
                )

        model.eval()
        val_result = predict_df(val_df, model, val_encodings, device, batch_size=batch_size)
        val_acc = (val_result["prediction"] == val_result[val_label_col]).mean()

        print(
            f"\nEpoch {epoch + 1} complete"
            f" | Train Loss: {total_loss / max(len(train_loader), 1):.4f}"
            f" | Train Acc: {correct / max(total, 1):.4f}"
            f" | Val Acc: {val_acc:.4f}"
        )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
            print(f"Best model saved (val_acc={val_acc:.4f})\n")
        else:
            print()

    print(f"Training complete. Best Val Acc: {best_val_acc:.4f}")
    return model


@app.function(image=image, gpu="L4", timeout=60 * 60 * 6,volumes={"/data": volume,"/models": output_volume})
def train_on_modal() -> str:
    torch.manual_seed(42)
    np.random.seed(42)

    df, val_df, df_test = prepare_data(DATA_DIR)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModel.from_pretrained(MODEL_NAME)

    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, 
        inference_mode=False, 
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.1,
        target_modules=["query", "value"] 
    )

    base_model=get_peft_model(base_model, peft_config)
    base_model.print_trainable_parameters()

    # for name, param in base_model.named_parameters():
    #     if  "classifier" in name :
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False

    classifier = nn.Sequential(
    nn.Linear(1024,128),
    nn.RELU(),
    nn.Dropout(0.2),
    nn.Linear(128, 1)
    )

    classifier.requires_grad_(True)

    model = MyModel(base_model, classifier)

    train_encodings = tokenize_dataframe(tokenizer, df, "resume_text", "job_description_text", max_length=512)
    
    val_encodings = tokenize_dataframe(tokenizer, val_df, "resume_text", "job_description_text", max_length=512)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = train(
        model=model,
        encodings=train_encodings,
        labels=df["label"].tolist(),
        val_df=val_df,
        val_encodings=val_encodings,
        val_label_col="label",
        device=device,
        epochs=8,
        batch_size=32,
        lr=1e-4,
    )

    torch.save(model.state_dict(), OUTPUT_DIR / "final_model.pt")
    test_encodings = tokenize_dataframe(tokenizer, df_test, "resume_text", "job_description_text", max_length=512)

    test_result = predict_df(df_test, model, test_encodings, torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    test_acc = (test_result["prediction"] == test_result["label"]).mean()
    print(f"\nTest Accuracy: {test_acc:.4f}")

    return str(OUTPUT_DIR / "best_model.pt")


@app.local_entrypoint()
def main() -> None:
    path = train_on_modal.remote()
    print(path)
