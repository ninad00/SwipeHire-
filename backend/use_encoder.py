import json
import torch
import torch.nn as nn
import numpy as np

from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, PeftModel, TaskType, get_peft_model


MODEL_NAME = "BAAI/bge-large-en-v1.5"


class MyModel(nn.Module):

    def __init__(
        self,
        base_model,
        classifier
    ):
        super().__init__()

        self.base_model = base_model
        self.classifier = classifier

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None
    ):

        if token_type_ids is None:

            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        else:

            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

        cls_output = (
            outputs.last_hidden_state[:, 0, :]
        )

        logits = self.classifier(
            cls_output
        )

        return logits


def load_cross_encoder(
    model_path,
    lora_path=None
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    base_model = AutoModel.from_pretrained(MODEL_NAME)
    
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=True,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "value"]
    )

    base_model = get_peft_model(
        base_model,
        peft_config
    )

    classifier = nn.Sequential(
        nn.Linear(1024,128),
        nn.Dropout(0.2),
        nn.Linear(128,1)
    )

    model = MyModel(
        base_model,
        classifier
    )

    model.load_state_dict(
        torch.load(
            "./best_model.pt",
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    return model, tokenizer, device


def predict_cross_scores(
    resume_text,
    candidate_jobs,
    model,
    tokenizer,
    device,
    batch_size=16
):

    pairs = []

    for job in candidate_jobs:

        job_text = (
            f"{job.get('title','')} "
            f"{job.get('description','')}"
        )

        pairs.append(
            (
                resume_text,
                job_text
            )
        )

    scores = []

    with torch.no_grad():

        for i in range(
            0,
            len(pairs),
            batch_size
        ):

            batch = pairs[
                i:i + batch_size
            ]

            encodings = tokenizer(
                [x[0] for x in batch],
                [x[1] for x in batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            encodings = {
                k: v.to(device)
                for k, v in encodings.items()
            }

            logits = model(
                encodings["input_ids"],
                encodings["attention_mask"],
                encodings.get(
                    "token_type_ids"
                )
            )

            batch_scores = (
                torch.sigmoid(logits)
                .squeeze(-1)
                .cpu()
                .numpy()
                .tolist()
            )

            scores.extend(
                batch_scores
            )

    return scores


def cross_encoder_rerank(
    job_dict,
    candidate_jobs,
    model,
    tokenizer,
    device,
    top_k=20
):

    resume_text = json.dumps(
        job_dict,
        sort_keys=True
    )

    scores = predict_cross_scores(
        resume_text,
        candidate_jobs,
        model,
        tokenizer,
        device
    )

    for job, score in zip(
        candidate_jobs,
        scores
    ):

        job["cross"] = float(score)

    candidate_jobs.sort(
        key=lambda x: x["cross"],
        reverse=True
    )

    return candidate_jobs[:top_k]