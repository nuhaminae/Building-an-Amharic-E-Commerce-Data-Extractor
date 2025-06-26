import torch

def predict_with_logits(model, tokenizer, id2label, tokens):
    """
    Inference function compatible with SHAP or manual use. Handles both strings and tokenized input.
    """
    model.eval()

    is_split = isinstance(tokens, list)
    enc = tokenizer(
        tokens,
        return_tensors="pt",
        is_split_into_words=is_split,
        truncation=True,
        #return_offsets_mapping=False,
        padding=True
    )

    with torch.no_grad():
        output = model(**enc)

    logits = output.logits.squeeze(0)  # [seq_len, num_labels]
    probs = torch.nn.functional.softmax(logits, dim=-1)
    pred_ids = torch.argmax(probs, dim=-1)

    word_ids = enc.word_ids()
    token_strs = tokenizer.convert_ids_to_tokens(enc["input_ids"][0], skip_special_tokens=True)

    decoded_labels = [
        id2label[i.item()]
        for idx, i in enumerate(pred_ids)
        if word_ids[idx] is not None
    ]

    if is_split:
        # Map predictions back to original token list
        word_to_label = {}
        for idx, word_id in enumerate(word_ids):
            if word_id is not None and word_id not in word_to_label:
                word_to_label[word_id] = id2label[pred_ids[idx].item()]
        word_level_labels = [word_to_label.get(i, "O") for i in range(len(tokens))]
    else:
        # Return original decoded sequence
        word_level_labels = decoded_labels

    return {
        "tokens": token_strs,
        "word_ids": word_ids,
        "logits": logits.detach().numpy(),
        "probabilities": probs.detach().numpy(),
        "predicted_labels": decoded_labels,
        "predicted_labels_word_level": word_level_labels
    }
