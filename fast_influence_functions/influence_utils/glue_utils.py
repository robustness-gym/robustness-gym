from transformers import BertForSequenceClassification


def freeze_BERT_parameters(model: BertForSequenceClassification, verbose: bool = True) -> None:
    # https://github.com/huggingface/transformers/issues/400
    if not isinstance(model, BertForSequenceClassification):
        raise TypeError

    # Table 3 in https://arxiv.org/pdf/1911.03090.pdf
    params_to_freeze = [
        "bert.embeddings.",
        "bert.encoder.layer.0.",
        "bert.encoder.layer.1.",
        "bert.encoder.layer.2.",
        "bert.encoder.layer.3.",
        "bert.encoder.layer.4.",
        "bert.encoder.layer.5.",
        "bert.encoder.layer.6.",
        "bert.encoder.layer.7.",
        "bert.encoder.layer.8.",
        "bert.encoder.layer.9.",
    ]
    for name, param in model.named_parameters():
        # if "classifier" not in name:  # classifier layer
        #     param.requires_grad = False

        if any(pfreeze in name for pfreeze in params_to_freeze):
            param.requires_grad = False

    if verbose is True:
        num_trainable_params = sum([
            p.numel() for n, p in model.named_parameters()
            if p.requires_grad])
        trainable_param_names = [
            n for n, p in model.named_parameters()
            if p.requires_grad]
        print(f"Params Trainable: {num_trainable_params}\n\t" +
              f"\n\t".join(trainable_param_names))
