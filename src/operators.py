# weights growth operators
import torch


def double_split_matrix_weight(x, is_grad, is_avg_sq):
    embed_dim = x.shape[0]
    split_dim = x.shape[1] // x.shape[0]
    y_shape = [2 * i for i in x.shape]
    y = x.new_zeros(*y_shape)

    for split_idx in range(split_dim):
        start_idx, end_idx = split_idx * x.shape[0], (1 + split_idx) * x.shape[0]
        y_split = double_matrix_weight(x[:, start_idx:end_idx], is_grad, is_avg_sq)
        y[:, start_idx * 2 : end_idx * 2] = y_split.detach().clone()

    return y


def double_split_bias(x, embed_dim, is_grad, is_avg_sq):
    split_dim = x.shape[0] // embed_dim
    y_shape = [2 * i for i in x.shape]
    y = x.new_zeros(*y_shape)

    for split_idx in range(split_dim):
        start_idx, end_idx = split_idx * embed_dim, (1 + split_idx) * embed_dim
        y[start_idx * 2 : start_idx * 2 + embed_dim] = x[start_idx:end_idx].detach().clone()
        y[start_idx * 2 + embed_dim : end_idx * 2] = x[start_idx:end_idx].detach().clone()

    if is_grad:
        y /= 2.0 if not is_avg_sq else 4.0

    return y


def double_matrix_weight(x, is_grad, is_avg_sq):
    # x = (n, m), returns y = (2 * n, 2 * m), used for FF layers
    y_shape = [2 * i for i in x.shape]
    y = x.new_zeros(*y_shape)

    x_shape = x.shape
    y[: x_shape[0], : x_shape[1]] = x.detach().clone()
    y[-x_shape[0] :, -x_shape[1] :] = x.detach().clone()
    if is_grad:
        y /= 2.0 if not is_avg_sq else 4.0

    return y


def double_bias(x, is_grad, is_avg_sq):
    # x = (n, ), returns y = (2 * n, ), used for bias weights
    y_shape = [2 * i for i in x.shape]
    y = x.new_zeros(*y_shape)

    x_shape = x.shape

    y[: x_shape[0]] = x.detach().clone()
    y[-x_shape[0] :] = x.detach().clone()
    if is_grad:
        y /= 2.0 if not is_avg_sq else 4.0

    return y


def double_embedding(x, is_grad, is_avg_sq):
    # x = (vocab size, M), returns y = (vocab size, 2 * M), used for embedding layers
    y_shape = [i for i in x.shape]
    y_shape[1] *= 2
    y = x.new_zeros(*y_shape)

    x_shape = x.shape
    y[:, : x_shape[1]] = x.detach().clone()
    y[:, -x_shape[1] :] = x.detach().clone()
    if is_grad:
        y /= 2.0 if not is_avg_sq else 4.0
    # if args.noise_std is not None and args.noise_std != 0.0:
    #     y += torch.normal(mean=0, std=args.noise_std, size=y.shape)
    return y


def double_param(key, weight, is_double_embedding, is_grad, is_avg_sq):
    if "lm_head" in key:  # for roberta
        # the lm_head is a linear layer then the softmax layer
        if "dense" in key:
            # this is the linear layer - need to expand as other linear layers
            if "weight" in key:
                return double_matrix_weight(weight, is_grad=is_grad, is_avg_sq=is_avg_sq)
            elif "bias" in key:
                return double_bias(weight, is_grad=is_grad, is_avg_sq=is_avg_sq)

        elif "layer_norm" in key or "ln" in key:
            # layer norm is weight * (x - mean)/std + bias, so need to divide both weight and bias by 2.0
            new_weight = double_bias(weight, is_grad=False, is_avg_sq=False)
            if not is_grad:
                new_weight /= 2.0
            return new_weight
        elif "weight" in key:
            return double_embedding(weight, is_grad=is_grad, is_avg_sq=is_avg_sq)
        elif "bias" in key:
            # this is the bias parameter added for the final softmax logit, shape (vocab_size, )
            return weight

    elif "pooler" in key:
        # don't think this pooler is used without next-sentence-prediction in bert, but we'll double it anyway
        if "weight" in key:
            return double_matrix_weight(weight, is_grad=is_grad, is_avg_sq=is_avg_sq)
        elif "bias" in key:
            return double_bias(weight, is_grad=is_grad, is_avg_sq=is_avg_sq)

    elif "cls" in key:
        # the masked LM head.
        # in BERT it is top layer activations -> dense with same hidden dim -> activation -> layer norm -> decoder
        # where the decoder is linear that has the same weights as the word embeddings and a new bias layer
        # to maintain the same loss, we want the logit outputs to remain the same - so can do it by duplicating
        # the word embeddings / bias, and dividing the input by 2.  We accomplish the two division by modifying the
        # layer norm parameters right before prediction.
        #
        # cls.predictions.bias torch.Size([30522])
        # cls.predictions.transform.dense.weight torch.Size([768, 768])
        # cls.predictions.transform.dense.bias torch.Size([768])
        # cls.predictions.transform.LayerNorm.weight torch.Size([768])
        # cls.predictions.transform.LayerNorm.bias torch.Size([768])
        # cls.predictions.decoder.weight torch.Size([30522, 768])
        # cls.predictions.decoder.bias torch.Size([30522])
        if key.endswith("cls.predictions.bias") or key.endswith("cls.predictions.decoder.bias"):
            # these are size(vocab) and remain unchanged
            return weight
        elif key.endswith("cls.predictions.transform.dense.bias"):
            return double_bias(weight, is_grad=is_grad, is_avg_sq=is_avg_sq)
        elif key.endswith("cls.predictions.transform.dense.weight"):
            return double_matrix_weight(weight, is_grad=is_grad, is_avg_sq=is_avg_sq)
        elif key.endswith("cls.predictions.decoder.weight"):
            return double_embedding(weight, is_grad=is_grad, is_avg_sq=is_avg_sq)
        elif "LayerNorm" in key:
            # layer norm is weight * (x - mean)/std + bias, so need to divide both weight and bias by 2.0
            new_weight = double_bias(weight, is_grad=False, is_avg_sq=False)
            if not is_grad:
                new_weight /= 2.0
            return new_weight

    elif (
        "word_embeddings" in key
        or "position_embeddings" in key
        or "token_type_embeddings" in key
        or "wte.weight" in key
        or "wpe.weight" in key
    ):
        if is_double_embedding:
            return double_embedding(weight, is_grad=is_grad, is_avg_sq=is_avg_sq)
        else:
            return weight.detach().clone()
    elif "masked_bias" in key or ("attn.bias" in key and len(weight.shape) != 1):
        return weight.detach().clone()
    elif "c_attn.weight" in key:
        return double_split_matrix_weight(weight, is_grad=is_grad, is_avg_sq=is_avg_sq)
    elif "c_attn.bias" in key:
        # TODO: this is hacked for GPT2
        return double_split_bias(weight, embed_dim=weight.shape[0] // 3, is_grad=is_grad, is_avg_sq=is_avg_sq)
    elif (
        "query.weight" in key
        or "key.weight" in key
        or "value.weight" in key
        or "dense.weight" in key
        or "c_proj.weight" in key
        or "c_fc.weight" in key
    ):
        return double_matrix_weight(weight, is_grad=is_grad, is_avg_sq=is_avg_sq)
    elif "ln_f" in key:
        new_weight = double_bias(weight, is_grad=False, is_avg_sq=False)
        if not is_grad:
            new_weight /= 2.0
        return new_weight
    elif "LayerNorm" in key or "bias" in key or "ln" in key:
        return double_bias(weight, is_grad=is_grad, is_avg_sq=is_avg_sq)
    elif "position_ids" in key:
        return weight

    # Not found
    print(key)
    import ipdb

    ipdb.set_trace()
    # raise ValueError(key, weight.shape)


def double_state_dict(old_state_dict, is_double_embedding):
    new_state_dict = {}
    for key, weight in old_state_dict.items():
        new_state_dict[key] = double_param(key, weight, is_double_embedding=is_double_embedding, is_grad=False, is_avg_sq=False)
    return new_state_dict


# depth growth operator
def deep_split_matrix_weight(x, is_identical, is_grad, is_avg_sq):
    if not is_identical:
        return x.detach().clone()

    embed_dim = x.shape[0]
    split_dim = x.shape[1] // x.shape[0]
    y_shape = [i for i in x.shape]
    y = x.new_zeros(*y_shape)

    for split_idx in range(split_dim):
        start_idx, end_idx = split_idx * x.shape[0], (1 + split_idx) * x.shape[0]
        y_split = deep_matrix_weight(x[:, start_idx:end_idx], is_identical, is_grad, is_avg_sq)
        y[:, start_idx:end_idx] = y_split.detach().clone()

    return y


def deep_matrix_weight(x, is_identical, is_grad, is_avg_sq):
    # x = (n, m), returns y = (2 * n, 2 * m), used for FF layers
    if is_identical:
        y = torch.zeros_like(x)
        if len(y.shape) > 1:
            y.fill_diagonal_(1)
        return y
    else:
        return x.detach().clone()


def deep_bias(x, is_identical, is_grad, is_avg_sq):
    # x = (n, ), returns y = (2 * n, ), used for bias weights
    if is_identical:
        return torch.zeros_like(x)
    else:
        return x.detach().clone()


def deep_param(key, weight, is_identical, is_grad, is_avg_sq):
    if "c_attn.weight" in key:
        return deep_split_matrix_weight(weight, is_identical=is_identical, is_grad=is_grad, is_avg_sq=is_avg_sq)
    elif "weight" in key:
        return deep_matrix_weight(weight, is_identical=is_identical, is_grad=is_grad, is_avg_sq=is_avg_sq)
    elif "bias" in key:
        return deep_bias(weight, is_identical=is_identical, is_grad=is_grad, is_avg_sq=is_avg_sq)


def deep_state_dict(old_state_dict, map_positions, is_identical):
    # how to insert layers: direct copy, identical copy
    # operator over the blocks: hacked for GPT-3
    new_state_dict = {}
    for key, weight in old_state_dict.items():
        if map_positions.get(key):
            for (new_key, new_key_copy_flag) in map_positions.get(key):
                # print( new_key_copy_flag, is_identical, new_key, key )
                new_state_dict[new_key] = deep_param(
                    key, weight, is_identical=new_key_copy_flag and is_identical, is_grad=False, is_avg_sq=False
                )
        else:
            new_state_dict[key] = weight.detach().clone()

    return new_state_dict


def test_double_matrix_weight():
    weight = torch.rand(52, 88)
    x = torch.rand(88, 1)
    reset_model_opt_copy = True
    weight2 = double_matrix_weight(weight, is_grad=False, is_avg_sq=False, reset_model_opt_copy=reset_model_opt_copy)
    y = torch.matmul(weight, x)
    y2 = torch.matmul(weight2, torch.cat([x, x], dim=0))
    print(torch.allclose(y, y2[:52], atol=1e-05, rtol=1e-03))
    assert torch.abs(y - y2[:52]).max() + torch.abs(y - y2[-52:]).max() < 1e-4

    x = torch.rand(1, 11)
    c_attn = torch.rand(11, 11 * 3)
    print(len(torch.matmul(x, c_attn).split(11, dim=1)))
    y0, y1, y2 = torch.matmul(x, c_attn).split(11, dim=1)

    c_attn2 = double_split_matrix_weight(c_attn, is_grad=False, is_avg_sq=False, reset_model_opt_copy=reset_model_opt_copy)

    y00, y11, y22 = torch.matmul(torch.cat([x, x], dim=1), c_attn2).split(11 * 2, dim=1)

    allcose = torch.allclose(y0, y00[:, :11], atol=1e-05, rtol=1e-03)
    print("reset_model_opt_copy", reset_model_opt_copy, allcose, y0.sum(), y00[:, :11].sum(), y00[:, 11:].sum())
    if not allcose:
        import ipdb

        ipdb.set_trace()
    # import ipdb; ipdb.set_trace()


def test_double_gradients():
    test_double_matrix_weight()
    # config = AutoConfig.from_pretrained('roberta-base')
    # config.hidden_size = 4
    # config.intermediate_size = 16
    # config.max_position_embeddings = 8
    # config.num_attention_heads = 1
    # config.num_hidden_layers = 1
    # config.vocab_size = 6

    # config.attention_probs_dropout_prob = 0
    # config.hidden_dropout_prob = 0
    # model = RobertaForMaskedLM(config=config)

    from gpt_pretrain import double_weights, double_param
    from transformers import AutoConfig, RobertaForMaskedLM, AutoModelForMaskedLM, AutoModelForCausalLM
    from transformers.optimization import AdamW, get_linear_schedule_with_warmup
    import torch

    # model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    optimizer = AdamW(model.parameters(), lr=0.00000, betas=(0.0, 0.0))
    model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[1, 2, 3, 4]])
    loss = model(input_ids=input_ids, labels=labels)[0]
    loss.backward()
    optimizer.step()
    # model.roberta.embeddings.word_embeddings.weight.grad
    reset_model_opt_copy = True
    reset_model_noise = False
    double_model = double_weights(
        model, is_double_embedding=True, reset_model_opt_copy=reset_model_opt_copy, reset_model_noise=reset_model_noise
    )
    double_optimizer = AdamW(double_model.parameters(), lr=0.00000, betas=(0.0, 0.0))
    double_model.eval()
    double_loss = double_model(input_ids=input_ids, labels=labels)[0]
    double_loss.backward()
    double_optimizer.step()

    print(double_loss.item(), loss.item(), torch.allclose(double_loss, loss, atol=1e-05, rtol=1e-03))
    assert torch.allclose(double_loss, loss, atol=1e-05, rtol=1e-03)
    # exit()
    for (name, parameter), (double_name, double_parameter), (opt_key, opt_val), (double_opt_key, double_opt_val) in zip(
        model.named_parameters(), double_model.named_parameters(), optimizer.state.items(), double_optimizer.state.items()
    ):
        assert name == double_name
        assert id(parameter) == id(opt_key)
        assert id(double_parameter) == id(double_opt_key)
        predicted = double_param(
            name,
            parameter.grad,
            is_double_embedding=True,
            is_grad=True,
            is_avg_sq=False,
            reset_optimizer_copy=True,
            reset_model_opt_copy=reset_model_opt_copy,
        )
        all_close = torch.allclose(predicted, double_parameter.grad, atol=1e-05, rtol=1e-03)

        if not all_close:
            print(
                "1", all_close, name, parameter.shape,
            )
            print(predicted)
            print(double_parameter.grad)

        predicted = double_param(
            name,
            opt_val["exp_avg"],
            is_double_embedding=True,
            is_grad=True,
            is_avg_sq=False,
            reset_optimizer_copy=True,
            reset_model_opt_copy=reset_model_opt_copy,
        )
        all_close = torch.allclose(predicted, double_opt_val["exp_avg"], atol=1e-05, rtol=1e-03)
        if not all_close:
            print(
                "2", all_close, name, parameter.shape,
            )
            print(predicted)
            print(double_opt_val["exp_avg"],)

        predicted = double_param(
            name,
            opt_val["exp_avg_sq"],
            is_double_embedding=True,
            is_grad=True,
            is_avg_sq=True,
            reset_optimizer_copy=True,
            reset_model_opt_copy=reset_model_opt_copy,
        )
        all_close = torch.allclose(predicted, double_opt_val["exp_avg_sq"], atol=1e-05, rtol=1e-03)
        if not all_close:
            print(
                "3", all_close, name, parameter.shape,
            )
            print(predicted)
            print(double_opt_val["exp_avg_sq"],)
            import ipdb

            ipdb.set_trace()
        else:
            print(
                "3", all_close, name, parameter.shape,
            )

    import ipdb

    ipdb.set_trace()


def double_weights(model, is_double_embedding):
    print(model)
    config = model.config

    # create an instance of the model twice the size
    new_config_dict = config.to_dict()
    new_config_dict["n_embd"] *= 2
    new_config_dict["n_inner"] = new_config_dict["n_inner"] * 2 if new_config_dict["n_inner"] is not None else None
    new_config_dict["n_head"] *= 2

    new_config = type(config).from_dict(new_config_dict)
    new_model = type(model)(new_config)

    # load the weights from the old model into new model after duplicating them
    model.tie_weights()
    new_model.tie_weights()

    new_state_dict = double_state_dict(model.state_dict(), is_double_embedding=is_double_embedding)
    new_model.load_state_dict(new_state_dict)
    new_model.tie_weights()

    return new_model


def double_depth(model):
    print(model)
    config = model.config

    # create an instance of the model twice the size
    new_config_dict = config.to_dict()
    print(new_config_dict)
    new_config_dict["num_hidden_layers"] *= 2

    new_config = type(config).from_dict(new_config_dict)
    new_model = type(model)(new_config)

    # load the weights from the old model into new model after duplicating them
    model.tie_weights()
    new_model.tie_weights()

    new_state_dict = deep_state_dict(model.state_dict())
    new_model.load_state_dict(new_state_dict)
    new_model.tie_weights()

    return new_model
