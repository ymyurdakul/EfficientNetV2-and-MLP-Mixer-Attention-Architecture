################################################################################
# EfficientNetV2-S + Linear-Attention MLP-Mixer Head
################################################################################

from keras_cv_attention_models.backend import functional  # already imported above
from keras_cv_attention_models.backend import is_channels_last
from keras_cv_attention_models.backend import layers, models
from keras_cv_attention_models.models import register_model


def linear_attention_1d(inputs, attn_dim=None, name=None, eps=1e-6):
    """
    Linear attention on sequence [B, N, C].
    Uses phi(x) = elu(x) + 1 to keep features positive.
    """
    channel_dim = inputs.shape[-1]
    attn_dim = attn_dim or channel_dim

    # Q, K, V projections
    q = layers.Dense(attn_dim, use_bias=False, name=name and name + "q")(inputs)
    k = layers.Dense(attn_dim, use_bias=False, name=name and name + "k")(inputs)
    v = layers.Dense(channel_dim, use_bias=False, name=name and name + "v")(inputs)

    def _linear_attn(qkv):
        q, k, v = qkv  # shapes: [B, N, Dq], [B, N, Dk], [B, N, C]

        phi_q = functional.nn.elu(q) + 1.0
        phi_k = functional.nn.elu(k) + 1.0

        # K^T V  --> [B, Dk, C]
        kv = functional.einsum("bnd,bnc->bdc", phi_k, v)

        # K^T 1  --> [B, Dk]
        k_sum = functional.einsum("bnd->bd", phi_k)

        # Numerator: phi(Q) (K^T V)   --> [B, N, C]
        numer = functional.einsum("bnd,bdc->bnc", phi_q, kv)

        # Denominator: phi(Q) (K^T 1) --> [B, N]
        denom = functional.einsum("bnd,bd->bn", phi_q, k_sum)
        denom = functional.expand_dims(denom + eps, axis=-1)  # [B, N, 1]

        return numer / denom

    out = layers.Lambda(_linear_attn, name=name and name + "linear_attn")([q, k, v])
    return out



def mlp_mixer_attention_block(
    inputs,
    tokens_mlp_dim,
    channels_mlp_dim,
    use_bias=True,
    drop_rate=0.0,
    activation="gelu",
    name=None,
):
    """
    MLP-Mixer style block with linear attention in both token and channel mixing.
    Input / output: [B, N, C]
    """
    # Token mixing
    nn = layer_norm(inputs, axis=-1, name=name and name + "LayerNorm_0")
    nn = linear_attention_1d(nn, name=name and name + "token_attn_")
    nn = mlp_block(nn, hidden_dim=tokens_mlp_dim, use_bias=use_bias, activation=activation, name=name and name + "token_mixing/")
    if drop_rate > 0:
        nn = layers.Dropout(drop_rate, noise_shape=(None, 1, 1), name=name and name + "token_drop")(nn)
    token_out = layers.Add(name=name and name + "add_0")([nn, inputs])

    # Channel mixing
    nn = layer_norm(token_out, axis=-1, name=name and name + "LayerNorm_1")
    nn = linear_attention_1d(nn, name=name and name + "channel_attn_")
    nn = mlp_block(nn, hidden_dim=channels_mlp_dim, use_bias=use_bias, activation=activation, name=name and name + "channel_mixing/")
    if drop_rate > 0:
        nn = layers.Dropout(drop_rate, noise_shape=(None, 1, 1), name=name and name + "channel_drop")(nn)
    out = layers.Add(name=name and name + "output")([nn, token_out])
    return out



def EfficientNetV2S_MLPMixerAttention(
    # EfficientNetV2-S config
    expands=[1, 4, 4, 4, 6, 6],
    out_channels=[24, 48, 64, 128, 160, 256],
    depthes=[2, 4, 4, 6, 9, 15],
    strides=[1, 2, 2, 2, 1, 2],
    se_ratios=[0, 0, 0, 0.25, 0.25, 0.25],
    is_fused="auto",
    use_shortcuts=True,
    first_conv_filter=24,
    output_conv_filter=1280,
    kernel_sizes=3,
    # Global model settings
    input_shape=(384, 384, 3),
    num_classes=3,
    dropout=0.2,
    first_strides=2,
    is_torch_mode=False,
    use_global_context_instead_of_se=False,
    drop_connect_rate=0.0,
    activation="swish",
    classifier_activation="softmax",
    include_preprocessing=False,
    pretrained=None,           # set to "imagenet" if you later wire in weight loading
    model_name="efficientnet_v2-s_mlp_mixer_attention",
    rescale_mode="tf",
    # Mixer head settings
    mixer_tokens_mlp_dim=128,
    mixer_channels_mlp_dim=512,
    mixer_blocks=4,
    mixer_use_bias=True,
    kwargs=None,
):
    if kwargs is None:
        kwargs = {}

    # Align input shape with current image_data_format
    input_shape = backend.align_input_shape_by_image_data_format(input_shape)
    inputs = layers.Input(shape=input_shape)
    bn_eps = TORCH_BATCH_NORM_EPSILON if is_torch_mode else TF_BATCH_NORM_EPSILON

    # Optional preprocessing
    if include_preprocessing and rescale_mode == "torch":
        channel_axis = 1 if backend.image_data_format() == "channels_first" else -1
        Normalization = layers.Normalization if hasattr(layers, "Normalization") else layers.experimental.preprocessing.Normalization
        mean = np.array([0.485, 0.456, 0.406], dtype="float32") * 255.0
        std = (np.array([0.229, 0.224, 0.225], dtype="float32") * 255.0) ** 2
        nn = Normalization(mean=mean, variance=std, axis=channel_axis)(inputs)
    elif include_preprocessing and rescale_mode == "tf":
        Rescaling = layers.Rescaling if hasattr(layers, "Rescaling") else layers.experimental.preprocessing.Rescaling
        nn = Rescaling(scale=1.0 / 128.0, offset=-1)(inputs)
    else:
        nn = inputs

    # Stem
    stem_width = make_divisible(first_conv_filter, 8)
    nn = conv2d_no_bias(nn, stem_width, 3, strides=first_strides, padding="same", use_torch_padding=is_torch_mode, name="stem_")
    nn = batchnorm_with_activation(nn, activation=activation, epsilon=bn_eps, name="stem_")

    blocks_kwargs = {
        "is_torch_mode": is_torch_mode,
        "use_global_context_instead_of_se": use_global_context_instead_of_se,
    }

    # EfficientNetV2-S main blocks
    pre_out = stem_width
    global_block_id = 0
    total_blocks = sum(depthes)
    se_ratios = se_ratios if isinstance(se_ratios, (list, tuple)) else ([se_ratios] * len(depthes))
    kernel_sizes = kernel_sizes if isinstance(kernel_sizes, (list, tuple)) else ([kernel_sizes] * len(depthes))

    for stack_id, (expand, out_channel, depth, stride, se_ratio, kernel_size) in enumerate(
        zip(expands, out_channels, depthes, strides, se_ratios, kernel_sizes)
    ):
        out = make_divisible(out_channel, 8)
        if is_fused == "auto":
            cur_is_fused = True if se_ratio == 0 else False
        else:
            cur_is_fused = is_fused[stack_id] if isinstance(is_fused, (list, tuple)) else is_fused
        cur_use_shortcuts = use_shortcuts[stack_id] if isinstance(use_shortcuts, (list, tuple)) else use_shortcuts

        for block_id in range(depth):
            name = "stack_{}_block{}_".format(stack_id, block_id)
            cur_stride = stride if block_id == 0 else 1
            shortcut = cur_use_shortcuts if out == pre_out and cur_stride == 1 else False
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            nn = inverted_residual_block(
                nn,
                out,
                cur_stride,
                expand,
                shortcut,
                kernel_size,
                block_drop_rate,
                se_ratio,
                cur_is_fused,
                **blocks_kwargs,
                activation=activation,
                name=name,
            )
            pre_out = out
            global_block_id += 1

    # Last 1×1 conv of EfficientNetV2-S backbone
    if output_conv_filter > 0:
        output_conv_filter = make_divisible(output_conv_filter, 8)
        nn = conv2d_no_bias(nn, output_conv_filter, 1, strides=1, padding="valid", use_torch_padding=is_torch_mode, name="post_")
        nn = batchnorm_with_activation(nn, activation=activation, epsilon=bn_eps, name="post_")


    # Project channels to mixer_channels_mlp_dim with 1×1 conv
    nn = conv2d_no_bias(nn, mixer_channels_mlp_dim, 1, strides=1, padding="same", use_torch_padding=is_torch_mode, name="mixer_proj_")
    nn = batchnorm_with_activation(nn, activation=activation, epsilon=bn_eps, name="mixer_proj_")

    # [B, H, W, C] -> [B, N, C]
    if is_channels_last():
        nn = layers.Reshape((-1, mixer_channels_mlp_dim), name="mixer_tokens")(nn)
    else:
        nn = layers.Permute((2, 3, 1), name="mixer_perm_to_last")(nn)
        nn = layers.Reshape((-1, mixer_channels_mlp_dim), name="mixer_tokens")(nn)

    # Mixer blocks with linear attention
    if mixer_blocks > 1:
        drop_connect_s, drop_connect_e = (0.0, drop_connect_rate)
    else:
        drop_connect_s = drop_connect_e = drop_connect_rate

    for ii in range(mixer_blocks):
        block_drop = drop_connect_s + (drop_connect_e - drop_connect_s) * float(ii) / max(1, mixer_blocks - 1)
        nn = mlp_mixer_attention_block(
            nn,
            tokens_mlp_dim=mixer_tokens_mlp_dim,
            channels_mlp_dim=mixer_channels_mlp_dim,
            use_bias=mixer_use_bias,
            drop_rate=block_drop,
            activation=activation,
            name=f"mixer_block_{ii}/",
        )

    # Final LN + GAP + classifier
    nn = layer_norm(nn, axis=-1, name="mixer_pre_head_layer_norm")
    nn = layers.GlobalAveragePooling1D(name="mixer_head_gap")(nn)
    if dropout > 0:
        nn = layers.Dropout(dropout, name="mixer_head_dropout")(nn)
    nn = layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="mixer_head")(nn)

    model = models.Model(inputs=inputs, outputs=nn, name=model_name)

    # Pre/post process (matches your other models)
    add_pre_post_process(model, rescale_mode="raw" if include_preprocessing else rescale_mode)

    # If you want, you can manually load EfficientNetV2-S backbone weights here
    # with your own weight-loading utility.

    return model



@register_model
def EfficientNetV2S_MLPMixerAttentionSmall(
    input_shape=(224, 224, 3),
    num_classes=3,
    dropout=0.2,
    classifier_activation="softmax",
    mixer_tokens_mlp_dim=128,
    mixer_channels_mlp_dim=512,
    mixer_blocks=4,
    **kwargs,
):
    """
    Small model:
    - EfficientNetV2-S backbone
    - Linear-attention MLP-Mixer head
    - Default num_classes=3
    """
    return EfficientNetV2S_MLPMixerAttention(
        input_shape=input_shape,
        num_classes=num_classes,
        dropout=dropout,
        classifier_activation=classifier_activation,
        mixer_tokens_mlp_dim=mixer_tokens_mlp_dim,
        mixer_channels_mlp_dim=mixer_channels_mlp_dim,
        mixer_blocks=mixer_blocks,
        **kwargs,
    )
