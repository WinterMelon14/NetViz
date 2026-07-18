import torch
import torch.nn as nn
import torch.nn.functional as F


class MetadataContractModel(nn.Module):
    """Small, deliberately broad model for testing NetViz metadata."""

    def __init__(self):
        super().__init__()

        # ------------------------------------------------------------------
        # Learned layers
        # ------------------------------------------------------------------

        self.linear = nn.Linear(
            in_features=4,
            out_features=3,
            bias=False,
        )

        self.embedding = nn.Embedding(
            num_embeddings=11,
            embedding_dim=4,
            padding_idx=0,
            max_norm=1.5,
            norm_type=1.0,
            scale_grad_by_freq=True,
            sparse=False,
        )

        # ------------------------------------------------------------------
        # Convolutions
        # ------------------------------------------------------------------

        self.conv1d = nn.Conv1d(
            in_channels=4,
            out_channels=4,
            kernel_size=3,
            stride=2,
            padding=2,
            dilation=2,
            groups=2,
            bias=False,
            padding_mode="circular",
        )

        # Exercises string padding metadata.
        self.conv2d = nn.Conv2d(
            in_channels=4,
            out_channels=4,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding="same",
            dilation=(2, 2),
            groups=2,
            bias=False,
            padding_mode="zeros",
        )

        self.conv3d = nn.Conv3d(
            in_channels=4,
            out_channels=4,
            kernel_size=(2, 3, 3),
            stride=(1, 2, 2),
            padding=(1, 1, 1),
            dilation=(1, 1, 1),
            groups=2,
            bias=False,
            padding_mode="replicate",
        )

        # ------------------------------------------------------------------
        # Max pooling
        # ------------------------------------------------------------------

        self.max_pool1d = nn.MaxPool1d(
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            return_indices=True,
            ceil_mode=True,
        )

        self.max_pool2d = nn.MaxPool2d(
            kernel_size=(3, 2),
            stride=(2, 1),
            padding=(1, 0),
            dilation=(1, 2),
            return_indices=True,
            ceil_mode=True,
        )

        self.max_pool3d = nn.MaxPool3d(
            kernel_size=(2, 2, 2),
            stride=(2, 2, 1),
            padding=(0, 0, 0),
            dilation=(1, 1, 1),
            return_indices=True,
            ceil_mode=True,
        )

        # ------------------------------------------------------------------
        # Average pooling
        # ------------------------------------------------------------------

        self.avg_pool1d = nn.AvgPool1d(
            kernel_size=3,
            stride=2,
            padding=1,
            ceil_mode=True,
            count_include_pad=False,
        )

        self.avg_pool2d = nn.AvgPool2d(
            kernel_size=(3, 2),
            stride=(2, 1),
            padding=(1, 0),
            ceil_mode=True,
            count_include_pad=False,
            divisor_override=2,
        )

        self.avg_pool3d = nn.AvgPool3d(
            kernel_size=(2, 2, 2),
            stride=(2, 1, 2),
            padding=(0, 0, 0),
            ceil_mode=True,
            count_include_pad=False,
            divisor_override=4,
        )

        # ------------------------------------------------------------------
        # Adaptive average pooling
        # ------------------------------------------------------------------

        self.adaptive_avg_pool1d = nn.AdaptiveAvgPool1d(output_size=3)
        self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d(output_size=(3, 5))
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(
            output_size=(2, 3, 4)
        )

        # ------------------------------------------------------------------
        # Normalization
        # ------------------------------------------------------------------

        self.batch_norm1d = nn.BatchNorm1d(
            num_features=4,
            eps=1e-4,
            momentum=0.2,
            affine=False,
            track_running_stats=False,
        )

        self.batch_norm2d = nn.BatchNorm2d(
            num_features=4,
            eps=2e-4,
            momentum=0.3,
            affine=True,
            track_running_stats=True,
        )

        self.batch_norm3d = nn.BatchNorm3d(
            num_features=4,
            eps=3e-4,
            momentum=None,
            affine=True,
            track_running_stats=True,
        )

        # Tests the separate LayerNorm bias metadata.
        self.layer_norm = nn.LayerNorm(
            normalized_shape=(4,),
            eps=2e-5,
            elementwise_affine=True,
            bias=False,
        )

        self.rms_norm = nn.RMSNorm(
            normalized_shape=(4,),
            eps=1e-6,
            elementwise_affine=True,
        )

        self.group_norm = nn.GroupNorm(
            num_groups=2,
            num_channels=4,
            eps=3e-5,
            affine=False,
        )

        self.instance_norm1d = nn.InstanceNorm1d(
            num_features=4,
            eps=4e-5,
            momentum=0.15,
            affine=True,
            track_running_stats=True,
        )

        self.instance_norm2d = nn.InstanceNorm2d(
            num_features=4,
            eps=5e-5,
            momentum=0.25,
            affine=False,
            track_running_stats=False,
        )

        self.instance_norm3d = nn.InstanceNorm3d(
            num_features=4,
            eps=6e-5,
            momentum=0.35,
            affine=True,
            track_running_stats=True,
        )

        # ------------------------------------------------------------------
        # Activations
        # ------------------------------------------------------------------

        self.relu = nn.ReLU(inplace=False)

        self.leaky_relu = nn.LeakyReLU(
            negative_slope=0.2,
            inplace=False,
        )

        self.elu = nn.ELU(
            alpha=1.25,
            inplace=False,
        )

        self.selu = nn.SELU(inplace=False)
        self.sigmoid = nn.Sigmoid()
        self.hardsigmoid = nn.Hardsigmoid(inplace=False)
        self.tanh = nn.Tanh()
        self.silu = nn.SiLU(inplace=False)
        self.mish = nn.Mish(inplace=False)
        self.hardswish = nn.Hardswish(inplace=False)

        self.softmax = nn.Softmax(dim=-1)
        self.log_softmax = nn.LogSoftmax(dim=1)

        self.gelu = nn.GELU(approximate="tanh")

        self.softplus = nn.Softplus(
            beta=2.0,
            threshold=10.0,
        )

        # ------------------------------------------------------------------
        # Regularization
        # ------------------------------------------------------------------

        self.dropout = nn.Dropout(
            p=0.10,
            inplace=False,
        )

        self.dropout1d = nn.Dropout1d(
            p=0.20,
            inplace=False,
        )

        self.dropout2d = nn.Dropout2d(
            p=0.30,
            inplace=False,
        )

        self.dropout3d = nn.Dropout3d(
            p=0.40,
            inplace=False,
        )

        self.alpha_dropout = nn.AlphaDropout(
            p=0.15,
            inplace=False,
        )

        # Tests non-default Flatten dimensions.
        self.flatten_module = nn.Flatten(
            start_dim=2,
            end_dim=3,
        )

        # ------------------------------------------------------------------
        # Attention and Transformer blocks
        # ------------------------------------------------------------------

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=8,
            num_heads=2,
            dropout=0.20,
            bias=False,
            add_zero_attn=True,
            batch_first=True,
        )

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=8,
            nhead=2,
            dim_feedforward=16,
            dropout=0.30,
            batch_first=True,
            norm_first=True,
            bias=False,
        )

        self.transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=8,
            nhead=2,
            dim_feedforward=20,
            dropout=0.40,
            batch_first=True,
            norm_first=True,
            bias=False,
        )

        encoder_stack_layer = nn.TransformerEncoderLayer(
            d_model=8,
            nhead=2,
            dim_feedforward=12,
            dropout=0.10,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_stack_layer,
            num_layers=2,
        )

        decoder_stack_layer = nn.TransformerDecoderLayer(
            d_model=8,
            nhead=2,
            dim_feedforward=12,
            dropout=0.10,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_stack_layer,
            num_layers=2,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_ids: torch.Tensor,
    ):
        """
        Expected inputs:

        x:
            float tensor with shape [2, 4, 8, 8]

        token_ids:
            int64 tensor with shape [2, 4], values in [0, 11)
        """

        # ------------------------------------------------------------------
        # Construct small 1D, 2D, 3D and vector inputs
        # ------------------------------------------------------------------

        vector = x.mean(dim=(2, 3))                 # [2, 4]
        x1 = x.mean(dim=2)                         # [2, 4, 8]
        x2 = x                                      # [2, 4, 8, 8]
        x3 = x.unsqueeze(2).repeat(1, 1, 4, 1, 1)  # [2, 4, 4, 8, 8]

        # ------------------------------------------------------------------
        # Learned layers
        # ------------------------------------------------------------------

        linear = self.linear(vector)
        embedding = self.embedding(token_ids)

        # ------------------------------------------------------------------
        # Convolution branches
        # ------------------------------------------------------------------

        conv1d = self.conv1d(x1)
        conv2d = self.conv2d(x2)
        conv3d = self.conv3d(x3)

        # ------------------------------------------------------------------
        # Pooling branches
        # ------------------------------------------------------------------

        max_pool1d, max_indices1d = self.max_pool1d(conv1d)
        max_pool2d, max_indices2d = self.max_pool2d(conv2d)
        max_pool3d, max_indices3d = self.max_pool3d(conv3d)

        avg_pool1d = self.avg_pool1d(conv1d)
        avg_pool2d = self.avg_pool2d(conv2d)
        avg_pool3d = self.avg_pool3d(conv3d)

        adaptive1d = self.adaptive_avg_pool1d(conv1d)
        adaptive2d = self.adaptive_avg_pool2d(conv2d)
        adaptive3d = self.adaptive_avg_pool3d(conv3d)

        # ------------------------------------------------------------------
        # Normalization
        # ------------------------------------------------------------------

        batch_norm1d = self.batch_norm1d(conv1d)
        batch_norm2d = self.batch_norm2d(conv2d)
        batch_norm3d = self.batch_norm3d(conv3d)

        layer_norm = self.layer_norm(embedding)
        rms_norm = self.rms_norm(embedding)

        group_norm = self.group_norm(x2)

        instance_norm1d = self.instance_norm1d(conv1d)
        instance_norm2d = self.instance_norm2d(conv2d)
        instance_norm3d = self.instance_norm3d(conv3d)

        # ------------------------------------------------------------------
        # Activation modules
        # ------------------------------------------------------------------

        relu = self.relu(linear)
        leaky_relu = self.leaky_relu(linear)
        elu = self.elu(linear)
        selu = self.selu(linear)
        sigmoid = self.sigmoid(linear)
        hardsigmoid = self.hardsigmoid(linear)
        tanh = self.tanh(linear)
        silu = self.silu(linear)
        mish = self.mish(linear)
        hardswish = self.hardswish(linear)
        softmax = self.softmax(linear)
        log_softmax = self.log_softmax(linear)
        gelu = self.gelu(linear)
        softplus = self.softplus(linear)

        # Functional ReLU exercises the lowercase `relu` matcher.
        functional_relu = F.relu(linear, inplace=False)

        # ------------------------------------------------------------------
        # Regularization modules
        # ------------------------------------------------------------------

        dropout = self.dropout(linear)
        dropout1d = self.dropout1d(conv1d)
        dropout2d = self.dropout2d(conv2d)
        dropout3d = self.dropout3d(conv3d)
        alpha_dropout = self.alpha_dropout(linear)

        # ------------------------------------------------------------------
        # Flatten, reshape and layout operations
        # ------------------------------------------------------------------

        module_flatten = self.flatten_module(x)

        # Positional functional arguments intentionally test extraction.
        functional_flatten = torch.flatten(x, 2, 3)

        # Keyword method arguments test call_method kwargs.
        method_flatten = x.flatten(start_dim=1, end_dim=2)

        reshaped = x.reshape(2, 4, -1)
        viewed = x.view(2, 4, -1)

        permuted = x.permute(0, 2, 3, 1)
        transposed = x.transpose(-2, -1)
        unsqueezed = vector.unsqueeze(-1)

        expanded = vector.unsqueeze(1).expand(
            -1,
            3,
            -1,
        )

        contiguous = permuted.contiguous(
            memory_format=torch.contiguous_format
        )

        # ------------------------------------------------------------------
        # Attention and Transformer blocks
        # ------------------------------------------------------------------

        attention_input = x.mean(dim=2)             # [2, 4, 8]
        attention_memory = x.mean(dim=3)            # [2, 4, 8]

        multihead_attention, multihead_weights = self.multihead_attention(
            attention_input,
            attention_input,
            attention_input,
            need_weights=False,
            average_attn_weights=False,
            is_causal=False,
        )

        attention_qkv = attention_input.view(2, 2, 4, 4)
        scaled_dot_product_attention = F.scaled_dot_product_attention(
            attention_qkv,
            attention_qkv,
            attention_qkv,
            dropout_p=0.0,
            is_causal=False,
        )

        transformer_encoder_layer = self.transformer_encoder_layer(
            attention_input
        )
        transformer_decoder_layer = self.transformer_decoder_layer(
            attention_input,
            attention_memory,
        )
        transformer_encoder = self.transformer_encoder(attention_input)
        transformer_decoder = self.transformer_decoder(
            attention_input,
            attention_memory,
        )

        # ------------------------------------------------------------------
        # Splitting and recombination operations
        # ------------------------------------------------------------------

        # Size 4 split into 3 requested chunks normally returns fewer than 3.
        chunked = vector.chunk(3, dim=-1)

        split = vector.split(
            split_size=[1, 2, 1],
            dim=-1,
        )

        # Positional dim intentionally tests positional argument metadata.
        stacked = torch.stack(
            (vector, vector + 1.0),
            -1,
        )

        unbound = vector.unbind(dim=-1)

        repeated = vector.repeat(1, 2)

        narrowed = vector.narrow(
            -1,
            1,
            2,
        )

        rolled = torch.roll(
            vector,
            shifts=(1, -1),
            dims=(0, 1),
        )

        flipped = torch.flip(
            vector,
            dims=(-1,),
        )

        # Positional dim.
        concatenated = torch.cat(
            (vector, vector),
            -1,
        )

        # Alias plus keyword dim.
        concatenated_alias = torch.concat(
            (vector, vector),
            dim=0,
        )

        # ------------------------------------------------------------------
        # Element-wise arithmetic
        # ------------------------------------------------------------------
        relu_padding = F.pad(
            functional_relu,
            pad=(0, 1),
            mode="constant",
            value=0.0,
        )
        added = torch.add(
            vector,
            relu_padding,
            alpha=0.5,
        )

        multiplied = vector * 1.5
        subtracted = vector - 0.25
        divided = vector / 2.0
        powered = torch.pow(vector.abs() + 1.0, 2.0)
        square_rooted = torch.sqrt(vector.abs() + 1.0)
        exponentiated = torch.exp(vector)
        logged = torch.log(vector.abs() + 1.0)
        absolute = torch.abs(vector)
        negated = torch.neg(vector)
        clamped = torch.clamp(vector, min=-0.5, max=0.5)

        # ------------------------------------------------------------------
        # Matrix multiplication
        # ------------------------------------------------------------------

        mat_left = x[:, :3, :4, 0]                 # [2, 3, 4]
        mat_right = x[:, :4, :5, 1]                # [2, 4, 5]
        matrix_multiplied = torch.matmul(mat_left, mat_right)
        mm_multiplied = torch.mm(mat_left[0], mat_right[0])
        bmm_multiplied = torch.bmm(mat_left, mat_right)
        einsum_multiplied = torch.einsum(
            "bij,bjk->bik",
            mat_left,
            mat_right,
        )

        # ------------------------------------------------------------------
        # Padding and reductions
        # ------------------------------------------------------------------

        padded = F.pad(
            vector,
            pad=(1, 2),
            mode="constant",
            value=0.25,
        )

        reduced_mean = x.mean(
            dim=(-2, -1),
            keepdim=True,
        )

        reduced_sum = x.sum(
            dim=1,
            keepdim=False,
        )
        reduced_max = torch.max(
            x,
            dim=-1,
            keepdim=False,
        ).values
        reduced_min = x.min(
            dim=1,
            keepdim=True,
        ).values
        reduced_norm = torch.norm(
            x,
            p=2,
            dim=-1,
            keepdim=True,
        )
        reduced_std = torch.std(
            x,
            dim=-1,
            keepdim=False,
            unbiased=False,
        )
        reduced_var = torch.var(
            x,
            dim=-1,
            keepdim=True,
            unbiased=False,
        )

        # ------------------------------------------------------------------
        # Conditional and masking operations
        # ------------------------------------------------------------------

        conditioned = torch.where(vector > 0, vector, -vector)
        mask_filled = vector.masked_fill(
            vector > 0,
            0.5,
        )
        mask_selected = torch.masked_select(
            vector,
            vector > 0,
        )

        # ------------------------------------------------------------------
        # Resizing
        # ------------------------------------------------------------------

        interpolated = F.interpolate(
            x,
            size=(7, 9),
            mode="bilinear",
            align_corners=False,
        )

        # Returning every result prevents future optimization passes from
        # treating contract-test nodes as unused.
        return {
            "linear": linear,
            "embedding": embedding,

            "conv1d": conv1d,
            "conv2d": conv2d,
            "conv3d": conv3d,

            "max_pool1d": max_pool1d,
            "max_indices1d": max_indices1d,
            "max_pool2d": max_pool2d,
            "max_indices2d": max_indices2d,
            "max_pool3d": max_pool3d,
            "max_indices3d": max_indices3d,

            "avg_pool1d": avg_pool1d,
            "avg_pool2d": avg_pool2d,
            "avg_pool3d": avg_pool3d,

            "adaptive1d": adaptive1d,
            "adaptive2d": adaptive2d,
            "adaptive3d": adaptive3d,

            "batch_norm1d": batch_norm1d,
            "batch_norm2d": batch_norm2d,
            "batch_norm3d": batch_norm3d,
            "layer_norm": layer_norm,
            "rms_norm": rms_norm,
            "group_norm": group_norm,
            "instance_norm1d": instance_norm1d,
            "instance_norm2d": instance_norm2d,
            "instance_norm3d": instance_norm3d,

            "relu": relu,
            "leaky_relu": leaky_relu,
            "elu": elu,
            "selu": selu,
            "sigmoid": sigmoid,
            "hardsigmoid": hardsigmoid,
            "tanh": tanh,
            "silu": silu,
            "mish": mish,
            "hardswish": hardswish,
            "softmax": softmax,
            "log_softmax": log_softmax,
            "gelu": gelu,
            "softplus": softplus,
            "functional_relu": functional_relu,

            "dropout": dropout,
            "dropout1d": dropout1d,
            "dropout2d": dropout2d,
            "dropout3d": dropout3d,
            "alpha_dropout": alpha_dropout,

            "module_flatten": module_flatten,
            "functional_flatten": functional_flatten,
            "method_flatten": method_flatten,
            "reshaped": reshaped,
            "viewed": viewed,
            "permuted": permuted,
            "transposed": transposed,
            "unsqueezed": unsqueezed,
            "expanded": expanded,
            "contiguous": contiguous,

            "multihead_attention": multihead_attention,
            "multihead_weights": multihead_weights,
            "scaled_dot_product_attention": scaled_dot_product_attention,
            "transformer_encoder_layer": transformer_encoder_layer,
            "transformer_decoder_layer": transformer_decoder_layer,
            "transformer_encoder": transformer_encoder,
            "transformer_decoder": transformer_decoder,

            "chunked": chunked,
            "split": split,
            "stacked": stacked,
            "unbound": unbound,
            "repeated": repeated,
            "narrowed": narrowed,
            "rolled": rolled,
            "flipped": flipped,
            "concatenated": concatenated,
            "concatenated_alias": concatenated_alias,

            "added": added,
            "multiplied": multiplied,
            "subtracted": subtracted,
            "divided": divided,
            "powered": powered,
            "square_rooted": square_rooted,
            "exponentiated": exponentiated,
            "logged": logged,
            "absolute": absolute,
            "negated": negated,
            "clamped": clamped,
            "matrix_multiplied": matrix_multiplied,
            "mm_multiplied": mm_multiplied,
            "bmm_multiplied": bmm_multiplied,
            "einsum_multiplied": einsum_multiplied,
            "padded": padded,
            "reduced_mean": reduced_mean,
            "reduced_sum": reduced_sum,
            "reduced_max": reduced_max,
            "reduced_min": reduced_min,
            "reduced_norm": reduced_norm,
            "reduced_std": reduced_std,
            "reduced_var": reduced_var,
            "conditioned": conditioned,
            "mask_filled": mask_filled,
            "mask_selected": mask_selected,
            "interpolated": interpolated,
        }


def netviz_example_inputs():
    """Representative inputs recognized by NetViz."""
    x = torch.randn(2, 4, 8, 8)

    token_ids = torch.tensor(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 10],
        ],
        dtype=torch.int64,
    )

    return x, token_ids
