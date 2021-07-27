"""Dynamic Depth Separable Convolutional Transformer"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modelzoo1d.dain import DAIN_Layer


class DynamicDepthSeparableConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes=[3, 15],
        dilation=1,
        bias=False,
        intermediate_nonlinearity=False,
        pad=True
    ):
        super(DynamicDepthSeparableConv1d, self).__init__()
        self.pointwise = nn.Conv1d(
            in_channels,
            out_channels,
            1,
            bias=bias)

        self.intermediate_nonlinearity = intermediate_nonlinearity

        if self.intermediate_nonlinearity:
            self.nonlinear_activation = nn.ReLU()

        # Create dynamic kernel gating mechanism
        self.kernel_sizes = kernel_sizes
        self.num_kernels = len(kernel_sizes)

        self.dynamic_gate = nn.Parameter(
          torch.Tensor([1.0 / self.num_kernels for _ in self.kernel_sizes]))

        self.dynamic_depthwise = nn.ModuleList()
        for kernel_size in self.kernel_sizes:
            padding = ((kernel_size - 1) // 2 * dilation) if pad else 0
            conv = nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias)
            self.dynamic_depthwise.append(conv)

    def get_gate(self):
        return self.dynamic_gate

    def forward(self, x):
        out = self.pointwise(x)

        if self.intermediate_nonlinearity:
            out = self.nonlinear_activation(out)

        # Apply dynamically sized kernels to values
        dynamic_out = []
        for dynamic_conv in self.dynamic_depthwise:
            dynamic_out.append(dynamic_conv(out))

        # Fallback to normal depth separable convolution if single kernel size
        if self.num_kernels > 1:
            out = torch.sum(
                torch.stack(
                    dynamic_out,
                    dim=-1
                ) * F.softmax(self.dynamic_gate, dim=-1),
                dim=-1)
        else:
            out = dynamic_out[0]

        return out


class DynamicDepthSeparableConv1dMultiheadAttention(nn.Module):
    def __init__(
        self,
        c,
        heads=8,
        kernel_sizes=[3, 15],
        share_encoder=False,
        save_attn=False,
        has_cls_token=False
    ):
        super(DynamicDepthSeparableConv1dMultiheadAttention, self).__init__()
        self.heads = heads
        self.kernel_sizes = kernel_sizes
        self.save_attn = save_attn
        self.has_cls_token = has_cls_token

        # These compute the queries, keys, and values for all
        # heads (as a single concatenated vector)
        self.to_queries = DynamicDepthSeparableConv1d(
            c,
            c * heads,
            kernel_sizes=kernel_sizes)

        if share_encoder:
            self.to_keys = self.to_queries
        else:
            self.to_keys = DynamicDepthSeparableConv1d(
                c,
                c * heads,
                kernel_sizes=kernel_sizes)

        self.to_values = DynamicDepthSeparableConv1d(
            c,
            c * heads,
            kernel_sizes=kernel_sizes)

        if self.has_cls_token:
            self.to_q_cls_embed = nn.Conv1d(c, c * heads, 1)
            self.to_k_cls_embed = nn.Conv1d(c, c * heads, 1)
            self.to_v_cls_embed = nn.Conv1d(c, c * heads, 1)

        # This unifies the outputs of the different heads into a single
        # c-vector
        if self.heads > 1:
            self.unify_heads = nn.Conv1d(heads * c, c, 1)
        else:
            self.unify_heads = nn.Identity()

        self.attn = None
        self.norm_attn = None

    def get_attn(self, norm=False):
        if norm:
            return self.norm_attn
        return self.attn

    def forward(self, q, k=None, v=None):
        b, c, q_l = q.size()
        h = self.heads

        if k is None:
            k = q
            k_l = q_l
        else:
            _, _, k_l = k.size()

        if v is None:
            v = q
            v_l = q_l
        else:
            _, _, v_l = k.size()

        assert k_l == v_l, 'key and value length must be equal'

        if self.has_cls_token:
            q_cls_token = self.to_q_cls_embed(q[:, :, 0].unsqueeze(-1))
            q = q[:, :, 1:]
            q_l -= 1
            k_cls_token = self.to_k_cls_embed(k[:, :, 0].unsqueeze(-1))
            k = k[:, :, 1:]
            k_l -= 1
            v_cls_token = self.to_v_cls_embed(v[:, :, 0].unsqueeze(-1))
            v = v[:, :, 1:]
            v_l -= 1

        queries = self.to_queries(q).view(b, h, c, q_l)
        keys = self.to_keys(k).view(b, h, c, k_l)
        values = self.to_values(v).view(b, h, c, v_l)

        # Fold heads into the batch dimension
        queries = queries.view(b * h, c, q_l)
        keys = keys.view(b * h, c, k_l)
        values = values.view(b * h, c, v_l)

        if self.has_cls_token:
            q_cls_token = q_cls_token.view(b * h, c, 1)
            k_cls_token = k_cls_token.view(b * h, c, 1)
            v_cls_token = v_cls_token.view(b * h, c, 1)
            queries = torch.cat([q_cls_token, queries], dim=-1)
            keys = torch.cat([k_cls_token, keys], dim=-1)
            values = torch.cat([v_cls_token, values], dim=-1)
            q_l += 1
            k_l += 1
            v_l += 1

        # Scale and get dot product of queries and keys
        queries = queries / (c ** (1 / 4))
        keys = keys / (c ** (1 / 4))

        dot = torch.bmm(keys.transpose(1, 2), queries)
        # dot now has size (b*h, k_l, q_l) containing raw weights

        if self.save_attn:
            self.attn = dot.view(b, h, k_l, q_l)

        dot = F.softmax(dot, dim=1)
        # dot now has channel-wise self-attention probabilities

        if self.save_attn:
            self.norm_attn = dot.view(b, h, k_l, q_l)

        # Apply the self attention to the values
        out = torch.bmm(values, dot).view(b, h * c, q_l)

        # Unify heads
        out = self.unify_heads(out)

        return out

# Implement cls_token functionalities for template attention
class DynamicDepthSeparableConv1dTemplateAttention(nn.Module):
    def __init__(
        self,
        qk_c,
        v_c,
        heads=8,
        kernel_sizes=[3, 15],
        share_encoder=False,
        save_attn=False
    ):
        super(
            DynamicDepthSeparableConv1dTemplateAttention, self).__init__()
        self.heads = heads
        self.kernel_sizes = kernel_sizes
        self.save_attn = save_attn

        # These compute the queries, keys, and values for all
        # heads (as a single concatenated vector)
        self.to_queries = DynamicDepthSeparableConv1d(
            qk_c,
            qk_c * heads,
            kernel_sizes=kernel_sizes)

        if share_encoder:
            self.to_keys = self.to_queries
        else:
            self.to_keys = DynamicDepthSeparableConv1d(
                qk_c,
                qk_c * heads,
                kernel_sizes=kernel_sizes)

        self.to_values = DynamicDepthSeparableConv1d(
            v_c,
            v_c * heads,
            kernel_sizes=kernel_sizes)

        # This unifies the outputs of the different heads into a single
        # v_c-vector
        if self.heads > 1:
            self.unify_heads = nn.Conv1d(heads * v_c, v_c, 1)
        else:
            self.unify_heads = nn.Identity()

        self.attn = None
        self.norm_attn = None

    def get_attn(self, norm=False):
        if norm:
            return self.norm_attn
        return self.attn

    def forward(self, queries, keys, values):
        if len(values.size()) == 2:
            values = values.unsqueeze(1)

        q_b, qk_c, length = queries.size()
        kv_b, v_c, _ = values.size()
        h = self.heads

        queries = self.to_queries(queries).view(q_b, h, qk_c, length)
        keys = self.to_keys(keys).view(kv_b, h, qk_c, length)
        values = self.to_values(values).view(kv_b, h, v_c, length)

        # Fold heads into the batch dimension
        queries = queries.view(q_b * h, qk_c, length)
        keys = keys.view(kv_b * h, qk_c, length)
        values = values.view(kv_b * h, v_c, length)

        # Scale and get dot product of queries and key
        queries = queries / (qk_c ** (1 / 4))
        keys = keys / (v_c ** (1 / 4))

        if kv_b > 1:
            dot = torch.matmul(
                keys.transpose(1, 2).contiguous().view(kv_b * h, length, qk_c),
                queries
            )
            # dot now has size (q_b*h, kv_b*h, length, length) containing raw
            # weights

            if self.save_attn:
                self.attn = dot

            dot = F.softmax(dot, dim=2)
            # dot now has channel-wise self-attention probabilities

            if self.save_attn:
                self.norm_attn = dot

            # Apply the attention to the values
            out = torch.matmul(values, dot)
            # out now has size (q_b*h, kv_b*h, v_c, length)

            out = torch.mean(out, dim=1)
            # out now has size (q_b*h, v_c, length)
        else:
            dot = torch.matmul(keys.transpose(1, 2), queries)
            # dot now has size (q_b*h, length, length) containing raw weights

            if self.save_attn:
                self.attn = dot

            dot = F.softmax(dot, dim=1)
            # dot now has channel-wise self-attention probabilities

            if self.save_attn:
                self.norm_attn = dot

            # Apply the attention to the values
            out = torch.matmul(values, dot)
            # out now has size (q_b*h, v_c, length)

        out = out.view(q_b, h * v_c, length)

        # Unify heads
        out = self.unify_heads(out)

        return out


class Conv1dFeedForwardNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        activation='relu'
    ):
        super(Conv1dFeedForwardNetwork, self).__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Conv1d(
                n,
                k,
                1
            ) for n, k in zip([input_dim] + h, h + [output_dim]))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise NotImplementedError('invalid activation')

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.activation(
                layer(x)) if i < self.num_layers - 1 else layer(x)

        return x


class DynamicDepthSeparableConv1dTransformerBlock(nn.Module):
    def __init__(
        self,
        c,
        heads=8,
        kernel_sizes=[3, 15],
        share_encoder=False,
        save_attn=False,
        norm_type='layer',
        pre_norm=True,
        depth_multiplier=4,
        dropout=0.1
    ):
        super(DynamicDepthSeparableConv1dTransformerBlock, self).__init__()
        self.norm_type = norm_type
        self.pre_norm = pre_norm

        self.attention = DynamicDepthSeparableConv1dMultiheadAttention(
            c,
            heads=heads,
            kernel_sizes=kernel_sizes,
            share_encoder=share_encoder,
            save_attn=save_attn)

        # Allow instance norm instead of layer norm
        if self.norm_type == 'layer':
            self.norm1 = nn.LayerNorm(c)
            self.norm2 = nn.LayerNorm(c)
        elif self.norm_type == 'instance':
            self.norm1 = nn.InstanceNorm1d(c)
            self.norm2 = nn.InstanceNorm1d(c)
        elif self.norm_type == 'affine_instance':
            self.norm1 = nn.InstanceNorm1d(c, affine=True)
            self.norm2 = nn.InstanceNorm1d(c, affine=True)
        else:
            raise NotImplementedError('norm type must be layer or instance')

        # 1D Convolutions instead of FC
        self.feed_forward = Conv1dFeedForwardNetwork(
            c,
            depth_multiplier * c,
            c,
            num_layers=2,
            activation='relu')

        # 2D Dropout to simulate spatial dropout of entire channels
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, q, k=None, v=None):
        out = q

        if self.pre_norm:
            if self.norm_type == 'layer':
                q = self.norm1(q.transpose(1, 2)).transpose(1, 2)
            elif 'instance' in self.norm_type:
                q = self.norm1(q)

        attended = self.attention(q, k, v)
        out = self.dropout(attended) + out

        if not self.pre_norm:
            if self.norm_type == 'layer':
                out = self.norm1(out.transpose(1, 2)).transpose(1, 2)
            elif 'instance' in self.norm_type:
                out = self.norm1(out)

            fed_forward = self.feed_forward(out)
        else:
            if self.norm_type == 'layer':
                fed_forward = self.feed_forward(
                    self.norm2(out.transpose(1, 2)).transpose(1, 2))
            elif 'instance' in self.norm_type:
                fed_forward = self.feed_forward(self.norm2(out))

        out = self.dropout(fed_forward) + out
        
        if not self.pre_norm:
            if self.norm_type == 'layer':
                out = self.norm2(out.transpose(1, 2)).transpose(1, 2)
            elif 'instance' in self.norm_type:
                out = self.norm2(out)

        return out


class TimeSeriesQueryAttentionPooling(nn.Module):
    def __init__(
        self,
        c_in,
        c_q=None,
        heads=1,
        num_queries=1,
        save_attn=False
    ):
        super(TimeSeriesQueryAttentionPooling, self).__init__()
        self.heads = heads
        self.num_queries = num_queries
        self.save_attn = save_attn

        self.c_q = c_q if c_q else c_in

        # This represents the queries for the weak binary global label(s)
        self.query_embeds = nn.Parameter(
            torch.randn(self.heads, self.c_q, self.num_queries))

        if c_q:
            self.to_keys = Conv1dFeedForwardNetwork(
                c_in,
                self.c_q * self.heads,
                self.c_q * self.heads,
                num_layers=3,
                activation='relu')
        else:
            self.to_keys = None

        self.attn = None
        self.norm_attn = None

    def get_attn(self, norm=False):
        if norm:
            return self.norm_attn
        return self.attn

    def get_trainable_attn(self, norm=False):
        if norm:
            return torch.mean(self.norm_attn.transpose(2, 3), dim=1)
        return torch.mean(self.attn.transpose(2, 3), dim=1)

    def forward(self, x):
        b, c, length = x.size()
        h = self.heads
        queries = (self.query_embeds
                   .unsqueeze(0)
                   .repeat(b, 1, 1, 1)
                   .view(b * h, self.c_q, self.num_queries))

        # Repeat and fold heads into the batch dimension
        values = x.repeat(1, h, 1).view(b * h, c, length)

        if self.to_keys is not None:
            keys = self.to_keys(x).view(b * h, self.c_q, length)
        else:
            keys = values

        # Scale and get dot product of queries and keys
        queries = queries / (self.c_q ** (1 / 4))
        keys = keys / (self.c_q ** (1 / 4))
        dot = torch.bmm(keys.transpose(1, 2), queries)
        # dot now has size (b*h, length, num_queries) containing raw weights

        if self.save_attn:
            self.attn = dot.view(b, h, length, self.num_queries)

        dot = F.softmax(dot, dim=1)
        # dot now has channel-wise self-attention probabilities

        if self.save_attn:
            self.norm_attn = dot.view(b, h, length, self.num_queries)

        # Apply the self attention to the values
        out = torch.bmm(values, dot).view(b, h, c, self.num_queries)

        # Unify heads
        out = torch.mean(out, dim=1)
        # out now has size (b, c, num_queries)

        return out


class DDSCTransformer(nn.Module):
    def __init__(
        self,
        in_channels=6,
        normalize=False,
        normalization_mode='full',
        save_normalized=False,
        transformer_channels=32,
        depth=6,
        heads=8,
        kernel_sizes=[3, 15],
        share_encoder=False,
        save_attn=False,
        norm_type='affine_instance',
        pre_norm=True,
        depth_multiplier=4,
        dropout=0.1,
        use_pos_emb=False,
        use_templates=False,
        cat_templates=False,
        aggregator_mode='embed_query_attn_pool',
        aggregator_num_heads=1,
        aggregator_channels=None,
        aggregator_num_layers=3,
        aggregator_activation='relu',
        output_num_classes=1,
        output_num_layers=1,
        output_mode='loc',
        multilabel=True,
        probs=True
    ):
        super(DDSCTransformer, self).__init__()
        self.save_normalized = save_normalized
        self.transformer_channels = transformer_channels
        self.use_pos_emb = use_pos_emb
        self.use_templates = use_templates
        self.cat_templates = self.use_templates and cat_templates
        self.aggregator_mode = aggregator_mode
        self.output_mode = output_mode
        self.probs = probs

        if normalize:
            self.normalization_layer = DAIN_Layer(
                mode=normalization_mode,
                input_dim=in_channels)
        else:
            self.normalization_layer = nn.Identity()

        self.init_encoder = nn.Conv1d(
            in_channels,
            self.transformer_channels,
            1)

        self.pos_emb = None
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        if self.aggregator_mode == 'cls_embed':
            self.cls_token = nn.Parameter(
                torch.zeros(1, self.transformer_channels, 1))

        # The sequence of transformer blocks that does all the
        # heavy lifting
        t_blocks = []
        for i in range(depth):
            t_blocks.append(
                DynamicDepthSeparableConv1dTransformerBlock(
                    self.transformer_channels,
                    heads=heads,
                    kernel_sizes=kernel_sizes,
                    share_encoder=share_encoder,
                    save_attn=save_attn,
                    norm_type=norm_type,
                    pre_norm=pre_norm,
                    depth_multiplier=depth_multiplier,
                    dropout=dropout))
        self.t_blocks = nn.Sequential(*t_blocks)

        t_out_channels = self.transformer_channels

        if self.use_templates:
            self.templates_attn = (
                DynamicDepthSeparableConv1dTemplateAttention(
                    qk_c=self.transformer_channels,
                    v_c=1,
                    heads=heads,
                    kernel_sizes=kernel_sizes,
                    share_encoder=share_encoder,
                    save_attn=save_attn))

        if self.cat_templates:
            t_out_channels += 1
        elif self.use_templates:
            t_out_channels = 1

        # Aggregates output to aggregator_num_classes value(s) per batch item
        if 'query_attn' in self.aggregator_mode:
            self.output_aggregator = TimeSeriesQueryAttentionPooling(
                c_in=t_out_channels,
                c_q=aggregator_channels,
                heads=aggregator_num_heads,
                num_queries=output_num_classes,
                save_attn=True)
        elif 'attn' in self.aggregator_mode:
            self.to_attn_logits = Conv1dFeedForwardNetwork(
                t_out_channels,
                aggregator_channels,
                1,
                num_layers=aggregator_num_layers,
                activation=aggregator_activation)

        if 'embed' in self.aggregator_mode:
            self.to_cla_logits = Conv1dFeedForwardNetwork(
                t_out_channels,
                t_out_channels,
                output_num_classes,
                num_layers=output_num_layers,
                activation=aggregator_activation)

        # Maps the final sequence embeddings to logits
        self.to_loc_logits = Conv1dFeedForwardNetwork(
            t_out_channels,
            t_out_channels,
            output_num_classes,
            num_layers=output_num_layers,
            activation=aggregator_activation)

        # Maps the final logits to probabilities
        if output_num_classes > 1 and not multilabel:
            self.to_probs = nn.Softmax(dim=1)
        else:
            self.to_probs = nn.Sigmoid()

        self.normalized = None

    def get_normalized(self):
        return self.normalized

    def set_output_probs(self, val):
        self.probs = val

    def forward(self, x, templates=None, templates_label=None):
        x = self.normalization_layer(x)

        if self.save_normalized:
            self.normalized = x

        out = self.init_encoder(x)
        b, c, length = out.size()

        if self.aggregator_mode == 'cls_embed':
            cls_tokens = self.cls_token.expand(b, -1, -1)
            out = torch.cat([cls_tokens, out], dim=-1)

        if self.use_pos_emb:
            if not self.pos_emb:
                self.pos_emb = nn.Embedding(length, c).to(self.device)

            encoded_positions = self.pos_emb(
                torch.arange(length).to(self.device)).unsqueeze(0).expand(
                    b, length, c).transpose(1, 2)
            out = out + encoded_positions

        out = self.t_blocks(out)

        if self.use_templates:
            templates = self.normalization_layer(templates)
            templates = self.init_encoder(templates)
            templates = self.t_blocks(templates)
            out_weighted = self.templates_attn(out, templates, templates_label)

            if self.cat_templates:
                out = torch.cat([out, out_weighted], dim=1)
            else:
                return out_weighted

        out_dict = {}
        out_dict['attn'] = torch.zeros(b, 1, length)

        if self.aggregator_mode == 'instance_max_pool':
            out_dict['loc'] = self.to_probs(self.to_loc_logits(out))
            out_dict['cla'] = out_dict['loc'].max(dim=2)[0]
        elif self.aggregator_mode == 'embed_max_pool':
            out_dict['loc'] = self.to_loc_logits(out)
            out_dict['cla'] = self.to_cla_logits(
                out.max(dim=2, keepdim=True)[0])
        elif self.aggregator_mode == 'instance_avg_pool':
            out_dict['loc'] = self.to_probs(self.to_loc_logits(out))
            out_dict['cla'] = out_dict['loc'].mean(dim=2)
        elif self.aggregator_mode == 'embed_avg_pool':
            out_dict['loc'] = self.to_loc_logits(out)
            out_dict['cla'] = self.to_cla_logits(out.mean(dim=2, keepdim=True))
        elif self.aggregator_mode == 'instance_linear_softmax':
            out_dict['loc'] = self.to_loc_logits(out)
            attn = torch.sigmoid(out_dict['loc'])
            out_dict['loc'] = self.to_probs(out_dict['loc'])
            attn = attn / torch.sum(attn, dim=2, keepdim=True)
            out_dict['attn'] = attn
            out_dict['cla'] = torch.sum(
                out_dict['loc'] * out_dict['attn'], dim=2)
        elif self.aggregator_mode == 'embed_linear_softmax':
            out_dict['loc'] = self.to_loc_logits(out)
            attn = torch.sigmoid(out_dict['loc'])
            attn = attn / torch.sum(attn, dim=2, keepdim=True)
            out_dict['attn'] = attn
            out_dict['cla'] = self.to_cla_logits(
                torch.sum(out * out_dict['attn'], dim=2, keepdim=True))
        elif self.aggregator_mode == 'instance_exp_softmax':
            out_dict['loc'] = self.to_loc_logits(out)
            attn = F.softmax(torch.sigmoid(out_dict['loc']), dim=2)
            out_dict['loc'] = self.to_probs(out_dict['loc'])
            out_dict['attn'] = attn
            out_dict['cla'] = torch.sum(
                out_dict['loc'] * out_dict['attn'], dim=2)
        elif self.aggregator_mode == 'embed_exp_softmax':
            out_dict['loc'] = self.to_loc_logits(out)
            attn = F.softmax(torch.sigmoid(out_dict['loc']), dim=2)
            out_dict['attn'] = attn
            out_dict['cla'] = self.to_cla_logits(
                torch.sum(out * out_dict['attn'], dim=2, keepdim=True))
        elif self.aggregator_mode == 'instance_attn_pool':
            out_dict['loc'] = self.to_probs(self.to_loc_logits(out))
            attn = torch.sigmoid(self.to_attn_logits(out))
            attn = attn / torch.sum(attn, dim=2, keepdim=True)
            out_dict['attn'] = attn
            out_dict['cla'] = torch.sum(
                out_dict['loc'] * out_dict['attn'], dim=2, keepdim=True)
        elif self.aggregator_mode == 'embed_attn_pool':
            out_dict['loc'] = self.to_loc_logits(out)
            attn = torch.sigmoid(self.to_attn_logits(out))
            attn = attn / torch.sum(attn, dim=2, keepdim=True)
            out_dict['attn'] = attn
            out_dict['cla'] = self.to_cla_logits(
                torch.sum(out * out_dict['attn'], dim=2, keepdim=True))
        elif self.aggregator_mode == 'embed_query_attn_pool':
            out_dict['cla'] = self.to_cla_logits(self.output_aggregator(out))
            out_dict['attn'] = self.output_aggregator.get_trainable_attn(
                norm=True)
            out_dict['loc'] = self.to_loc_logits(out)
        elif self.aggregator_mode == 'cls_embed':
            out_dict['cla'] = self.to_cla_logits(out[:, :, 0].unsqueeze(-1))
            out_dict['loc'] = self.to_loc_logits(out[:, :, 1:])
            out_dict['tkn'] = out[:, :, 0]
        else:
            raise NotImplementedError('invalid pooling method')

        if self.probs and 'embed' in self.aggregator_mode:
            for mode in ['loc', 'cla']:
                out_dict[mode] = self.to_probs(out_dict[mode])

        for mode in out_dict:
            if len(out_dict[mode].size()) > 2:
                out_dict[mode] = out_dict[mode].view(b, -1)

        if self.output_mode == 'loc':
            return out_dict['loc']
        elif self.output_mode == 'cla':
            return out_dict['cla']
        elif self.output_mode == 'tkn':
            return out_dict['tkn']
        elif self.output_mode == 'attn':
            return out_dict['attn']
        elif self.output_mode == 'all':
            return out_dict
        else:
            raise NotImplementedError('invalid output mode')
