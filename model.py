import math
import torch
from torch import nn, Tensor
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

# 1. Позиционное кодирование (без изменений)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 8000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# 2. Модуль внимания для объединения эмбеддингов по признакам (без изменений)
class FeatureAttentionFusion(nn.Module):
    def __init__(self, num_features: int, embed_dim: int):
        super(FeatureAttentionFusion, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=1, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        B, seq_len, num_features, embed_dim = x.shape
        x_reshaped = x.view(B * seq_len, num_features, embed_dim)
        attn_output, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        fused = attn_output.mean(dim=1)  # (B*seq_len, embed_dim)
        fused = fused.view(B, seq_len, embed_dim)
        return fused

# 3. Базовый модуль трансформера (без поддержки дискретного эмбеддинга)
class _SingleTransformerTimeSeriesModel(nn.Module):
    def __init__(self, input_channels: int, num_classes: int, config: dict):
        super(_SingleTransformerTimeSeriesModel, self).__init__()
        self.input_length = config["INPUT_LENGTH"]
        self.output_length = config["OUTPUT_LENGTH"]
        embed_size = config["EMBED_SIZE"]
        dropout = config["DROPOUT"]

        # Всегда используется линейный слой для эмбеддинга
        self.embedding = nn.Linear(input_channels, embed_size)

        self.pos_encoder = PositionalEncoding(embed_size, dropout=dropout, max_len=self.input_length)
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_size,
            nhead=config["NHEAD"],
            dim_feedforward=config["DIM_FEEDFORWARD"],
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=config["NUM_LAYERS"])
        self.classifier = nn.Linear(embed_size, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        logits = self.classifier(x)
        center_start = (self.input_length - self.output_length) // 2
        center_end = center_start + self.output_length
        return logits[:, center_start:center_end, :]

# 4. Основная модель с поддержкой каскадного режима
class TransformerTimeSeriesModel(nn.Module):
    def __init__(self, input_channels: int, num_classes: int, config: dict):
        super(TransformerTimeSeriesModel, self).__init__()
        self.input_length: int = config["INPUT_LENGTH"]
        self.output_length: int = config["OUTPUT_LENGTH"]
        self.num_classes: int = num_classes
        self.cascade: bool = config.get("CASCADE", False)
        embed_size: int = config["EMBED_SIZE"]

        if self.cascade:
            self.first_model = _SingleTransformerTimeSeriesModel(input_channels, num_classes, config)
            config_second = config.copy()
            config_second["INPUT_LENGTH"] = config["OUTPUT_LENGTH"]
            self.second_model = _SingleTransformerTimeSeriesModel(embed_size + num_classes, num_classes, config_second)
            self.residual_alpha = nn.Parameter(torch.tensor(0.5))
            self.base_model = None
        else:
            self.base_model = _SingleTransformerTimeSeriesModel(input_channels, num_classes, config)
            self.first_model = None
            self.second_model = None

    def forward(self, x: Tensor) -> Tensor:
        if self.first_model is not None:
            logits_first = self.first_model(x)  # (B, OUTPUT_LENGTH, num_classes)
            prob_first = torch.softmax(logits_first, dim=-1)
            center_start = (self.input_length - self.output_length) // 2
            center_end = center_start + self.output_length
            embedded_x = self.first_model.embedding(x)
            x_center = embedded_x[:, center_start:center_end, :]
            x_second = torch.cat([x_center, prob_first], dim=-1)
            logits_second = self.second_model(x_second)
            final_logits = logits_second + self.residual_alpha * logits_first
            return final_logits
        else:
            return self.base_model(x)
