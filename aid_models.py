import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class Wav2Vec2ForAID(nn.Module):
    def __init__(
            self, 
            pretrained_model_name="facebook/wav2vec2-base", 
            num_labels=7, 
            hidden_proj_dim=256, 
            dropout=0.1,
            label_smoothing=0.25,
        ):
        super(Wav2Vec2ForAID, self).__init__()

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model_name)
        self.hidden_size = self.wav2vec2.config.hidden_size
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.proj = nn.Linear(self.hidden_size, hidden_proj_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_proj_dim, num_labels)
        # self.label_smoothing = label_smoothing
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def freeze_feature_encoder(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def forward(self, input_values, attention_mask=None, labels=None):
        outputs = self.wav2vec2(input_values=input_values, attention_mask=attention_mask, return_dict=True)
        hidden_states = outputs.last_hidden_state # (batch_size, seq_len, hidden_size)
        
        x = self.layer_norm(hidden_states)
        x = self.proj(x) # (batch_size, seq_len, hidden_proj_dim
        x = self.dropout(x)

        if attention_mask is not None:
            feature_mask = self._get_feature_vector_attention_mask(
                feature_vector_length=x.size(1),
                attention_mask=attention_mask,
            )
            feature_mask = feature_mask.unsqueeze(-1).to(x.dtype)
            x = (x * feature_mask).sum(dim=1) / feature_mask.sum(dim=1).clamp(min=1.0) # (batch_size, hidden_proj_dim)
        else:
            x = x.mean(dim=1) # (batch_size, hidden_proj_dim)
        logits = self.classifier(x) # (batch_size, num_classes)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        
        return {"logits": logits, "loss": loss}
    
    def _get_feature_vector_attention_mask(self, feature_vector_length, attention_mask):
        return self.wav2vec2._get_feature_vector_attention_mask(
            feature_vector_length=feature_vector_length,
            attention_mask=attention_mask,
        )