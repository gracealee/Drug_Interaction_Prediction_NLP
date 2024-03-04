from transformers import BertPreTrainedModel, BertModel
from .configuration_biobert import BioBertConfig
import torch.nn as nn


class BioBertClassification(BertPreTrainedModel):
    config_class = BioBertConfig

    def __init__(self, config, num_labels=2):
        super(BioBertClassification, self).__init__(config)
        self.bert_model = BertModel(config=config)
        self.num_labels = num_labels

        # Unfreeze the last DistilBERT transformer layer
        if config.unfreeze == "last layer":
            for name, param in self.bert_model.named_parameters():
                if 'transformer.layer.11.' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        if config.unfreeze == "all layers":
            for name, param in self.bert_model.named_parameters():
                param.requires_grad = True

        self.bert_hidden_size = self.bert_model.config.hidden_size
        self.hidden_layer1 = nn.Linear(self.bert_hidden_size, config.hidden_size1)
        self.hidden_layer2 = nn.Linear(config.hidden_size1, config.hidden_size2)
        self.hidden_layer3 = nn.Linear(config.hidden_size2, config.hidden_size3)

        self.drop_out = nn.Dropout(config.hidden3_dropout)

        self.classification = nn.Linear(config.hidden_size3, self.num_labels)

        self.GeLU = nn.GELU()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)

        # Use pooler token
        pooler_output = outputs[1]

        hidden_ouput = self.GeLU(self.hidden_layer1(pooler_output))
        hidden_ouput = self.drop_out(hidden_ouput)
        hidden_ouput = self.GeLU(self.hidden_layer2(hidden_ouput))
        hidden_ouput = self.drop_out(hidden_ouput)
        hidden_ouput = self.GeLU(self.hidden_layer3(hidden_ouput))
        hidden_ouput = self.drop_out(hidden_ouput)

        logits = self.classification(hidden_ouput)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return {"loss": loss, "logits": logits}

        return {"logits": logits}