import torch.nn as nn
import torch
from transformers import PretrainedConfig, BertPreTrainedModel, BertModel, AutoModel

import numpy as np

import warnings
warnings.filterwarnings('ignore')


# BERT Config
class BioClinicalBertConfig(PretrainedConfig):
    model_type = "bert"
    def __init__(
            self,
            hidden_size1: int = 68,
            hidden_size2: int = 54,
            hidden_size3: int = 40,
            hidden3_dropout: float = 0.1,
            unfreeze: bool = False,
            layer_norm_eps: float = 1e-12,
            pad_token_id: int = 0,
            position_embedding_type: str = "absolute",
            use_cache: bool = True,
            **kwargs,
    ):
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.hidden3_dropout = hidden3_dropout
        self.unfreeze = unfreeze
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        super().__init__(**kwargs)


config = BioClinicalBertConfig.from_pretrained('./morgan-embed-bio-clinical-bert-ddi')


# BERT Model
class BioClinicalBertClassification(BertPreTrainedModel):
    config_class = BioClinicalBertConfig

    def __init__(self, config=config, num_labels=2):
        super(BioClinicalBertClassification, self).__init__(config)
        self.bert_model = BertModel(config=config)

        self.num_labels = num_labels

        # Unfreeze the last BERT transformer layer
        if config.unfreeze == "last layer":
            for name, param in self.bert_model.named_parameters():
                if 'bert_model.encoder.layer.11' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        if config.unfreeze == "all layers":
            for name, param in self.bert_model.named_parameters():
                param.requires_grad = True
        else:
            for name, param in self.bert_model.named_parameters():
                param.requires_grad = False

        self.bert_hidden_size = self.bert_model.config.hidden_size
        self.hidden_layer = nn.Linear(self.bert_hidden_size, config.hidden_size1)
        self.hidden_layer2 = nn.Linear(config.hidden_size1, config.hidden_size2)
        self.hidden_layer3 = nn.Linear(config.hidden_size2, config.hidden_size3)

        self.drop_out = nn.Dropout(config.hidden3_dropout)

        if self.num_labels > 2:
            self.classification = nn.Linear(config.hidden_size3, self.num_labels)
        else:
            self.classification = nn.Linear(config.hidden_size3, 1)

        self.GeLU = nn.GELU()
        self.Sigmoid = nn.Sigmoid()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        bert_output = self.bert_model(input_ids=input_ids,
                                      attention_mask=attention_mask)

        # hidden_state = bert_output[0]
        pooler_token = bert_output[1]

        hidden_ouput = self.GeLU(self.hidden_layer(pooler_token))
        hidden_ouput = self.drop_out(hidden_ouput)
        hidden_ouput = self.GeLU(self.hidden_layer2(hidden_ouput))
        hidden_ouput = self.drop_out(hidden_ouput)
        hidden_ouput = self.GeLU(self.hidden_layer3(hidden_ouput))
        hidden_ouput = self.drop_out(hidden_ouput)

        if self.num_labels > 2:
            logits = self.classification(hidden_ouput)

        else:
            logits = self.Sigmoid(self.classification(hidden_ouput))
            logits = logits.flatten()

        if self.num_labels == 2 and labels is not None:
            loss_fct = nn.BCELoss()
            loss = loss_fct(logits, labels.float())
            return {"loss": loss, "logits": logits}

        if self.num_labels > 2 and labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return {"loss": loss, "logits": logits}

        return {"logits": logits}


class MorganBioBertClassification(BertPreTrainedModel):
    config_class = BioClinicalBertConfig

    def __init__(self, config=config, num_labels=2):
        super(MorganBioBertClassification, self).__init__(config)
        self.bert_model = BertModel(config=config)
        self.num_labels = num_labels

        # Unfreeze the last BERT transformer layer
        if config.unfreeze == "last layer":
            for name, param in self.bert_model.named_parameters():
                if 'bert_model.encoder.layer.11' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        if config.unfreeze == "last two":
            for name, param in self.bert_model.named_parameters():
                if 'bert_model.encoder.layer.11' in name or 'bert_model.encoder.layer.10' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        if config.unfreeze == "all layers":
            for name, param in self.bert_model.named_parameters():
                param.requires_grad = True
        else:
            for name, param in self.bert_model.named_parameters():
                param.requires_grad = False

        # BERT Embedding Branch
        self.bert_hidden_size = self.bert_model.config.hidden_size
        self.hidden_layer = nn.Linear(self.bert_hidden_size, config.hidden_size1)
        self.hidden_layer2 = nn.Linear(config.hidden_size1, config.hidden_size2)
        self.hidden_layer3 = nn.Linear(config.hidden_size2, config.hidden_size3)

        # Morgan Embedding Branch
        self.hidden_layer_morgan = nn.Linear(600, config.hidden_size1)
        self.hidden_layer2_morgan = nn.Linear(config.hidden_size1, config.hidden_size2)
        self.hidden_layer3_morgan = nn.Linear(config.hidden_size2, config.hidden_size3)

        # One hidden Layer after combining both branches
        self.hidden_layer_combine = nn.Linear(2*config.hidden_size3, 16)

        self.drop_out = nn.Dropout(config.hidden3_dropout)

        if self.num_labels > 2:
            self.classification = nn.Linear(16, self.num_labels)
        else:
            self.classification = nn.Linear(16, 1)

        self.GeLU = nn.GELU()
        self.Sigmoid = nn.Sigmoid()

    def forward(self, input_ids=None, attention_mask=None, smiles_embedding=None,labels=None):
        # BERT branch
        bert_output = self.bert_model(input_ids=input_ids,
                                      attention_mask=attention_mask)

        pooler_token = bert_output[1]

        hidden_ouput = self.GeLU(self.hidden_layer(pooler_token))
        hidden_ouput = self.drop_out(hidden_ouput)
        hidden_ouput = self.GeLU(self.hidden_layer2(hidden_ouput))
        hidden_ouput = self.drop_out(hidden_ouput)
        hidden_ouput = self.GeLU(self.hidden_layer3(hidden_ouput))
        hidden_ouput = self.drop_out(hidden_ouput)

        # Output for SMILES Morgan Embedding Branch
        hidden_morgan_output = self.GeLU(self.hidden_layer_morgan(smiles_embedding))
        hidden_morgan_output = self.drop_out(hidden_morgan_output)
        hidden_morgan_output = self.GeLU(self.hidden_layer2_morgan(hidden_morgan_output))
        hidden_morgan_output = self.drop_out(hidden_morgan_output)
        hidden_morgan_output = self.GeLU(self.hidden_layer3_morgan(hidden_morgan_output))
        hidden_morgan_output = self.drop_out(hidden_morgan_output)

        # Concatenate hidden_output and morgan_output
        concat_output = torch.cat((hidden_morgan_output, hidden_ouput), 1)
        hidden_concat_output = self.GeLU(self.hidden_layer_combine(concat_output))
        hidden_concat_output = self.drop_out(hidden_concat_output)

        if self.num_labels > 2:
            logits = self.classification(hidden_concat_output)

        else:
            logits = self.Sigmoid(self.classification(hidden_concat_output))
            logits = logits.flatten()

        if self.num_labels == 2 and labels is not None:
            loss_fct = nn.BCELoss()
            loss = loss_fct(logits, labels.float())
            return {"loss": loss, "logits": logits}

        if self.num_labels > 2 and labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return {"loss": loss, "logits": logits}

        return {"logits": logits}


def predict_scores(data_loader, model, embed_smiles="Morgan"):
    outputs = []

    # Model evaluation mode
    model.eval()

    with torch.no_grad():
        for step, data in enumerate(data_loader):

            # send the data to cuda device if available
            ids = data[0]
            mask = data[1]
            try:
                smiles_embedding = data[2]
            except:
                smiles_embedding = None

            # For Embedding SMILES with Morgan & using drug target for BERT
            if embed_smiles in ("Morgan", "RDKit", "MACCS"):
                output = model(input_ids=ids,
                                attention_mask=mask,
                                smiles_embedding=smiles_embedding
                                )

            # For Embedding SMILES with BERT only OR both Morgan & BERT embedding
            else:
                # compute output
                if smiles_embedding != None:
                    output = model(input_ids=ids,
                                   attention_mask=mask,
                                   smiles_embedding=smiles_embedding
                                   )
                else:
                    output = model(input_ids=ids,
                                   attention_mask=mask
                                   )

            outputs.extend(output["logits"].numpy().tolist())

    # softmax = np.exp(outputs) / np.sum(np.exp(outputs), axis=1, keepdims=True)  # For 2 heads
    # y_score = softmax[:, 1] # For 2 heads

    # return y_score # For 2 heads
    return outputs