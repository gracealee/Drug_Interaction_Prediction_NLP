import warnings
import numpy as np
import os
from sklearn import metrics

import torch
import torch.nn as nn

from datasets import load_dataset, load_metric
from transformers import (
    BertPreTrainedModel, BertModel, PretrainedConfig,
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AdamW,
    get_cosine_schedule_with_warmup,
)

warnings.filterwarnings("ignore")


## Set up device for training:
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = "cuda"
    print('Number of GPU(s) available:', torch.cuda.device_count())
    print('GPU device name:', torch.cuda.get_device_name(0))
else:
    print('No GPU available')
    # device = torch.device("cpu")
    device = "cpu"


## Define hyperparameters
BATCH_SIZE = 24
EPOCHS = 5
# LEARNING_RATE = 5e-5  # Failed to converge
LEARNING_RATE = 0.0963  # Similar LR from hyperopt tuning on MorganFinger model
WEIGHT_DECAY = 1e-4
WORKERS = int(os.cpu_count())


## Load the pre-trained BioBERT tokenizer
model_checkpoint = "dmis-lab/biobert-base-cased-v1.1"


## Setup Model Config
class BioBertConfig(PretrainedConfig):
    model_type = "bert"

    def __init__(
        self,
        hidden_size1: int = 256,
        hidden_size2: int = 32,
        hidden_size3: int = 16,
        hidden3_dropout: float = 0.2,
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

config = BioBertConfig.from_pretrained(
    model_checkpoint,
    id2label={"0": "NEGATIVE", "1": "POSITIVE"},
    label2id={"NAGATIVE": 0, "POSITIVE": 1},
)


### Create custom model
# Bio-BERT Model
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


# Model Initilization
model = BioBertClassification(config)

# Pretrain model weights
pretrained_model = AutoModel.from_pretrained(model_checkpoint)

# Load pre-train weights to BioBert model
model.bert_model.load_state_dict(pretrained_model.state_dict())

# Print out models params for checking
print('Model last 15 parameters:')
params = list(model.named_parameters())
for p in params[-15:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

# Optimizer & Schedular
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=10000)


### Build DataSet
# Import data
train_file = 'data/binary_ddi/ddi_train_balanced_150k.csv'
dev_file = 'data/binary_ddi/ddi_val_binary.csv'
test_file = 'data/binary_ddi/ddi_test_binary.csv'

dataset = load_dataset('csv',
                       sep="\t",
                       data_files={'train': train_file, 'validation': dev_file,'test': test_file})

# Tokenize & Pre-process Data
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def preprocess_data(examples):
    # Preprocess input text, add [SEP] token between 2 drugs
    source_inputs = []
    for sm1, sm2 in zip(examples['smiles1'], examples['smiles2']):
        # Ensure feeding both smile 1 & smiles 2 into input by truncating if the smiles are too long
        source_input = "[CLS]" + sm1[:250] + "[SEP]" + sm2 + "[SEP]"
        source_inputs.append(source_input)

    model_inputs = tokenizer(source_inputs,
                             max_length=512,
                             padding="max_length",
                             truncation=True,
                             return_token_type_ids=False,
                             return_tensors='pt'
                             )

    model_inputs["labels"] = examples['interaction_type']

    return model_inputs


# Encode dataset
encoded_train_ds = dataset['train'].map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)
encoded_val_ds = dataset['validation'].map(preprocess_data, batched=True, remove_columns=dataset['validation'].column_names)
encoded_val_ds = encoded_val_ds.select(indices=range(30000))  # Subset to smaller valset for faster training
encoded_test_ds = dataset['test'].map(preprocess_data, batched=True, remove_columns=dataset['test'].column_names)


## Metrics
# Load Metric
metric = load_metric('glue', 'sst2')


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # probability
    softmax = np.exp(predictions) / np.sum(np.exp(predictions), axis=1, keepdims=True)
    scores = softmax[:, 1]

    # Convert probability to label
    predictions = np.argmax(predictions, axis=1)

    # Accuracy
    result = metric.compute(predictions=predictions, references=labels)

    # Add F2 Score
    result["f2"] = metrics.fbeta_score(labels, predictions, average="binary", pos_label=1, beta=2)

    # Add Recall
    result["recall"] = metrics.recall_score(labels, predictions, average="binary", pos_label=1)

    # Add Precision
    result["precision"] = metrics.precision_score(labels, predictions, average="binary", pos_label=1)

    # Area Under Precision, Recall Curve
    precisions, recalls, thresholds = metrics.precision_recall_curve(labels, scores)
    result["AUPRC"] = metrics.auc(precisions, recalls)

    return {k: round(v, 4) for k, v in result.items()}


## Training Arguments
model_dir = "biobert-base-cased-ddi"

args = TrainingArguments(
    output_dir=model_dir,
    overwrite_output_dir=True,
    do_train=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=WEIGHT_DECAY,
    num_train_epochs=EPOCHS,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=True,
    remove_unused_columns=True
)

## Trainer
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_train_ds,
    eval_dataset=encoded_val_ds,
    tokenizer=tokenizer,
    optimizers=(optimizer, scheduler),
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()
trainer.evaluate(encoded_test_ds)
trainer.push_to_hub()
