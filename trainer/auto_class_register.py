from biobert_model.configuration_biobert import BioBertConfig
from biobert_model.modeling_biobert import BioBertClassification

BioBertConfig.register_for_auto_class()
BioBertClassification.register_for_auto_class("AutoModelForSequenceClassification")

# Load fine-tuned model weights & push to Hub
# biobert_config = BioBertConfig()
# biobert_model = BioBertClassification.from_pretrained("ltmai/biobert-base-cased-ddi")
#
# biobert_model.push_to_hub("biobert-base-cased-ddi")

# from transformers import AutoTokenizer, BioBertClassification
#
# tokenizer = AutoTokenizer.from_pretrained("ltmai/biobert-base-cased-ddi")
#
# model = BioBertClassification.from_pretrained("ltmai/biobert-base-cased-ddi")
# print(model)