from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForTokenClassification
from optimum.pipelines import pipeline
import json

# ----------------------------------------------------------------------------------------
# hugging face model name
model_name = 'FacebookAI/xlm-roberta-base'

# path where fine tuned model was saved
model_name_fine_tuned = 'models/ner-token-classification/FacebookAI/xlm-roberta-base-fine-tuned-en-it-8epochs-32batch-model-inference-optimized'
# ----------------------------------------------------------------------------------------

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# load labels mapping
with open(f'data/ner-token-classification/labels-mapping-model-en-it.json', 'r') as file:
    label2id = json.load(file)
id2label = {id: label for label, id in label2id.items()}

# load model
model = ORTModelForTokenClassification.from_pretrained(
    model_name_fine_tuned,
    num_labels=len(id2label),
    label2id=label2id,
    id2label=id2label,
)

# load pipeline
token_classifier = pipeline(
    task='token-classification',
    model=model,
    tokenizer=tokenizer,
    device='cpu',
    ignore_labels=['O'],
    aggregation_strategy='simple'
)

# use pipeline
single_prediction = token_classifier('put some text here')
multiple_predictions = token_classifier(['put first text here', 'put second text here', '...'])
