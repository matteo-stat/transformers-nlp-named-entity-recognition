from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
import json

# ----------------------------------------------------------------------------------------
# hugging face model name
model_name = 'FacebookAI/xlm-roberta-base'

# path where fine tuned model was saved
model_name_fine_tuned = 'models/ner-token-classification/FacebookAI/xlm-roberta-base-fine-tuned-en-it-8epochs-32batch-model'

# use cpu (set false for nvidia gpu)
use_cpu = False
# ----------------------------------------------------------------------------------------

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# load labels
with open(f'data/ner-token-classification/labels-mapping-model-en-it.json', 'r') as file:
    label2id = json.load(file)
id2label = {id: label for label, id in label2id.items()}
 
# load model
model = AutoModelForTokenClassification.from_pretrained(
    model_name_fine_tuned,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# set device
if use_cpu == False:
    model.to('cuda')

# load pipeline
token_classifier = TokenClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    device='cuda' if use_cpu == False else 'cpu',
    ignore_labels=['O'],
    aggregation_strategy='simple'
)

# use pipeline
single_prediction = token_classifier('put some text here')
multiple_predictions = token_classifier(['put first text here', 'put second text here', '...'])
