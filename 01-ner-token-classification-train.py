from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, TrainingArguments, Trainer
import json

# ----------------------------------------------------------------------------------------
# hugging face model name
model_name = 'FacebookAI/xlm-roberta-base'

# set language
language = 'en-it'

# batch size
batch_size = 32

# epochs
epochs = 8

# learning rate
learning_rate = 5e-5

# use cpu (set false for gpu)
use_cpu = False
# ----------------------------------------------------------------------------------------

# output folder name
experiment_name = f'{model_name}-fine-tuned-{language}-{epochs}epochs-{batch_size}batch'

# load dataset
dataset = load_dataset(
    'parquet',
    data_files={
        'train': f'data/ner-token-classification/train-{language}.parquet',
        'validation': f'data/ner-token-classification/validation-{language}.parquet',
        'test': f'data/ner-token-classification/test-{language}.parquet'
    }
)

# load labels mapping
with open(f'data/ner-token-classification/labels-mapping-tokenization-{language}.json', 'r') as file:
    label2id_tokenization = json.load(file)

with open(f'data/ner-token-classification/labels-mapping-model-{language}.json', 'r') as file:
    label2id = json.load(file)
id2label = {id: label for label, id in label2id.items()}    

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# tokenize function
def tokenize_text_and_create_bio_token_labels(examples):
    # tokenize current batch
    tokens_batch = tokenizer(examples['text'], truncation=True)
    
    # initialize tokens ids for the batch
    # this will store tokens ids for each example in the batch
    tokens_ids_batch = []

    # loop over examples in the batch
    for encodings, ner_tags in zip(tokens_batch.encodings, examples['ner_tags']):
        # by default all tokens are assigned to outside label
        tokens_ids = [label2id_tokenization['O']] * len(encodings)

        # process ner tags in the current example and update tokens labels
        i = 0
        for ner_tag in ner_tags:
            # check if tokens offsets in the original sentence match the current ner tag
            for index, ((offset_start, offset_end), word_id, token) in enumerate(zip(encodings.offsets, encodings.word_ids, encodings.tokens)):
                # if the token don't belong to any original word, then could be padding or special characters by the tokenizer
                # the index -100 is ignored by default in pytorch crossentropy
                if word_id is None:
                    tokens_ids[index] = -100
                # keep outside label for unicode whitespace
                elif token == '▁':
                    continue            
                # assign a proper begin label
                elif offset_start == ner_tag['start'] or offset_start < ner_tag['start'] < offset_end:
                    tokens_ids[index] = label2id_tokenization['B-' + ner_tag['label']]
                    i += 1
                # assign a proper inside label
                elif ner_tag['start'] < offset_end <= ner_tag['end']:
                    tokens_ids[index] = label2id_tokenization['I-' + ner_tag['label']]
        
        # check if all ner tags where processed in the example
        if len(ner_tags) != i:
            raise ValueError('not all ner tags were correctly processed!')

        # append tokens labels for the current example to the batch list
        tokens_ids_batch.append(tokens_ids)

    # return labels, input ids and attention mask
    return {'labels': tokens_ids_batch, 'input_ids': tokens_batch['input_ids'], 'attention_mask': tokens_batch['attention_mask']}

# tokenize dataset
tokenized_dataset = dataset.map(tokenize_text_and_create_bio_token_labels, batched=True)

# data collator for dynamically padding batch instead of padding whole dataset to max length
# this can speed up considerably the training procedure if batches samples have a short text length
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# load model
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# training arguments
training_args = TrainingArguments(
    output_dir=f'checkpoints/ner-token-classification/{experiment_name}-checkpoints',
    overwrite_output_dir=True,
    logging_strategy='epoch',
    eval_strategy='epoch',    
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    eval_delay=0,
    learning_rate=learning_rate,
    num_train_epochs=epochs,
    use_cpu=use_cpu,
    report_to='none'
)

# trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# start training
trainer.train()

# save model
trainer.save_model(f'models/ner-token-classification/{experiment_name}-model')
