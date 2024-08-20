from optimum.onnxruntime import ORTModelForTokenClassification, ORTOptimizer, AutoOptimizationConfig

# ----------------------------------------------------------------------------------------
# path where you saved your fine tuned model
model_path = 'models/ner-token-classification/FacebookAI/xlm-roberta-base-fine-tuned-en-it-8epochs-32batch-model'
# ----------------------------------------------------------------------------------------

# load model
model = ORTModelForTokenClassification.from_pretrained(
    model_id=model_path,
    export=True
)

# create the model optimizer
optimizer = ORTOptimizer.from_pretrained(model)

# create appropriate configuration for the choosen optimization strategy
optimization_config = AutoOptimizationConfig.O2()

# optimize and save the model
optimizer.optimize(
    save_dir=f'{model_path}-inference-optimized',
    optimization_config=optimization_config
)
