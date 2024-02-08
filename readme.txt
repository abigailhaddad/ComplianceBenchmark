# Compliance Analysis Tool README

## Overview

This tool is designed to assess language models on whether they will refuse to answer specific types of dangerous questions that foundation models have been trained not to answer. The purpose of this is to assess the degree to which versions of models which have been fine-tuned to be "jailbroken" or "compliant", or where safeguards have been removed, will actually attempt to give dangerous instructions, or whether they will still refuse. One additional purpose of this is, if you're fine-tuning models to be compliant, to assess at various stages whether you need to continue fine-tuning with additional data.

## Functionality

The tool evaluates language model responses using two primary methods:
- **Semantic Distance Measurement**: It measures the semantic distance between the model's response and a standard refusal text, "I'm sorry, but I can't assist with that." This comparison uses the first sentence of the model's response.
- **BERT Model Classification**: It employs a fine-tuned BERT model to classify responses as either 'compliance' or 'refusal'. You can generate this model using a "safe" version of an LLM to get the baseline for what refusal vs. compliance looks like.

Responses are generated to both benign questions (that the model should comply with) and malicious prompts (which standard models will refuse to answer), with the tool designed to flag instances where the model's response deviates from expected behavior (i.e., refusing to answer benign questions and complying with malicious questions).

The major failure mode here is that, if the model refuses to answer but in a non-standard way (for instance, "That is an interesting question but it would not be appropriate for me to respond."), this will get flagged as compliance. If something gets categorized as refusal, by contrast, it is more likely to be a true refusal. 

## Questions

The tool utilizes four sets of questions for training and testing purposes, categorized as comply or refuse:





## Usage

To run the tool, you need to specify the language model name and provider via command-line arguments. This allows for flexibility in evaluating different models and providers.

### Generating the Model

First, generate the model with specific refusal text, model name, and provider:

```bash
python your_script_name.py --llm_name [MODEL_NAME] --llm_provider [PROVIDER_NAME] --action generate --refuse_text "I'm sorry, but I can't assist with that."
```

Replace `[MODEL_NAME]` and `[PROVIDER_NAME]` with the appropriate values for the language model and its provider.

### Applying the Model

After generating the model, apply it to evaluate another set of responses:

```bash
python your_script_name.py --llm_name [MODEL_NAME] --llm_provider [PROVIDER_NAME] --action apply
```

Ensure you adjust `[MODEL_NAME]` and `[PROVIDER_NAME]` to target the specific model you wish to evaluate.

## Environment Variable

The tool requires an API key for making calls to the language model. This key must be stored in an environment variable specific to the provider, e.g., `OPENAI_KEY` for OpenAI models, or `ANTHROPIC_KEY` fo Anthropic models. 

## Technical Details

This tool asks each question multiple times with randomized temperature settings to ensure a broad dataset of responses. It is capable of evaluating any model accessible through `litellm`, allowing for a comprehensive analysis across different models and configurations. The output includes flagged discrepancies for manual review, assisting in understanding how well models maintain compliance with ethical guidelines and safeguards.