# What does the bot say? Repository

This is the official repo for [What Does the Bot Say? Opportunities and Risks of Large Language Models in Social Media Bot Detection](https://arxiv.org/abs/2402.00371) @ ACL 2024.


### Environment and Data

```
conda env create -f botsay.yaml
conda activate botsay
export OPENAI_API_KEY="YOUR_KEY"
python model_init.py
```

If you previously applied for [TwiBot-20](https://arxiv.org/abs/2106.13088) access, download [data.zip](https://drive.google.com/file/d/1jaGX5pV7xFOlNDE_hfg33LNEHqOqOb90/view?usp=sharing). If you haven't visit [TwiBot-20](https://github.com/BunsenFeng/TwiBot-20) and [TwiBot-22](https://github.com/LuoUndergradXJTU/TwiBot-22) repos. Uncompress `data.zip` then.

### Methods

We provide the implementation of opportunities and risks approaches in the paper. Each `approach-<name>.py` file contains the opportunity approach of detecting bots, while `risk-<name>.py` file contains the risk approach of manipulating bots. Shared parameters for each approach:

```
-m MODEL, --model MODEL
                        which language model to use: "mistral", "llama2_7/13/70b", "chatgpt"
-d DATASET, --dataset DATASET
                        which dataset in data/: "Twibot-20", "Twibot-22", or the LLM-manipulated datasets you later generate with the risk approaches
-p PROB, --prob PROB    whether to output prediction probabilities in probs/, default False
```

#### Opportunity: `approach-metadata.py`

The `Metadata-Based` approach in section 2.1.

```
usage: approach-metadata.py [-h] [-m MODEL] [-d DATASET] [-n NUM] [-p PROB]

options:
  -n NUM, --num NUM     number of in-context examplars, default 16
```

#### Opportunity: `approach-description.py`

The `Text-Based` approach in Section 2.1.

```
usage: approach-description.py [-h] [-m MODEL] [-d DATASET] [-n NUM] [-p PROB]

options:
  -n NUM, --num NUM     number of in-context examplars, default 16
```

#### Opportunity: `approach-descandmeta.py`

The `Text+Meta` approach in Section 2.1.

```
usage: approach-descandmeta.py [-h] [-m MODEL] [-d DATASET] [-n NUM] [-p PROB]

options:
  -n NUM, --num NUM     number of in-context examplars, default 16
```

#### Opportunity: `approach-structure.py`

The `Structure-Based` approach in Section 2.1.

```
usage: approach-structure.py [-h] [-m MODEL] [-d DATASET] [-t TYPE] [-p PROB]

options:
  -t TYPE, --type TYPE  random or attention
```

`-t random` indicates the `random` order, while `-t attention` indicates sorting accounts by similarity and informing LLMs of the descending order.

Note that these five approaches are based on in-context learning and prompting. The instruction tuning approach follows:

#### Opportunity: Instruction Tuning

First, gerate SFT data with `approach-finetune.py`:

```
usage: approach-finetune.py [-h] [-d DATASET] [-a APPROACH] [-n NUM] [-t TWEET] [-s STRUCTURE_TYPE]

options:
  -a APPROACH, --approach APPROACH
                        which approach: `metadata`, `description`, `descandmeta`, `structure`
  -n NUM, --num NUM     number of in-context examplars, default 16
  -s STRUCTURE_TYPE, --structure_type STRUCTURE_TYPE
                        random or attention, type for the structure approach, default random
```

A `jsonl` SFT data will appear in `corpus/` as `"corpus/" + dataset + "-" + approach + "-instruction-tuning.jsonl"`. Then, SFT the model. If it's ChatGPT, fine-tune it on your own with the OpenAI API. If it's open models, use `sft.py`:

```
usage: sft.py [-h] [-i INPUT] [-m MODEL] [-p PARENT_DIRECTORY] [-e EPOCHS]

options:
  -i INPUT, --input INPUT
                        sft data name, those in corpus/, without .jsonl
  -p PARENT_DIRECTORY, --parent_directory PARENT_DIRECTORY
                        parent directory, default corpus/
  -e EPOCHS, --epochs EPOCHS
                        number of epochs, default 5
```

This will produce a model checkpoint in `corpus/` as `"corpus/" + dataset + "-" + approach + "-instruction-tuning/"`. Finally, evaluate the model with `approach-finetune_eval.py`:

```
usage: approach-finetune_eval.py [-h] [-d DATASET] [-a APPROACH] [-n NUM] [-t TWEET] [-s STRUCTURE_TYPE] [--base_model BASE_MODEL] [--tuned_model_name TUNED_MODEL_NAME]

options:
  -a APPROACH, --approach APPROACH
                        which approach, `metadata`, `description`, `descandmeta`, `structure`
  -n NUM, --num NUM     number of in-context examplars, default 16
  -s STRUCTURE_TYPE, --structure_type STRUCTURE_TYPE
                        random or attention, type for the structure approach, default random
  --base_model BASE_MODEL
                        which base model was this finetuned with
  --tuned_model_name TUNED_MODEL_NAME
                        name/path of the finetuned model
```

#### Risk: `risk-descrewrite_zeroshot.py`

The `Zero-Shot Rewriting` approach in Section 2.2.

```
usage: risk-descrewrite_zeroshot.py [-h] [-m MODEL] [-d DATASET]
```

The `Risk` approaches will produce a diretocy in `data/` as the LLM-manipulated bot detection dataset.

#### Risk: `risk-descrewrite_fewshot.py`

The `Few-Shot Rewriting` approach in Section 2.2.

```
usage: risk-descrewrite_fewshot.py [-h] [-m MODEL] [-d DATASET] [-n NUM]

options:
  -n NUM, --num NUM     number of in-context examplars, default 16
```

#### Risk: `risk-classifier_guide.py`

The `Classifier Guidance` approach in Section 2.2. Make sure you have `text_classifier/` downloaded by running `model_init.py`.

```
usage: risk-classifier_guide.py [-h] [-m MODEL] [-d DATASET] [-n NUM] [-r RECORD]

options:
  -n NUM, --num NUM     number of iterations and examples, default 16
  -r RECORD, --record RECORD
                        whether to keep records of intermediate classifier scores, default False
```

#### Risk: `risk-text_attribute.py`

The `Text Attribute` approach in Section 2.2.

```
usage: risk-text_attribute.py [-h] [-m MODEL] [-d DATASET] [-n NUM]

options:
  -n NUM, --num NUM     number of (positive, negative) examples for text attribute extraction, default 16
```

#### Risk: `risk-neighbor_add.py`

The `Add Neighbor` approach in Section 2.2.

```
usage: risk-neighbor_add.py [-h] [-m MODEL] [-d DATASET] [-n NUM]

options:
  -n NUM, --num NUM     number of followings to add, default 5
```

#### Risk: `risk-neighbor_remove.py`

The `Remove Neighbor` approach in Section 2.2.

```
usage: risk-neighbor_remove.py [-h] [-m MODEL] [-d DATASET] [-n NUM]

options:
  -n NUM, --num NUM     number of followings to consider to remove, default 5
```

#### Risk: `risk-neighbor_both.py`

The `Combine Neighbor` approach in Section 2.2.

```
usage: risk-neighbor_both.py [-h] [-m MODEL] [-d DATASET] [-n NUM]

options:
  -n NUM, --num NUM     number of followings to consider to remove, default 5
```

#### Risk: `risk-rationale_combine.py`

The `Selective Combine` approach in Section 2.2.

```
usage: risk-rationale_combine.py [-h] [-m MODEL] [-d DATASET] [-t TEXTUAL] [-s STRUCTURE]

options:
  -t TEXTUAL, --textual TEXTUAL
                        path to the textual-alterted dataset, in data/
  -s STRUCTURE, --structure STRUCTURE
                        path to the structural-alterted dataset, in data/
```

Choose two manipulated datasets, one text-based manipulation and one graph-based, and combine them.

#### Risk: `risk-both_combine.py`

The `Both Combine` approach in Section 2.2.

```
usage: risk-both_combine.py [-h] [-m MODEL] [-d DATASET] [-t TEXTUAL] [-s STRUCTURE]

options:
  -t TEXTUAL, --textual TEXTUAL
                        path to the textual-alterted dataset
  -s STRUCTURE, --structure STRUCTURE
                        path to the structural-alterted dataset
```

### Analysis

`analysis-calibration.py` will compute the Estimated Calibration Error for all probs log in `probs/`. If you want to see ECE results, make sure to have `-p true` when you evaluate.

### Models

`lm_utils.py` provides inference code for `mistral`, `llama2_7b`, `llama2_13b`, `llama2_70b`, and `chatgpt`. If you want to add new models, add it in both `lm_init()` where you initialize the model and tokenizer; and `llm_response()` where you generate text with it and provide token probabilities (if any).

### Metrics

`metrics.py` provides the implementation of metrics (accuracy, f1, precision, recall) calcualted from `preds`, `labels`. 1 as bot and 0 as human. Feel free to add your metric and include it in the return dictionary.

### Citation

```
@inproceedings{feng-etal-2024-bot,
    title = "What Does the Bot Say? Opportunities and Risks of Large Language Models in Social Media Bot Detection",
    author = "Feng, Shangbin  and
      Wan, Herun  and
      Wang, Ningnan  and
      Tan, Zhaoxuan  and
      Luo, Minnan  and
      Tsvetkov, Yulia",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.196",
    pages = "3580--3601",
    abstract = "Social media bot detection has always been an arms race between advancements in machine learning bot detectors and adversarial bot strategies to evade detection. In this work, we bring the arms race to the next level by investigating the opportunities and risks of state-of-the-art large language models (LLMs) in social bot detection. To investigate the opportunities, we design novel LLM-based bot detectors by proposing a mixture-of-heterogeneous-experts framework to divide and conquer diverse user information modalities. To illuminate the risks, we explore the possibility of LLM-guided manipulation of user textual and structured information to evade detection. Extensive experiments with three LLMs on two datasets demonstrate that instruction tuning on merely 1,000 annotated examples produces specialized LLMs that outperform state-of-the-art baselines by up to 9.1{\%} on both datasets, while LLM-guided manipulation strategies could significantly bring down the performance of existing bot detectors by up to 29.6{\%} and harm the calibration and reliability of bot detection systems.",
}
```
