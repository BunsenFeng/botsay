from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
import transformers
import torch
import openai
import os
import time
import numpy as np
import random
import metrics
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

def llm_init(model_name):
    global device
    global model
    global tokenizer
    global pipeline

    if model_name == "mistral":
        
        device = "cuda"
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        model.to(device)

    if model_name == "llama2_70b":
        device = "cuda"
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-chat-hf", device_map="auto", torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf")

    if model_name == "llama2_7b":
        device = "cuda"
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        model.to(device)

    if model_name == "llama2_13b":
        device = "cuda"
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", device_map="auto", torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
    
    if model_name == "chatgpt":
        openai.api_key = os.getenv("OPENAI_API_KEY")

def wipe_model():
    global device
    global model
    global tokenizer
    global pipeline
    device = None
    model = None
    tokenizer = None
    pipeline = None
    del device
    del model
    del tokenizer
    del pipeline

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(10))
def llm_response(prompt, model_name, probs = False, temperature = 0.1, max_new_tokens = 200):
    if model_name == "mistral":
        messages = [
        {"role": "user", "content": prompt},
        ]

        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

        model_inputs = encodeds.to(device)

        outputs = model.generate(model_inputs, max_new_tokens=max_new_tokens, do_sample=True, return_dict_in_generate=True, output_scores=True, temperature = temperature, pad_token_id=tokenizer.eos_token_id)
        transition_scores = model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        input_length = encodeds.shape[1]
        generated_ids = outputs.sequences[:, input_length:]

        token_probs = {}
        if probs:
            for tok, score in zip(generated_ids[0], transition_scores[0]):
                token_probs[tokenizer.decode(tok)] = np.exp(score.item())

        decoded = tokenizer.batch_decode(generated_ids)
        if probs:
            return decoded[0], token_probs
        else:
            return decoded[0]

    elif "llama2" in model_name:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        outputs = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=True, return_dict_in_generate=True, output_scores=True, temperature = temperature, pad_token_id=tokenizer.eos_token_id)

        transition_scores = model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )

        input_length = input_ids.shape[1]
        generated_ids = outputs.sequences[:, input_length:]

        token_probs = {}
        if probs:
            for tok, score in zip(generated_ids[0], transition_scores[0]):
                token_probs[tokenizer.decode(tok)] = np.exp(score.item())

        decoded = tokenizer.batch_decode(generated_ids)
        if probs:
            return decoded[0], token_probs
        else:
            return decoded[0]
    
    if model_name == "chatgpt":
        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_new_tokens,
            logprobs=1,
        )
        time.sleep(0.1)
        token_probs = {}
        for tok, score in zip(response.choices[0].logprobs.tokens, response.choices[0].logprobs.token_logprobs):
            token_probs[tok] = np.exp(score)
        if probs:
            return response.choices[0].text, token_probs
        else:
            return response.choices[0].text

def answer_parsing(response):
    # mode 1: answer in the first token
    if "human" in response.strip().split(" ")[0].lower():
        return 0
    elif "bot" in response.strip().split(" ")[0].lower():
        return 1
    
    # mode 2: answer in the second token
    try:
        if "human" in response.strip().split(" ")[1].lower():
            return 0
        elif "bot" in response.strip().split(" ")[1].lower():
            return 1
    except:
        pass

    # mode 3: answer in the first line
    if "human" in response.strip().split("\n")[0].lower():
        return 0
    elif "bot" in response.strip().split("\n")[0].lower():
        return 1
    
    # fail to parse label
    print("fail to parse label", response, "------------------")
    return random.choice([0, 1])

# prompt = "Question: Who is the 44th president of the United States?\nProposed Answer:Donald Trump\nIs this answer correct? (True/False)\nAnswer:"

# llm_init("mistral")
# print(llm_response(prompt, "mistral", probs=True))

# llm_init("llama2_70b")
# print(llm_response(prompt, "llama2_70b", probs=True))

# llm_init("chatgpt")
# print(llm_response(prompt, "chatgpt", probs=True))

# a = torch.randn(3,5)
# target = torch.tensor([0, 2, 4])
# loss = torch.nn.CrossEntropyLoss()
# output = loss(a, target)
# print(output.item())

# a = torch.tensor([0.1, 0.2, 0.3])
# print(torch.nn.functional.log_softmax(a, dim=0))
# print(torch.nn.functional.softmax(a, dim=0))

# prompt =  "The following task focuses on evaluating whether a Twitter user is a bot or human with the help of several labeled examples. You should output the label first and explanation after.\n" + \
# "Screen Name: BunsenFeng, Follower Count: 100, Following Count: 200, Tweet Count: 300, Verified: False\nLabel: human\n" + \
# "Screen Name: yuliatsvetkova, Follower Count: 1000, Following Count: 2000, Tweet Count: 3000, Verified: False\nLabel: human\n" + \
# "Screen Name: dizlight, Follower Count: 1034, Following Count: 2069, Tweet Count: 30001, Verified: False\nLabel: bot\n" + \
# "Screen Name: 1stDibs, Follower Count: 376, Following Count: 274, Tweet Count: 3058, Verified: False\nLabel: bot\n" + \
# "Screen Name: aoisfghkadfh, Follower Count: 1376, Following Count: 1274, Tweet Count: 13058, Verified: False\nLabel:"

# llm_init("mistral")
# print(llm_response(prompt, "mistral", probs=True))

# llm_init("llama2_70b")
# print(llm_response(prompt, "llama2_70b", probs=True))

text_classifier = None
tokenizer = None

def mlm_text_classifier(texts, labels, test_texts, test_labels, EPOCHS=10, BATCH_SIZE=32, LR=5e-5):
    # train a roberta-base model to classify texts
    # texts: a list of strings
    # labels: a list of labels of 0 or 1

    # load model
    global text_classifier, tokenizer
    if not "text_classifier" in os.listdir():
        text_classifier = AutoModelForSequenceClassification.from_pretrained("roberta-base")
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    if "text_classifier" in os.listdir():
        text_classifier = AutoModelForSequenceClassification.from_pretrained("text_classifier/")
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        print("text_classifier loaded")
        return

    # tokenize
    encodeds = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = encodeds["input_ids"]
    attention_mask = encodeds["attention_mask"]

    # train
    optimizer = torch.optim.Adam(text_classifier.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    batch_size = BATCH_SIZE
    for epoch in tqdm(range(EPOCHS)):
        for i in tqdm(range(0, len(input_ids), batch_size)):
            optimizer.zero_grad()
            outputs = text_classifier(input_ids[i:i+batch_size], attention_mask=attention_mask[i:i+batch_size])
            logits = outputs.logits
            loss = loss_fn(logits, torch.tensor(labels[i:i+batch_size]))
            loss.backward()
            optimizer.step()

    # save model
    text_classifier.save_pretrained("text_classifier/")

    # test
    text_classifier.eval()
    golds = test_labels
    preds = []
    for txt in test_texts:
        encodeds = tokenizer(txt, return_tensors="pt", padding=True, truncation=True)
        input_ids = encodeds["input_ids"]
        attention_mask = encodeds["attention_mask"]
        outputs = text_classifier(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        preds.append(predictions[0].item())

    # print(metrics.compute_metrics(preds, golds))

    # test
    # text_classifier.eval()
    # outputs = text_classifier(input_ids, attention_mask=attention_mask)
    # logits = outputs.logits
    # predictions = torch.argmax(logits, dim=1)
    # print(predictions)
    # print(labels)
    # print(logits)

def text_classifier_inference(text):
    # provide predicted labels and probability
    # text: a string
    # return: label, probability
    global text_classifier, tokenizer

    assert text_classifier is not None, "text_classifier is not initialized"
    # tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    text_classifier.eval()
    encodeds = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = encodeds["input_ids"]
    attention_mask = encodeds["attention_mask"]
    outputs = text_classifier(input_ids, attention_mask=attention_mask)
    # print(outputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1)
    probs = torch.nn.functional.softmax(logits, dim=1)
    return predictions[0].item(), probs[0][predictions[0]].item() # label, probability for the predicted label

# texts = ["I like this movie", "I hate this movie", "I like this movie", "I hate this movie"] * 100
# labels = [1, 0, 1, 0] * 100
# mlm_text_classifier(texts, labels, texts, labels)
# print(text_classifier_inference("I like this movie"))
# print(text_classifier_inference("I hate this movie"))