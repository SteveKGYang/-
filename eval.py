import torch
from collections import Counter
from transformers import BertTokenizer, BertConfig, BertForQuestionAnswering


model = BertForQuestionAnswering.from_pretrained("./final_model_split")
tokenizer = BertTokenizer.from_pretrained("./final_model_split")
config = BertConfig.from_pretrained("./final_model_split")


def f1_score(pred, ref):
    pred_tokens = list(pred)
    ref_tokens = list(ref)
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(ref_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def evaluate(predictions, references):
    f1 = total = 0
    for ref, pred in zip(references, predictions):
        total += 1
        f1 += f1_score(pred, ref)
    f1 = 100.0 * f1 / total
    return f1

batch_size = 1
PAD = 0

with open("data/divided_data/eval.text", "r", encoding="utf8") as f:
    eval_data = [line for line in f]
    results = []
    labels = []
    for exp_no in range(0, len(eval_data), batch_size):
        inputs = []
        words = eval_data[exp_no:exp_no + batch_size]
        attention_masks = []
        token_type_ids = []
        position_ids = []

        for i in range(batch_size):
            if i == len(words):
                break
            if len(words[i].split("|||")) != 2:
                continue
            word, label = words[i].split("|||")
            labels.append(label)
            tokenized_data = tokenizer.tokenize(word)
            if len(tokenized_data) > 512:
                tokenized_data = tokenized_data[:512]
            inpu = tokenizer.convert_tokens_to_ids(tokenized_data)
            for _ in range(len(inpu), 512):
                inpu.append(PAD)

            token_type_id = [0] * 512
            if '[SEP]' not in tokenized_data:
                continue
            x = tokenized_data.index("[SEP]")
            for j in range(x + 1, len(token_type_id)):
                token_type_id[j] = 1

            position_id = [m for m in range(512)]

            attention_mask = [1] * 512
            for k, item in enumerate(inpu):
                if item == PAD:
                    attention_mask[k] = 0

            inputs.append(inpu)
            attention_masks.append(attention_mask)
            token_type_ids.append(token_type_id)
            position_ids.append(position_id)

        inputs = torch.tensor(inputs)
        attention_masks = torch.tensor(attention_masks)
        position_ids = torch.tensor(position_ids)
        token_type_ids = torch.tensor(token_type_ids)

        if True:
            model.cuda()
            attention_masks = attention_masks.cuda()
            position_ids = position_ids.cuda()
            inputs = inputs.cuda()
            token_type_ids = token_type_ids.cuda()

        if len(inputs.size()) != 2:
            print(inputs.size())
            continue

        starts_score, end_score = model(inputs, attention_mask=attention_masks, position_ids=position_ids,
                                        token_type_ids=token_type_ids)

        for starts, ends, token_type_id in zip(starts_score, end_score, token_type_ids):
            max_score = -100000
            count1 = 0
            best_start = 0
            best_end = 0
            for ty, start, end in zip(token_type_id, starts, ends):
                if ty == 1:
                    if count1 < len(end_score) - 6:
                        for i in range(1, 7):
                            m = count1 + i
                            k = start + ends[m]
                            if k > max_score:
                                max_score = k
                                best_start = count1
                                best_end = m
                count1 += 1
            pre = ''.join(tokenized_data[best_start:best_end + 1])
            results.append(pre)

    f1 = evaluate(results, labels)
    print(f1)