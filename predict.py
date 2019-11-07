import torch
from transformers import BertTokenizer, BertModel, BertForQuestionAnswering

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tokenizer = BertTokenizer.from_pretrained('./final_model_whole')
model = BertForQuestionAnswering.from_pretrained('./final_model_whole')

count = 0

sens = []

fo = open("./data/evaluate/test_predict_whole.answer", "w+", encoding="utf8")
with open("./data/train data/processed_test.doc_query", "r", encoding="utf8") as f:
    results = []
    for line in f:
        sens.append(line.strip())
    with open("./data/train data/train.answer", "r", encoding="utf8") as fl:
        '''for line in fl:
            labs.append(line.split("|||")[1].strip())'''
        for sen in sens:
            count += 1
            data, num = sen.split("|||")
            #label = labs[int(num.strip())]
            tokenized_data = tokenizer.tokenize(data)
            if len(tokenized_data) > 512:
                tokenized_data = tokenized_data[:512]
            token_type_id = [0]*len(tokenized_data)
            x = tokenized_data.index("[SEP]")
            for i in range(x+1, len(token_type_id)):
                token_type_id[i] = 1
            token_type_id = torch.tensor(token_type_id)
            position_ids = torch.tensor([i for i in range(len(tokenized_data))])
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_data)
            tokens_tensor = torch.tensor([indexed_tokens])
            #input_ids = torch.tensor(tokenizer.encode(tokenized_data)).unsqueeze(0)
            model.cuda()
            tokens_tensor = tokens_tensor.cuda()
            token_type_id = token_type_id.cuda()
            position_ids = position_ids.cuda()
            outputs = model(tokens_tensor, token_type_ids=token_type_id, position_ids=position_ids)

            start_score = outputs[0].squeeze(0)
            end_score = outputs[1].squeeze(0)

            max_score = -100000
            count1 = 0
            best_start = 0
            best_end = 0
            for ty, start in zip(token_type_id, start_score):
                if ty == 1:
                    if count1 < len(end_score) - 2:
                        for i in range(1, 3):
                            m = count1+i
                            k = start + end_score[m]
                            if k > max_score:
                                max_score = start + end_score[count1+i]
                                best_start = count1
                                best_end = count1 + i
                count1 += 1
            pre = tokenized_data[best_start:best_end+1]
            result = ''.join(pre) + "|||"
            if len(results) < int(num) + 1:
                results.append([''.join(pre) + "|||" + str(max_score)])
            else:
                results[int(num)].append(''.join(pre) + "|||" + str(max_score))
            if count % 200 == 0:
                print("{} lines processed.".format(count))

        for i, answers in enumerate(results):
            if len(answers) == 1:
                fo.write("<qid_"+str(i)+"> ||| "+answers[0].split("|||")[0]+"\n")
            else:
                k = 0
                word = []
                score = []
                for item in answers:
                    word.append(item.split("|||")[0])
                    score.append(item.split("|||")[1])
                if score[0] > score[1]:
                    k = 0
                else:
                    k = 1
                fo.write("<qid_" + str(i) + "> ||| " + answers[k].split("|||")[0] + "\n")
fo.close()
