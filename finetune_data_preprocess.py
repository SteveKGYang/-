import json
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("./pretrained_bert")

count = 0
ft = open("./finetune_data/text.trial", "w+")

with open("./cmrc2018-master/squad-style-data/cmrc2018_trial.json","r") as f:
    a = json.load(f)
    d = a['data']
    for i in d:
      z = i['paragraphs'][0]
      context = z['context'].strip()
      '''print(z['qas'][0])
      print(z['qas'][1])
      print(z['qas'][2])'''
      for record in z['qas']:
         question = record['question']
         answer = record['answers'][0]['text']
         ques = '[CLS]' + question + '[SEP]'

         tokenized_ques = tokenizer.tokenize(ques)
         final_line = ques + context
         w = tokenizer.tokenize(final_line)
         tokenized_context = tokenizer.tokenize(context)
         if len(w) <= 512:
            ft.write(final_line+"|||"+answer.strip()+"\n")
         else:
            if 400<len(tokenized_context)<=700:
               final_line1 = ''.join(tokenized_ques + tokenized_context[:401]) + "|||" + answer.strip() + "\n"
               final_line2 = ''.join(tokenized_ques + tokenized_context[300:]) + "|||" + answer.strip() + "\n"
               ft.write(final_line1)
               ft.write(final_line2)
            if 700<len(tokenized_context):
               final_line1 = ''.join(tokenized_ques + tokenized_context[:401]) + "|||" + answer.strip() + "\n"
               final_line2 = ''.join(tokenized_ques + tokenized_context[300:701]) + "|||" + answer.strip() + "\n"
               final_line3 = ''.join(tokenized_ques + tokenized_context[600:]) + "|||" + answer.strip() + "\n"
               ft.write(final_line1)
               ft.write(final_line2)
               ft.write(final_line3)
               
         count += 1
         if count % 1000 == 0:
               print("{} records processed.".format(count))

ft.close()
