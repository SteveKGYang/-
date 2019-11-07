#coding=utf-8

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("./final_model_whole")

splited = 0
no_splited = 0
count = 0
fi = open("./data/train data/processed_test.doc_query", "w+", encoding='utf8')
with open("test.doc_query", "r", encoding="utf8") as f:
    record = []
    for line in f:
        seg = line.split("|||")
        assert len(seg) == 2
        if seg[0].strip().isdigit():
            record.append(seg[1].strip())
        else:
            doc = ''.join(record)
            final_line = '[CLS]' + seg[1].strip() + '[SEP]' + doc
            w = tokenizer.tokenize(final_line)
            final_line = ''.join(w)
            if len(w) > 512:
                splited += 1
                tokenized_head = tokenizer.tokenize('[CLS]' + seg[1].strip() + '[SEP]')
                tokenized_doc = tokenizer.tokenize(doc)
                record1 = tokenized_doc[:470-len(tokenized_head)]
                record2 = tokenized_doc[(len(tokenized_doc)-(470-len(tokenized_head))):]

                final_line1 = ''.join(tokenized_head + record1) + "|||" + str(count)
                final_line2 = ''.join(tokenized_head + record2) + "|||" + str(count)

                fi.write(final_line1 + "\n")
                fi.write(final_line2 + "\n")

            else:
                no_splited += 1
                final_line += ("|||"+str(count))
                fi.write(final_line + "\n")
            record.clear()
            count += 1
            if count % 1000 == 0:
                print("{} records processed. {} splited, {} not splited".format(count, splited, no_splited))
fi.close()
