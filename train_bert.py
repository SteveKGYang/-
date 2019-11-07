import torch
import os
import time
from torch.nn.utils import clip_grad_norm_
import numpy as np
import argparse
from collections import Counter

from transformers import AdamW, WarmupLinearSchedule
from transformers import BertTokenizer, BertConfig, BertForQuestionAnswering

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

PAD = 0
CLS = 101
SEP = 102

parser = argparse.ArgumentParser(description='Utility Functions')

parser.add_argument('-gpu', action="store_true",
					help="""use cuda""")
parser.add_argument('-dataset', type=str, default=None,
					help="path to input")
parser.add_argument('-eval_dataset', type=str, default=None,
					help="path to eval input")
parser.add_argument('-save_path', type=str, default=None,
					help="path to input")
parser.add_argument('-mode', type=str, default=None,
					choices=["train", "test"],
					help="path to input")
parser.add_argument('-batch_size', type=int, default=None,
					help="path to input")
parser.add_argument('-epoch', type=int, default=None,
					help="path to file containing generated summaries")
parser.add_argument('-lr', type=float, default=None,
					help="prefix of .dict and .labels files")
parser.add_argument('-max_grad_norm', type=float, default=None,
					help="prefix of .dict and .labels files")
parser.add_argument("--weight_decay", default=0.0, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
args = parser.parse_args()

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

def train(args, model, tokenizer):

	no_decay = ['bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
		 'weight_decay': args.weight_decay},
		{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
	]
	optim = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
	scheduler = WarmupLinearSchedule(optim, warmup_steps=8442, t_total=168840)
	t = open(args.dataset, "r", encoding="utf8")
	e = open(args.eval_dataset, "r", encoding="utf8")
	train_data = [line for line in t]
	eval_data = [line for line in e]

	tr_loss = 0.0
	global_step = 0

	for epoch_no in range(args.epoch):
		print("training for epoch {}".format(epoch_no))
		start_time = time.time()
		model.train()
		for exp_no in range(0, len(train_data), args.batch_size):
			inputs = []
			words = train_data[exp_no:exp_no+args.batch_size]
			attention_masks = []
			token_type_ids = []
			position_ids = []
			start_positions = []
			end_positions = []

			for i in range(args.batch_size):
				if i == len(words):
					break
				if len(words[i].split("|||")) != 2:
					continue
				word, label = words[i].split("|||")
				tokenized_data = tokenizer.tokenize(word)
				if len(tokenized_data) > 512:
					tokenized_data = tokenized_data[:512]
				inpu = tokenizer.convert_tokens_to_ids(tokenized_data)
				for _ in range(len(inpu), 512):
					inpu.append(PAD)

				token_type_id = [0] * 512
				#print(len(token_type_id))
				if '[SEP]' not in tokenized_data:
					continue
				x = tokenized_data.index("[SEP]")
				for j in range(x + 1, len(token_type_id)):
					token_type_id[j] = 1

				position_id = [m for m in range(512)]

				attention_mask = [1]*512
				for k, item in enumerate(inpu):
					if item == PAD:
						attention_mask[k] = 0

				start = 0
				end = 0
				if label.strip() in word:
					tokenized_label = tokenizer.tokenize(label)
					for num, token in enumerate(tokenized_data):
						if token == tokenized_label[0]:
							right = True
							for w in range(1, len(tokenized_label)):
								if not tokenized_label[w] == tokenized_data[min(num+w, len(tokenized_data)-1)]:
									right = False
									break
							if right:
								start = num
								end = num + len(tokenized_label) - 1

				inputs.append(inpu)
				attention_masks.append(attention_mask)
				token_type_ids.append(token_type_id)
				position_ids.append(position_id)
				start_positions.append(start)
				end_positions.append(end)
			

			inputs = torch.tensor(inputs)
			attention_masks = torch.tensor(attention_masks)
			token_type_ids = torch.tensor(token_type_ids)
			position_ids = torch.tensor(position_ids)
			start_positions = torch.tensor(start_positions)
			end_positions = torch.tensor(end_positions)
			
			#start_positions = start_positions.unsqueeze(0)
			#end_positions = end_positions.unsqueeze(0)

			if args.gpu:
				model.cuda()
				attention_masks = attention_masks.cuda()
				position_ids = position_ids.cuda()
				token_type_ids = token_type_ids.cuda()
				inputs = inputs.cuda()
				start_positions = start_positions.cuda()
				end_positions = end_positions.cuda()
			
			#print(inputs.size())
			#print(start_positions.size())
			#print(end_positions.size())
			
			if len(inputs.size()) != 2:
				print(inputs.size())
				continue

			output = model(inputs,attention_mask=attention_masks,position_ids=position_ids,
                    token_type_ids=token_type_ids, start_positions=start_positions, end_positions=end_positions)
			loss = output[0]
			# training
			optim.zero_grad()
			loss.backward()
			clip_grad_norm_(model.parameters(), args.max_grad_norm)
			tr_loss += loss.item()
			optim.step()
			scheduler.step()
			global_step += 1

			if global_step % 10000 == 0 :
				output_dir = os.path.join(args.save_path, 'checkpoint-{}'.format(global_step))
				if not os.path.exists(output_dir):
					os.makedirs(output_dir)
				model_to_save = model.module if hasattr(model, 'module') else model
				model_to_save.save_pretrained(output_dir)
				torch.save(args, os.path.join(output_dir, 'training_args.bin'))
		
		model.eval()
		results = []
		labels = []
		for exp_no in range(0, len(eval_data), args.batch_size):
			inputs = []
			words = eval_data[exp_no:exp_no + args.batch_size]
			attention_masks = []
			token_type_ids = []
			position_ids = []

			for i in range(args.batch_size):
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

			if args.gpu:
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
		print("Evaluate F1 score for epoch {} is {}.".format(epoch_no, f1))

		end_time = time.time()
		print("Epoch {} runs {} time.".format(epoch_no, (end_time-start_time)))

	e.close()
	t.close()
	return global_step, tr_loss/global_step

def main():

	model = BertForQuestionAnswering.from_pretrained("./finetune_model/add_position")
	tokenizer = BertTokenizer.from_pretrained("./finetune_model/add_position")
	config = BertConfig.from_pretrained("./finetune_model/add_position")


	if args.mode == "train":
		global_step, tr_loss = train(args, model, tokenizer)
		print(" global_step = {}, average loss = {}".format(global_step, tr_loss))

	if args.mode == 'train':
		if not os.path.exists(args.save_path):
			os.makedirs(args.save_path)
		print("Saving model checkpoint to {}".format(args.save_path))
		model_to_save = model.module if hasattr(model, 'module') else model
		model_to_save.save_pretrained(args.save_path)
		tokenizer.save_pretrained(args.save_path)

		torch.save(args, os.path.join(args.save_path, 'training_args.bin'))


if __name__ == "__main__":
	main()
