# %%
!pip install transformers datasets pytorch

# %%
!pip install datasets

# %%
from google.colab import drive
drive.mount("/content/gdrive")

# %%
import os
import pandas as pd
import re

# %%
DATASET_PATH =  "/content/gdrive/MyDrive/Gemastik FINAL Yok Bisa Yok/FinalGemastikSenin/data"


# %%
DATASET_TRAIN = os.path.join(DATASET_PATH,"train_preprocess.csv")
DATASET_VALID = os.path.join(DATASET_PATH, "valid_preprocess.csv")

# %%
df_train = pd.read_csv(DATASET_TRAIN)
df_valid = pd.read_csv(DATASET_VALID)

# %%
df_train

# %%
import numpy as np
def makeTheSentence(sentence_list,return_list = False):
  all_word = []
  sentence_list = sentence_list.replace(']', "")
  sentence_list = sentence_list.replace("[","")
  sentence_list = sentence_list.replace("','", "<TOKEN>")
  for word in sentence_list.split(","):

    if (word !=  ""):
      word = str(word)
      word = re.sub("'","",word )
      word = re.sub("'","",word )
      word = word.rstrip()
      word = re.sub("'","",word )
      word = word.rstrip()
      all_word.append(word)
    temp_string = "".join(all_word)
    temp_string = re.sub("<TOKEN>", ",", temp_string)
  if return_list:
    return temp_string.split(" ")
  return "".join(temp_string)

def create_question(question):
  return makeTheSentence(question)
def extract_answer_used(passage, seq_label):
  start_answer = -1
  end_answer = -1
  index = 0
  print(seq_label)
  seq_label = makeTheSentence(seq_label).split(" ")
  if(seq_label.count("B") == 1):
    for word in range(seq_label.__len__() - 1):
      if(seq_label[word] == "B"):
        start_answer = word
      elif(seq_label[-1] == "B" ):
        start_answer = word
      elif((seq_label[word]== "I" and seq_label[word + 1]) == "O" or (seq_label[word] =="I" and seq_label[-1] == "I")):
        end_answer = word
      index += 1
    if (end_answer != -1):
      return start_answer, end_answer
    return start_answer, start_answer
  else:
    list_answers = []
    b_count = seq_label.count("B")
    print("b count{}".format(b_count))
    answer_tuple = []
    for word in seq_label:
      if word == "B":
        answer_tuple.append(index)
      elif word == "I" :
        answer_tuple.append(index)
      index+=1
    print(answer_tuple)
    return answer_tuple   


def extract_answer_not_used(passage, seq_label):
  start_answer = -1
  end_answer = -1
  index = 0
  print(seq_label)
  seq_label = makeTheSentence(seq_label).split(" ")
  if(seq_label.count("B") == 1):
    for word in seq_label:
      if(word == "B"):
        start_answer = index
      elif((word == "I" and seq_label[index + 1]) == "O" or (word =="I" and seq_label[-1] == "I")):
        end_answer = index
      index += 1
    if (end_answer != -1):
      return [(start_answer, end_answer)]
    return [(start_answer, start_answer)]
  else:
    list_answers = []
    b_count = seq_label.count("B")
    print("b count{}".format(b_count))
    answer_tuple = []
    for word in seq_label:
      if word == "B":
        answer_tuple.append(index)
      elif word == "I" :
        answer_tuple.append(index)
      index+=1
    b_start = answer_tuple[0]
    b_end = []
    for indx in range(len(answer_tuple)-1):
      if ( abs(answer_tuple[indx] - answer_tuple[indx + 1])) == 1:
        b_end.append(answer_tuple[indx])
      elif ( abs(answer_tuple[indx] - answer_tuple[indx + 1])) != 1:
        b_end.append(answer_tuple[indx])
        list_answers.append((b_start,b_end))
        b_start = answer_tuple[indx + 1]
        b_end = []
    list_answers.append((answer_tuple[indx+1],[answer_tuple[-1]]))
    returned = []
    for tup in range(len(list_answers)):
      print(tup)
      start = list_answers[tup][0]
      end = list_answers[tup][1][-1]
      returned.append((start,end))

    return returned
def create_passage_and_answer_not_used(passage,seq_label):
  sentence_not_splitted =  makeTheSentence(passage,False)
  sentence_splitted = makeTheSentence(passage,True)
  list_answers = extract_answer_used(passage, seq_label)
  returned_text = []
  if (len(list_answers) == 1):
    start_answer, end_answer = list_answers
    answer = sentence_splitted[start_answer : end_answer + 1]
    start = sentence_not_splitted.index(answer[0])
    end = sentence_not_splitted.index(answer[-1])
    end_len = len(answer[-1])
    end = end + end_len
    returned_text = sentence_not_splitted[start:end].split(" ")
  else:
    for tupleIndex in range(list_answers.__len__()):
      temp_final_text = ""
      start_answer, end_answer = list_answers[tupleIndex]
      answer_temp = sentence_splitted[start_answer : end_answer + 1]
      start = sentence_not_splitted.index(answer_temp[0])
      start,end,end_len = sentence_not_splitted.index(answer_temp[0]),sentence_not_splitted.index(answer_temp[-1]) ,len(answer_temp[-1])
      end  = end + end_len
      temp_final_text = sentence_not_splitted[start:end]
      returned_text.append(temp_final_text)
  return " ".join(returned_text)



def create_passage_and_answer(passage,seq_label):
  sentence_not_splitted =  makeTheSentence(passage,False)
  sentence_splitted = makeTheSentence(passage,True)
  list_answers = extract_answer_used(passage, seq_label)
  returned_text = []
  start_answer, end_answer = list_answers[0],list_answers[-1]
  answer = sentence_splitted[start_answer : end_answer + 1]
  start = sentence_not_splitted.index(answer[0])
  end = sentence_not_splitted.index(answer[-1])
  end_len = len(answer[-1])
  end = end + end_len
  returned_text = sentence_not_splitted[start:end].split(" ")
  print(returned_text)
  return start, end

print(create_passage_and_answer(df_train.loc[17,"passage"],df_train.loc[17,"seq_label"]))
#print(makeTheSentence(df_train.loc[2,"passage"]))
#extract_answer(df_train.loc[16,"passage"],df_train.loc[16,"seq_label"])

# %%
def validate_answer(df, location):
  passage = df.loc[location,"passage"]
  question = df.loc[location,"question"]
  seq_label = df.loc[location,"seq_label"]
  print(passage)
  print(question)
  print(seq_label)
  start,end = create_passage_and_answer(passage,seq_label)
  print(makeTheSentence(df.loc[location,"passage"])[start:end])
validate_answer(df_train,6)

# %%
range(len(df_train))

# %%
df_train["start"] = -1
df_train["end"] = -1
df_valid["start"] = -1
df_valid["end"] = -1
df_valid["index"] = range(len(df_valid))
df_train["index"] = range(len(df_train))
df_copy = df_train.copy()


list_ofAnswer = []
i = 0
def convert_toDictAnswer(index,passage, question, seq_label):
  sentence = makeTheSentence(passage)
  sentence_not_split = makeTheSentence(passage, True)
  question = makeTheSentence(question)
  
  start,end = create_passage_and_answer(passage,seq_label)
  return sentence,question, start, end

train_ready = df_copy[["passage","question","seq_label","index"]].apply(lambda x: convert_toDictAnswer(x["index"],x["passage"],x["question"], x["seq_label"]),axis=1)
valid_ready = df_valid[["passage","question","seq_label","index"]].apply(lambda x: convert_toDictAnswer(x["index"],x["passage"],x["question"], x["seq_label"]),axis=1)

# %%
df_train_ready = pd.DataFrame([],columns=["sentence","question","start_position","end_position"])
df_valid_ready = pd.DataFrame([],columns=['sentence', "question", "start_position", "end_position"])
for i in range(train_ready.__len__()):
  sentence,question,start,end = train_ready[i][0], train_ready[i][1], train_ready[i][2], train_ready[i][3]
  new_row = pd.Series(data={'sentence':sentence, 'question':question, 'start_position':start, "end_position":end})
  df_train_ready = df_train_ready.append(new_row, ignore_index=True)

for i in range(valid_ready.__len__()):
  sentence,question,start,end = valid_ready[i][0], valid_ready[i][1], valid_ready[i][2], valid_ready[i][3]
  new_row = pd.Series(data={'sentence':sentence, 'question':question, 'start_position':start, "end_position":end})
  df_valid_ready = df_valid_ready.append(new_row, ignore_index=True)

# %%
import datasets

# %%
df_dict = datasets.Dataset.from_pandas(df_train_ready)
df_valid_dict = datasets.Dataset.from_pandas(df_valid_ready)

# %%
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
tokenizer = AutoTokenizer.from_pretrained("Wikidepia/indobert-lite-squad")
model = AutoModelForQuestionAnswering.from_pretrained("Wikidepia/indobert-lite-squad")

# %%
def process_train_read(train_ready):
	tokenized_data = tokenizer(train_ready["sentence"], train_ready["question"], truncation="only_first", padding="max_length")
	input_ids  = tokenized_data["input_ids"]


	cls_index = input_ids.index(tokenizer.cls_token_id) ## CConst for CLS token Unknown word

	if train_ready["sentence"]:
		start_position =cls_index
		end_position = cls_index
	else:
		answer = train_ready["sentence"][train_ready["start"]:train_ready["end"]]
		start_position = train_ready["start"]
		end_position = train_ready["end"]

		start_token = tokenized_data.char_to_token(start_position)
		end_token = tokenized_data.char_to_token(end_position)

		if start_token is None:
			start_token = tokenizer.max_length
		if end_token is None:
			end_token = tokenizer.max_length; start_position = start_token;end_position = end_token
	return {
		"input_ids" : tokenized_data["input_ids"],
		"attention_mask" : tokenized_data["attention_mask"],
		"start_positions": start_position,
		"end_positions" : end_position
	}
processed_data = df_dict.map( process_train_read)
processed_eval_data = df_valid_dict.map(process_train_read)

# %%
thecols = ["input_ids", "attention_mask","start_positions","end_positions"]
processed_data.set_format(type="pt", columns=thecols)
processed_eval_data.set_format(type="pt", columns=thecols)

# %%
from sklearn.metrics import f1_score

def compute_f1_metrics(pred):    
    start_labels = pred.label_ids[0]
    start_preds = pred.predictions[0].argmax(-1)
    end_labels = pred.label_ids[1]
    end_preds = pred.predictions[1].argmax(-1)
    
    f1_start = f1_score(start_labels, start_preds, average='macro')
    f1_end = f1_score(end_labels, end_preds, average='macro')
    
    return {
        'f1_start': f1_start,
        'f1_end': f1_end,
    }

# %%
from transformers import Trainer, TrainingArguments
# EZPZ Gemastik Kalau udah paham Cuy
training_args = TrainingArguments(
    output_dir='model_results5',          # output directory
    overwrite_output_dir=True,
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=8,   # batch size for evaluation
    warmup_steps=20,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir=None,            # directory for storing logs
    logging_steps=50
)

trainer = Trainer(
    model=model,
    args=training_args, # training arguments, defined above
    train_dataset=processed_data, # training dataset
    eval_dataset=processed_eval_data, # evaluation dataset
    compute_metrics=compute_f1_metrics             
)

trainer.train()


