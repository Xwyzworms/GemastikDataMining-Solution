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

def extract_answer(passage, seq_label):
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
      return start_answer, end_answer
    return start_answer, start_answer
  else:
    list_answers = []
    b_count = seq_label.count("B")
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
      elif ( abs(answer_tuple[indx] - answer_tuple[indx + 1])) == 2:
        b_end.append(answer_tuple[indx])
        list_answers.append((b_start,b_end))
        b_start = answer_tuple[indx + 1]
        b_end = []
    
    returned = []
    for tup in range(len(list_answers)):
      start = list_answers[tup][0]
      end = list_answers[tup][1][-1]
      returned.append((start,end))
    print(returned)

def create_passage_and_answer(passage,seq_label):
  sentence_not_splitted =  makeTheSentence(passage,False)
  sentence_splitted = makeTheSentence(passage,True)
  start_answer, end_answer =extract_answer(passage, seq_label)
  answer = sentence_splitted[start_answer : end_answer + 1]
  print(answer)
  start = sentence_not_splitted.index(answer[0])
  end = sentence_not_splitted.index(answer[-1])
  end_len = len(answer[-1])
  end = end + end_len
  return start, end

#print(create_passage_and_answer(df_train.loc[10,"passage"],df_train.loc[10,"seq_label"]))
#print(makeTheSentence(df_train.loc[2,"passage"]))
extract_answer(df_train.loc[156,"passage"],df_train.loc[156,"seq_label"])


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