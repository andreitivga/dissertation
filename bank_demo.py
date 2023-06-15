from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

ckpt = 'checkpoint-1000'
tokenizer = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForCausalLM.from_pretrained(ckpt)

def format_resp(system_resp):
    # format Belief, Action and Response tags
    system_resp = system_resp.replace('<|belief|>', '*Belief State: ')
    system_resp = system_resp.replace('<|action|>', '*Actions: ')
    system_resp = system_resp.replace('<|response|>', '*System Response: ')
    return system_resp

def predict(input, history):

    # print('Input: ', input)

    if history != []:
        # model expects only user and system responses, no belief or action sequences
        # therefore we clean up the history first.

        # history is  a list of token ids which represents all the previous states in the conversation
        # ie. tokenied user inputs + tokenized model outputs
        history_str = tokenizer.decode(history[0])
        
        # print('history_str:::', history_str)

        turns = re.split('<\|system\|>|<\|user\|>', history_str)[1:]

        for i in range(0, len(turns)-1, 2):
            turns[i] = '<|user|>' + turns[i]
            # keep only the response part of each system_out in the history (no belief and action)
            turns[i+1] = '<|system|>' + turns[i+1].split('<|response|>')[1]

        # print(turns)
        history4input = tokenizer.encode(''.join(turns), return_tensors='pt') 
    else:
        history4input = torch.LongTensor(history)

    # format input for model by concatenating <|context|> + history4input + new_input + <|endofcontext|>
    new_user_input_ids = tokenizer.encode(' <|user|> '+input, return_tensors='pt')
    context = tokenizer.encode('<|context|>', return_tensors='pt')
    endofcontext = tokenizer.encode(' <|endofcontext|>', return_tensors='pt')
    model_input = torch.cat([context, history4input, new_user_input_ids, endofcontext], dim=-1)

    # generate output
    out = model.generate(model_input, max_length=1024, eos_token_id=50262).tolist()[0]

    # formatting the history
    # leave out endof... tokens
    string_out = tokenizer.decode(out)

    # print('string_out:::', string_out)
    
    system_out = string_out.split('<|endofcontext|>')[1].replace('<|endofbelief|>', '').replace('<|endofaction|>', '').replace('<|endofresponse|>', '')
    
    # print('system_out:::', system_out)

    start_index = system_out.find('<|response|>')
    end_index = start_index + len('<|response|>')

    resp_lex = system_out[end_index:]

    resp_tokenized = tokenizer.encode(' <|system|> '+system_out, return_tensors='pt')
    history = torch.cat([torch.LongTensor(history), new_user_input_ids, resp_tokenized], dim=-1).tolist()
    # history = history + last user input + <|system|> <|belief|> ... <|action|> ... <|response|>...

    # format responses to print out
    # need to output all of the turns, hence why the history must contain belief + action info 
    # even if we have to take it out of the model input
    turns = tokenizer.decode(history[0])
    turns = re.split('<\|system\|>|<\|user\|>', turns)[1:] # list of all the user and system turns until now
    
    # print('turns:::', turns)

    # list of tuples [(user, system), (user, system)...]
    # 1 tuple represents 1 exchange at 1 turn 
    # system resp is formatted with function above to make more readable
    resps = [(turns[i], format_resp(turns[i+1])) for i in range(0, len(turns)-1, 2)] 

    return resps, history, resp_lex

history = []
resp_lex = ''
while True:
    user_input = input('You: ')
    system_resp, history, resp_lex = predict(user_input, history)
    # print('system_resp:::', system_resp)
    print('System: ', resp_lex)