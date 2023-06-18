from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

ckpt = 'checkpoint-9000'
tokenizer = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForCausalLM.from_pretrained(ckpt)

context_token = tokenizer.encode('<|context|>', return_tensors='pt')
endofcontext_token = tokenizer.encode(' <|endofcontext|>', return_tensors='pt')

def generate_response(input, history):
    if history == []:
        context_tokenized = torch.LongTensor(history)
    else:
        history_str = tokenizer.decode(history[0])
        turns = re.split('<\|system\|>|<\|user\|>', history_str)[1:]

        for i in range(0, len(turns)-1, 2):
            turns[i] = '<|user|>' + turns[i]
            turns[i+1] = '<|system|>' + turns[i+1]

        context_tokenized = tokenizer.encode(''.join(turns), return_tensors='pt') 

    user_input_tokenized = tokenizer.encode(' <|user|> '+ input, return_tensors='pt')

    model_input = torch.cat([context_token, context_tokenized, user_input_tokenized, endofcontext_token], dim=-1)
    attention_mask = torch.ones_like(model_input)

    out_tokenized = model.generate(model_input, max_length=1024, eos_token_id=50262, pad_token_id=50262, attention_mask=attention_mask).tolist()[0]
    out_str = tokenizer.decode(out_tokenized)

    generated_substring = out_str.split('<|endofcontext|>')[1] #belief, actions, system_response

    beliefs_start_index = generated_substring.find('<|belief|>') + len('<|belief|>')
    beliefs_end_index = generated_substring.find('<|endofbelief|>', beliefs_start_index)

    actions_start_index = generated_substring.find('<|action|>') + len('<|action|>')
    actions_end_index = generated_substring.find('<|endofaction|>', actions_start_index)

    response_start_index = generated_substring.find('<|response|>') + len('<|response|>')
    response_end_index = generated_substring.find('<|endofresponse|>', response_start_index)

    beliefs_str = generated_substring[beliefs_start_index:beliefs_end_index]
    actions_str = generated_substring[actions_start_index:actions_end_index]
    system_response_str = generated_substring[response_start_index:response_end_index]

    system_resp_tokenized = tokenizer.encode(' <|system|> '+ system_response_str, return_tensors='pt')
    history = torch.cat([torch.LongTensor(history), user_input_tokenized, system_resp_tokenized], dim=-1).tolist()

    return history, beliefs_str, actions_str, system_response_str

history = []
while True:
    print('----------------------------------------------------------------------------')
    user_input = input('You: ')

    if user_input == 'quit' or  user_input == 'q':
        break
    print()

    history, beliefs, actions, system_response = generate_response(user_input, history)

    if beliefs == '  ':
        print('\tNo belief could be extracted from this exact user input.')
    else:
        print('\tBeliefs:', beliefs)

    print('\tActions:', actions)

    print()
    print('System:', system_response)