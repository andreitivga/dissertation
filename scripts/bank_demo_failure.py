from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

ckpt = 'output_model_gpt2_100epoch'
tokenizer = AutoTokenizer.from_pretrained(ckpt, padding_side='left', use_safetensors=True)
model = AutoModelForCausalLM.from_pretrained(ckpt, use_safetensors=True)

context_token = tokenizer.encode('<|context|>', return_tensors='pt')
endofcontext_token = tokenizer.encode(' <|endofcontext|>', return_tensors='pt')

def extract_slot_or_value(text, target):
    start_index = text.find(target)
    start_index += len(target)
    return text[start_index:].strip()

def generate_response(input, history, d):
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

    out_tokenized = model.generate(model_input, max_length=1024, eos_token_id=50258, pad_token_id=50260, attention_mask=attention_mask).tolist()[0]
    out_str = tokenizer.decode(out_tokenized)
    out_str = out_str.split('\n')[0]

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

    system_response_str_delex = system_response_str

    if ',' in actions_str:
        act = [s.strip() for s in actions_str.split(',')]
    else:
        act = [actions_str.strip()]

    if ',' in beliefs_str:
        bel = [s.strip() for s in beliefs_str.split(',')]
    else:
        bel = [beliefs_str.strip()]

    if 'offer' in actions_str:
        for a in act:
            slot = extract_slot_or_value(a, 'offer')
            d[slot] = ''

            for b in bel:
                if slot in b:
                    value = extract_slot_or_value(b, slot)
                    d[slot] = value

        print(d)

        #CheckBalance api (select balance from the checking / saving account from the database)

        for k in d:
            if d[k] == '': # no value means that we select from the database
                # db_val = select slot (balance) from accounts where account_type = savings (search in d key value) and user = currentUser (app info)
            
                # if not succesful:
                # system_response_str = system_response_str_delex = 'I could not complete the operation. Please try again'
                # break

                #if successful: 
                d[k] = '900' #db_val

            if '[banks_' + k + ']' in system_response_str:
                system_response_str_delex = system_response_str_delex.replace('[banks_' + k + ']', d[k])
   
    elif 'confirm' in actions_str:
        for a in act:
            slot = extract_slot_or_value(a, 'confirm')
            d[slot] = ''

            for b in bel:
                if slot in b:
                    value = extract_slot_or_value(b, slot)
                    d[slot] = value

        print(d)

        #Confirm step, we just lexicalize
        for k in d:
            if '[banks_' + k + ']' in system_response_str:
                system_response_str_delex = system_response_str_delex.replace('[banks_' + k + ']', d[k])

    # moment when we call Transfer Api
    elif 'notify_success' in actions_str:
        # pass

            # if a value is still empty, check beliefs again
            if '' in d.values():
                for k in d:
                    for b in bel:
                        if k in b:
                            value = extract_slot_or_value(b, k)
                            d[k] = value
            print(d)

            # if '' in d.values(): # still an empty value, not successful

            # call transfer api
                
            #if not succesful:
            # system_response_str = system_response_str_delex = 'I could not complete the operation. Please try again'
                
            #else
            #do nothing, just inform the user that it worked
                   
    system_resp_tokenized = tokenizer.encode(' <|system|> '+ system_response_str, return_tensors='pt')
    history = torch.cat([torch.LongTensor(history), user_input_tokenized, system_resp_tokenized], dim=-1).tolist()

    return history, beliefs_str, actions_str, system_response_str, system_response_str_delex, d

history = []
d = {}
while True:
    print('----------------------------------------------------------------------------')
    user_input = input('You: ')

    if user_input == 'quit' or  user_input == 'q':
        break
    print()

    history, beliefs, actions, system_response, system_response_str_delex, d = generate_response(user_input, history, d)

    if beliefs == '  ':
        print('\tNo belief could be extracted from this exact user input.')
    else:
        print('\tBeliefs:', beliefs)

    print('\tActions:', actions)

    print()
    print('System:', system_response_str_delex)

    if system_response_str_delex == 'I could not complete the operation. Please try again':
        break