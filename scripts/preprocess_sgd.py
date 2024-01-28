
import json
import os
import copy
import random
import argparse

# banks_account_type --> checking, savings
# banks_recipient_account_type --> checking, savings
# banks_account_balance
# banks_transfer_amount
# banks_recipient_name
# banks_transfer_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./datasets/bank_dataset/", type=str, required=False, help="path to banking dataset")
    parser.add_argument("--target", default="./training_files/", type=str, required=False, help="path to output")
    args = parser.parse_args()

    datafolder = args.data
    targetfolder = args.target

    for folder in ["train", "val", "test"]:

        if folder == 'train':
            schema_file = open(os.path.join(datafolder, folder, 'banks1_schema.json'))
        elif folder == 'val' or folder == 'test':
            schema_file = open(os.path.join(datafolder, folder, 'banks2_schema.json'))

        schema = json.load(schema_file)
        
        replace_strings = {} # key = slot name (account_type), values = list of possible values

        for elem in schema['slots']:
            if len(elem['possible_values']) > 0:
                replace_strings[elem['name']] = elem['possible_values']

        inlm = []
        inlme = []

        files = os.listdir(datafolder + folder)
        files.sort()

        for file in files:
            if not file.startswith("bank_dialogues"):
                continue
        
            dialogues_file = open(os.path.join(datafolder, folder, file))
            dialogues = json.load(dialogues_file)

            for dialogue in dialogues:
                context = '<|context|> '
                for turn in dialogue['turns']:
                    speaker = turn['speaker']
                    frames = turn['frames'][0]
                    act = frames['actions']
                    utterance = turn['utterance']
                    utterance_lex = copy.deepcopy(utterance)
                    domain = frames['service'].split("_")[0].lower()


                    if speaker == 'USER': #here we build the belief
                        # we build the belief on user turn. so we reset it only on user turn
                        # this list will be available for the next iteration of for (the system)
                        beliefs = []

                        # target will combine beliefs (from user turn) and actions and response from system turn
                        # target needs to be reset on user turn
                        target = ''
                        slot_values = frames['state']['slot_values']

                        for key in slot_values: # slot values is a dict with the key = slot, value = list of values for that slot
                            belief_slot = key
                            belief_value = slot_values[key][0]
                            beliefs.append(domain + ' ' + belief_slot + ' ' + belief_value)

                        target += '<|belief|> ' + ", ".join(beliefs) + ' <|endofbelief|> '
                        context += '<|user|> ' + utterance_lex + ' '
                    else:
                        # we build the actions on the system turn
                        actions = []
                        # first build the delex utterance
                        
                        # change the values based on the position
                        if len(frames['slots']) > 0:
                            delta = 0
                            for replace_info in frames['slots']:
                                start = replace_info['start'] + delta
                                end = replace_info['exclusive_end'] + delta
                                replace_string = '[' + domain + '_' + replace_info['slot'] + ']'

                                utterance_list = list(utterance)
                                utterance_list[start:end] = replace_string
                                utterance = ''.join(utterance_list)

                                num_of_characters_old = end - start
                                num_of_characters_new = len(replace_string)
                                
                                delta = num_of_characters_new - num_of_characters_old

                        # change the slots from schema
                        for slot in replace_strings:
                            for replace_string in replace_strings[slot]:
                                if replace_string in utterance:
                                    action = act[0]
                                    if slot == action['slot']:
                                        utterance = utterance.replace(replace_string, '[' + domain + '_' + slot + ']')
                        
                        # utterance acum = propozitia delexicalizata pt system
                        # dialogues[i]['turns'][j]['utterance'] = utterance

                        # now we build the actions that should be done by the system
                        for action in act:
                            actions.append(domain + ' ' + action['act'].lower() + ' ' + action['slot'])
                        
                        target += '<|action|> ' + ", ".join(actions) + ' <|endofaction|> '
                        target += '<|response|> ' + utterance + ' <|endofresponse|>'

                        prev_context = context

                        context += '<|endofcontext|>'

                        inlm += [(context + target).replace("\n", " ").replace("\r", "")]
                        assert("\n" not in inlm[-1])
                        inlme += [(context).replace("\n", " ").replace("\r", "")]

                        context = prev_context + '<|system|> ' + utterance_lex + ' '

        random.shuffle(inlm)
        
        with open(targetfolder + "input_" + folder + "_entire_structure.txt", "w", encoding='utf8') as f: #SimpleTOD
            f.write('\n'.join(inlm))
        with open(targetfolder + "input_" + folder + "_only_context.txt", "w", encoding='utf8') as f: #used as the input during evaluation of SimpleTOD and SimpleTOD extension
            f.write('\n'.join(inlme))

if __name__ == "__main__":
    random.seed(42)
    main()
