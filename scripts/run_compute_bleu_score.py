import json
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    
def extract_response(text):
    response_start_index = text.find('<|response|>') + len('<|response|>')
    response_end_index = text.find('<|endofresponse|>', response_start_index)
    response_str = text[response_start_index:response_end_index].strip()

    return response_str

def compute_bleu_score(reference_file, output_file):
    reference_data = load_json(reference_file)
    output_data = load_json(output_file)

    # Ensure both files have the same number of samples
    if len(reference_data) != len(output_data):
        print("Error: The number of samples in the reference and output files do not match.")
        return

    scores = []

    for ref, out in zip(reference_data, output_data):
        # Tokenizing the sentences
        reference = word_tokenize(extract_response(ref))
        output = word_tokenize(extract_response(out))

        # reference = word_tokenize(ref)
        # output = word_tokenize(out)

        # Calculate BLEU score for the pair
        score = sentence_bleu([reference], output)
        scores.append(score)

    # Calculate the average BLEU score
    average_score = sum(scores) / len(scores)
    return average_score, scores

# Example usage
bleu_score, scores = compute_bleu_score('reference.json', 'output.json')
print(f"Average BLEU Score: {bleu_score}")
scores.sort()