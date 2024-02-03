import json

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def extract_beliefs(text):
    beliefs_start_index = text.find('<|belief|>') + len('<|belief|>')
    beliefs_end_index = text.find('<|endofbelief|>', beliefs_start_index)
    beliefs_str = text[beliefs_start_index:beliefs_end_index].strip()

    return beliefs_str

def compute_joint_accuracy(reference_file, output_file):
    reference_data = load_json(reference_file)
    output_data = load_json(output_file)

    # Ensure both files have the same number of samples
    if len(reference_data) != len(output_data):
        print("Error: The number of samples in the reference and output files do not match.")
        return

    total_samples = len(reference_data)
    correct_matches = 0

    for ref, out in zip(reference_data, output_data):
        ref_belief = extract_beliefs(ref)
        out_belief = extract_beliefs(out)

        if ref_belief == out_belief:
            correct_matches += 1

    joint_accuracy = correct_matches / total_samples
    return joint_accuracy

# Example usage
accuracy = compute_joint_accuracy('reference.json', 'output.json')
print(f"Joint Accuracy: {accuracy * 100:.2f}%")