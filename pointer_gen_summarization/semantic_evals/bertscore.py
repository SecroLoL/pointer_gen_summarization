"""
This script evaluates the similarity between two sets of texts using the BERTScore.
"""
import os
import argparse
from bert_score import score

def evaluate_bertscore(text1, text2):
    # Compute BERTScore
    P, R, F1 = score([text1], [text2], lang="en", verbose=True)
    
    # Extract the F1 score (single value)
    f1_score = F1.item()

    return f1_score


def evaluate_directories(dir1, dir2):
    scores = {}

    for filename in os.listdir(dir1)[:10]:
        filepath1 = os.path.join(dir1, filename)
        name = filename.split('_')[0]
        filepath2 = os.path.join(dir2, f"{name}_reference.txt")
       
        if os.path.isfile(filepath1) and os.path.isfile(filepath2):
            with open(filepath1, 'r', encoding='utf-8') as file1, open(filepath2, 'r', encoding='utf-8') as file2:
                text1 = file1.read()
                text2 = file2.read()

                # Compute BERTScore
                P, R, F1 = score([text1], [text2], lang="en", verbose=True)

                # Extract the F1 score (single value)
                f1_score = F1.item()

                scores[filename] = f1_score

    return scores

def average_score(scores):
    if not scores:
        return 0.0
    total_score = sum(scores.values())
    average = total_score / len(scores)
    return average

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate BERTScore similarity between texts in two directories.")
    parser.add_argument('--gen_dir', type=str, required=True, help="Directory containing generated summaries")
    parser.add_argument('--ref_dir', type=str, required=True, help="Directory containing reference summaries")
    args = parser.parse_args()

    scores = evaluate_directories(args.gen_dir, args.ref_dir)
    avg_score = average_score(scores)

    print(f"Average BERTScore similarity score: {avg_score:.4f}")
