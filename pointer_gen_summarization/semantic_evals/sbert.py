"""
This script evaluates the similarity between two sets of texts using the SBERT model.
"""
from sentence_transformers import SentenceTransformer, util
import os
import argparse

def evaluate_sbert_scores(text1, text2):
    # Load pre-trained SBERT model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Encode the texts
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)

    # Compute cosine similarity score
    cosine_score = util.pytorch_cos_sim(embedding1, embedding2)

    # Extract the score (single value)
    score = cosine_score.item()

    return score


def evaluate_directories(dir1, dir2):
    scores = {}
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    for filename in os.listdir(dir1):
        filepath1 = os.path.join(dir1, filename)
        name = filename.split('_')[0]
        filepath2 = os.path.join(dir2, f"{name}_reference.txt")
       
        if os.path.isfile(filepath1) and os.path.isfile(filepath2):
            with open(filepath1, 'r', encoding='utf-8') as file1, open(filepath2, 'r', encoding='utf-8') as file2:
                text1 = file1.read()
                text2 = file2.read()

                # Encode the texts
                embedding1 = model.encode(text1, convert_to_tensor=True)
                embedding2 = model.encode(text2, convert_to_tensor=True)

                # Compute cosine similarity score
                cosine_score = util.pytorch_cos_sim(embedding1, embedding2)

                # Extract the score (single value)
                score = cosine_score.item()

                scores[filename] = score

    return scores


def average_score(scores):
    if not scores:
        return 0.0
    total_score = sum(scores.values())
    average = total_score / len(scores)
    return average

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate SBERT similarity between texts in two directories.")
    parser.add_argument('--gen_dir', type=str, required=True, help="Directory containing generated summaries")
    parser.add_argument('--ref_dir', type=str, required=True, help="Directory containing reference summaries")
    args = parser.parse_args()

    scores = evaluate_directories(args.gen_dir, args.ref_dir)
    avg_score = average_score(scores)

    print(f"Average SBERT similarity score: {avg_score:.4f}")

