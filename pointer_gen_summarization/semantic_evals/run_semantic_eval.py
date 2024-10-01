"""
Runs the semantic eval of decoded model outputs using BERTScore and S-BERT embedding distances.
"""
from semantic_evals.bertscore import evaluate_directories as evaluate_bertscore
from semantic_evals.sbert import evaluate_directories as evaluate_sbert_scores
import argparse 


def main(gen_dir, ref_dir):
    bert_scores = evaluate_bertscore(gen_dir, ref_dir)
    sbert_scores = evaluate_sbert_scores(gen_dir, ref_dir)

    print(f"BERTScore similarity scores: {bert_scores}")
    print(f"SBERT similarity scores: {sbert_scores}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate BERTScore and SBERT similarity between texts in two directories.")
    parser.add_argument('--gen_dir', type=str, required=True, help="Directory containing generated summaries")
    parser.add_argument('--ref_dir', type=str, required=True, help="Directory containing reference summaries")
    args = parser.parse_args()

    main(args.gen_dir, args.ref_dir)