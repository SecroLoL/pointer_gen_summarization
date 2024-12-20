"""
Code for scoring quality of summaries produced by model vs reference summaries
"""

import argparse
import glob
import os 
from rouge_score import rouge_scorer


def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

def calculate_rouge_scores(generated_summary, reference_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    scores = scorer.score(reference_summary, generated_summary)
    return scores

def average_rouge_scores(score_list):
    average_scores = {}
    for metric in score_list[0]:
        average_scores[metric] = {
            'precision': sum([score[metric].precision for score in score_list]) / len(score_list),
            'recall': sum([score[metric].recall for score in score_list]) / len(score_list),
            'fmeasure': sum([score[metric].fmeasure for score in score_list]) / len(score_list)
        }
    return average_scores

def main(generated_dir, reference_dir):
    generated_files = sorted(glob.glob(f'{generated_dir}/*.txt'))
    reference_files = sorted(glob.glob(f'{reference_dir}/*.txt'))
    
    all_scores = []
    for gen_file, ref_file in zip(generated_files, reference_files):
        generated_summary = read_file(gen_file)
        reference_summary = read_file(ref_file)

        generated_file_base, ref_file_base = os.path.basename(gen_file), os.path.basename(ref_file)
        generated_base_prefix, ref_base_prefix = generated_file_base.split("_")[0], ref_file_base.split("_")[0]
        assert generated_base_prefix == ref_base_prefix, f"Generated file: {generated_base_prefix} | Ref file {ref_base_prefix}"

        if not generated_summary:
            print(f"Generated summary for {gen_file} not found. Skipping...")
            continue
        if not reference_summary:
            print(f"Reference summary for {ref_file} not found. Skipping...")
            continue
        scores = calculate_rouge_scores(generated_summary, reference_summary)
        all_scores.append(scores)
    
    avg_scores = average_rouge_scores(all_scores)
    
    print("Average ROUGE Scores:")
    for metric in avg_scores:
        print(f"{metric.upper()}: Precision={avg_scores[metric]['precision']:.3f}, Recall={avg_scores[metric]['recall']:.3f}, F1={avg_scores[metric]['fmeasure']:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate ROUGE scores for generated summaries against reference summaries.")
    parser.add_argument('--generated_dir', type=str, required=True, help="Directory containing generated summaries")
    parser.add_argument('--reference_dir', type=str, required=True, help="Directory containing reference summaries")
    
    args = parser.parse_args()
    main(args.generated_dir, args.reference_dir)
