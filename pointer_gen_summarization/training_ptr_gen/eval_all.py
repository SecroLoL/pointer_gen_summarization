from training_ptr_gen.eval import Evaluate
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model on the validation set.")
    parser.add_argument("--models_dir", type=str, help="Path to saved models dir", required=True)
    parser.add_argument("--custom_vocab_path", type=str, help="Path to custom vocab file", required=False)
    parser.add_argument("--charlm_forward_file", type=str, help="Path to forward charlm file", required=False)
    parser.add_argument("--charlm_backward_file", type=str, help="Path to backward charlm file", required=False)
    args = parser.parse_args()

    print(f"Models dir: {args.models_dir}")

    paths = [f"{args.models_dir}/{model}" for model in os.listdir(args.models_dir)]
    
    best_loss = float('inf')
    best_model = None 
    for model_file in paths:
        print(f"Model file: {model_file}")
        

        evaluator = Evaluate(model_file_path=model_file, 
                            custom_vocab_path=args.custom_vocab_path, 
                            charlm_forward_file=args.charlm_forward_file, 
                            charlm_backward_file=args.charlm_backward_file)
        running_avg_val_loss = evaluator.evaluate()

        if running_avg_val_loss < best_loss:
            best_loss = running_avg_val_loss
            best_model = model_file
    
    print(f"Best model: {best_model}, loss: {best_loss}")
    print("Done evaluating models.")
    