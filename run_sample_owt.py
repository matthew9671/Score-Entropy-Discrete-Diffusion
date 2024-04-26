import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import argparse
import numpy as np

from load_model import load_model, load_model_hf
from transformers import GPT2TokenizerFast, AutoTokenizer
import torch.nn.functional as F
import sampling
from datasets import load_from_disk
import mauve

def main():

    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model-path", default="/home/groups/swl1/yixiuz/torch_fid/downloads/sedd-medium", type=str)
    parser.add_argument("--prefix-len", type=int, default=25)
    parser.add_argument("--suffix-len", type=int, default=25)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--num-batch-examples", type=int, default=100)
    parser.add_argument("--num-test-examples", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--name", type=str, default="")
    # Corrector arguments
    parser.add_argument('--corrector-type', type=str, default="", help='')
    parser.add_argument('--corrector-steps', type=int, default=2, help='')
    parser.add_argument('--entry-time', type=float, default=0.9, help='')
    parser.add_argument('--corrector-step-size', type=float, default=.1, help='')
    args = parser.parse_args()

    # Tokenizer the pretrained SEDD model uses
    sedd_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    dataset = load_from_disk('/scratch/users/yixiuz/datasets/openwebtext_processed_pct100_blk200')
    full_dataset = dataset["train"]
    validation_ratio = 0.01 # 1 pct of the data is used for validation
    validation_len = int(len(full_dataset) * validation_ratio)
    train_len = len(full_dataset) - validation_len
    train_dataset, eval_dataset = torch.utils.data.random_split(full_dataset, [train_len, validation_len], generator=torch.Generator().manual_seed(42)) # fixing seed here

    # Tokenizer used by the SSD-LM paper to preprocess the OpenWebText dataset
    core_lm_name = "roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(core_lm_name, use_fast=False)
    # Get the first sequences to use for testing
    eval_seqs = eval_dataset[:args.num_test_examples]
    eval_text_seqs = tokenizer.batch_decode(eval_seqs['input_ids'])

    device = torch.device('cuda')
    model, graph, noise = load_model(args.model_path, device)

    all_sample_ids = []

    for start_id in range(0, args.num_test_examples, args.num_batch_examples):

        batch_examples = min(args.num_batch_examples, args.num_test_examples - start_id)

        curr_id_seqs = sedd_tokenizer(eval_text_seqs[:batch_examples]).input_ids
        curr_input_ids = [seq[:args.prefix_len] + seq[args.seq_len-args.suffix_len:args.seq_len] for seq in curr_id_seqs]
        input_ids = curr_input_ids * args.num_samples
        input_locs = list(range(args.prefix_len)) + list(range(args.seq_len-args.suffix_len, args.seq_len))
        
        input_ids = torch.tensor(input_ids, device="cuda")

        def proj_fun(x):
            x[:, input_locs] = input_ids
            return x

        sampling_fn = sampling.get_pc_sampler(
            graph, noise, (args.num_samples * batch_examples, args.seq_len), 'analytic', args.steps, 
            device=device, proj_fun=proj_fun,
            corrector_type=args.corrector_type, corrector_entry_time=args.entry_time, 
            num_corrector_steps=args.corrector_steps, corrector_step_size_multiplier=args.corrector_step_size)

        sample_ids = proj_fun(sampling_fn(model))
        all_sample_ids += sample_ids

    all_sample_ids = torch.stack(all_sample_ids)
    all_sample_texts = sedd_tokenizer.batch_decode(all_sample_ids)

    results_folder = "/home/groups/swl1/yixiuz/torch_fid/language_results"
    corrector_name = args.corrector_type or "no_corrector"
    experiment_name = corrector_name + args.name + "_{}_{}".format(args.prefix_len, args.suffix_len) + "_{}steps".format(args.steps)
    if args.corrector_type:
        experiment_name += "_entry{}_csteps{}_stepsize{}".format(args.entry_time, args.corrector_steps, args.corrector_step_size)
    save_folder = os.path.join(results_folder, experiment_name)
    # Save samples
    os.makedirs(save_folder, exist_ok=True)
    torch.save(all_sample_ids, os.path.join(save_folder, "sampled_tokens.pt"))
    with open(os.path.join(save_folder, "sampled_text.txt"), 'w') as file:
        for string in all_sample_texts:
            # Escape newlines before writing
            escaped_string = string.replace('\n', '\\n')
            file.write(escaped_string + '\n')
    with open(os.path.join(save_folder, "ground_truth_text.txt"), 'w') as file:
        for string in eval_text_seqs:
            # Escape newlines before writing
            escaped_string = string.replace('\n', '\\n')
            file.write(escaped_string + '\n')

    # Compute MAUVE score
    # Get the ground truth segments from the eval examples
    sedd_encoded_ids = sedd_tokenizer(eval_text_seqs).input_ids
    examples = []
    for i in range(args.num_test_examples):
        example = sedd_tokenizer.decode(sedd_encoded_ids[i][args.prefix_len:args.seq_len-args.suffix_len])
        examples.append(example)
    # Get the generated segments from the samples
    gen_text_samples = sedd_tokenizer.batch_decode(all_sample_ids[:,args.prefix_len:args.seq_len-args.suffix_len])
    out = mauve.compute_mauve(p_text=gen_text_samples, q_text=examples, device_id=0, max_text_length=1000, verbose=False)
    print(out.mauve)
    with open(os.path.join(save_folder, "mauve.txt"), 'w') as file:
        file.write(str(out.mauve))

if __name__=="__main__":
    main()