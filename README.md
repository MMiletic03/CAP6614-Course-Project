# CAP6614-Course-Project
Course project for the course CAP6614 - Current Topics in Machine Learning at the University of Central Florida. The project aims to reproduce the results of the paper 'A Simple and Effective Pruning Approach for Large Language Models' (https://arxiv.org/abs/2306.11695) and additionally perform sensitivity analysis using Meta's LlaMa-2-7B model.  

Team members include: Michael Miletic (MMiletic03), Jordan Haag (twinjordan02), and Megan Diehl.

To begin using this project, ensure that Meta's Llama-2-7b_hf model (~13GB) and the wikitext-2-raw-v1 dataset (<1 GB) are both donwloaded on your machine.

LlaMa-2-7b_hf: https://huggingface.co/meta-llama/Llama-2-7b-hf 

*Note: You must granted access to this model before downloading it. Once granted access, download the model and generate and authentication token to use to login to the Hugging Face CLI.

wikitext-2-raw-v1: https://huggingface.co/datasets/Salesforce/wikitext

Once these have been installed in your working directory, create a virtual environment containing the packages with the required versions listed in requirements.txt

Clone the folder: Pruning-and-Perplexity

Run the main.py file in your working directory.

example run:
python main.py \
  --model meta-llama/Llama-2-7b-hf \
  --seed 0 \
  --nsamples 128 \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured \
  --prune_method wanda \
  --cache_dir /workspace/wanda/llm_weights \
  --save /workspace/wanda_runs/llama2_7b_wanda_s50

Once ran successfully, LlaMa will be pruned according to your input arguments and the weights and tokenizer will be saved locally at your specified file path.

At this point you are ready to perform zero-shot evaluation on the model.
