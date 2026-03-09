This is a reproduction of the inference speed tests for structured vs unstructured sparsity. It does this by testing the matrix multiplication speeds of the linear layers of the llama-2 model (where most of the parameters are stored) on an Ampere-or-newer NVIDIA GPU, using NVIDIA Cutlass' GEMM (General Matrix Multiplication) library.

It requires a separate environment to the GEMM dependencies. Please create a new environment, then run

`pip install -r requirements_for_inference_speed_evaluation.txt`