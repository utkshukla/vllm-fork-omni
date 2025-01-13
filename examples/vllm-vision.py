from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# IMAGES = [
#     "/root/ld/ld_project/MiniCPM-V/assets/airplane.jpeg", # local_path
# ]

MODEL_NAME = "../models/MoE/hf-models/phio" # local_model_path or huggingface_model_name
# MODEL_NAME = "../models/MiniCPM-V-2_6" # local_model_path or huggingface_model_name
# print("test")
# image = Image.open(IMAGES[0]).convert("RGB")
llm = LLM(model=MODEL_NAME,
          gpu_memory_utilization=1,
          trust_remote_code=True,
          max_model_len=16) # if your memory is not enough,reduce it

image = Image.open("vllm/docs/source/assets/logos/vllm-logo-text-dark.png")

tokenizer = AutoTokenizer.from_pretrained("./", trust_remote_code=True)
# messages = [{'role': 'user', 'content': '(<image>./</image>)\n' + 'please describe the picture'}]
user_prompt = '<|user|>\n'
assistant_prompt = '<|assistant|>\n'
prompt_suffix = '<|end|>\n'
messages = [{'role': 'user', 'content': f'{user_prompt}<|image_1|>\nWhat is shown in this image?{prompt_suffix}{assistant_prompt}'}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# set stop token id
# 2.0
# stop_token_ids = [tokenizer.eos_id]
# 2.5
#stop_token_ids = [tokenizer.eos_id, tokenizer.eot_id]
# 2.6 
stop_tokens = ['<|im_end|>', '<|endoftext|>']
stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

# set generate param
sampling_params = SamplingParams(
    stop_token_ids=stop_token_ids,
    # temperature=0.7,
    # top_p=0.8,
    # top_k=100,
    # seed=3472,
    max_tokens=200,
    # min_tokens=150,
    temperature=0,
    # use_beam_search=True,
    # length_penalty=1.2,
    best_of=1)

# get output
outputs = llm.generate({
    "prompt": prompt,
    "multi_modal_data": {
        "image": [image]
    }
}, sampling_params=sampling_params)
print(outputs[0].outputs[0].text)