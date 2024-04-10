from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache_Q4,
    ExLlamaV2Tokenizer,
)

from exllamav2.generator import (
    ExLlamaV2BaseGenerator,
    ExLlamaV2Sampler
)


batch_size = 1
cache_max_seq_len = 4096

model_directory = "/Volumes/training/llm/model_snapshots/models--turboderp--command-r-plus-103B-exl2--4.0bpw/"

config = ExLlamaV2Config(model_directory)
config.max_output_len = 1
config.max_batch_size = batch_size

model = ExLlamaV2(config)
print("Loading model: " + model_directory)

# キャッシュはQ4を利用
cache = ExLlamaV2Cache_Q4(model, lazy = True, batch_size = batch_size, max_seq_len=cache_max_seq_len,) 
model.load_autosplit(cache)

tokenizer = ExLlamaV2Tokenizer(config)
generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)

# サンプリングの設定
settings = ExLlamaV2Sampler.Settings()
settings.temperature = 0.0
settings.top_k = 50
settings.top_p = 0.9
settings.token_repetition_penalty = 1.05

max_new_tokens = 512
