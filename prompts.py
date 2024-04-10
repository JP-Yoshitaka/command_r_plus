# 今回推論する内容
prompts = [
    "Hello, what is your name?",
    "Databricksとは何ですか？詳細に教えてください。",
    "まどか☆マギカでは誰が一番かわいい?",
    "ランダムな10個の要素からなるリストを作成してソートするコードをPythonで書いてください。",
    "現在の日本の首相は誰？",
    "あなたはマラソンをしています。今3位の人を抜きました。あなたの今の順位は何位ですか?",
]

# パディングを最小化するために文字列サイズでソート
s_prompts = sorted(prompts, key=len)

# プロンプトを整形
def format_prompt(sp, p):
    return f"<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{sp}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{p}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"


system_prompt = "You are helpful assistant.You must reply in Japanese."

f_prompts = [format_prompt(system_prompt, p) for p in s_prompts]

# 生計済みプロンプトをバッチに分割
batches = [f_prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]

collected_outputs = []
for b, batch in enumerate(batches):

    print(f"Batch {b + 1} of {len(batches)}...")

    outputs = generator.generate_simple(
        batch, settings, max_new_tokens, seed=1234, add_bos=True
    )

    trimmed_outputs = [o.split("<|CHATBOT_TOKEN|>")[1] for o in outputs] # 簡易分割
    collected_outputs += trimmed_outputs

# 結果出力
for q, a in zip(s_prompts, collected_outputs):
    print("---------------------------------------")
    print("Q: " + q)
    print("A: " + a.strip())
