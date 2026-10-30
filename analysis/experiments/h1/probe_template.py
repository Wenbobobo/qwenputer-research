import json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/mnt/c/Projects/Qcompute/.data/Qwen3.5-4B")
tools=[{"type":"function","function":{"name":"put","description":"x","parameters":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}}}]
msgs=[{"role":"system","content":"SYS"},{"role":"user","content":"U1"},
 {"role":"assistant","content":"pre","tool_calls":[{"type":"function","function":{"name":"put","arguments":{"name":"city","text":"Lisbon"}}}]},
 {"role":"tool","content":"ok"}]
print(tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, enable_thinking=False, tokenize=False))
print("EOS", tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("</tool_call>"))
