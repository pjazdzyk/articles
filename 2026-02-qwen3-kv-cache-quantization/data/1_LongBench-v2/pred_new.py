import os, csv, json
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset
import re
from openai import OpenAI
from transformers import AutoTokenizer
import tiktoken
import torch.multiprocessing as mp

model_map = json.loads(open('config/model2path.json', encoding='utf-8').read())
maxlen_map = json.loads(open('config/model2maxlen.json', encoding='utf-8').read())

URL = "http://192.168.1.11:1234/v1"
API_KEY = "lm-studio"
template_rag = open('prompts/0shot_rag.txt', encoding='utf-8').read()
template_no_context = open('prompts/0shot_no_context.txt', encoding='utf-8').read()
template_0shot = open('prompts/0shot.txt', encoding='utf-8').read()
template_0shot_cot = open('prompts/0shot_cot.txt', encoding='utf-8').read()
template_0shot_cot_ans = open('prompts/0shot_cot_ans.txt', encoding='utf-8').read()

def query_llm(prompt, model, tokenizer, client=None, temperature=0.5, max_new_tokens=128, stop=None):
    # truncate
    # Modified: Ensure max_len with default max
    max_len = maxlen_map.get(model, 120000)
    if model in model_map:
        input_ids = tokenizer.encode(prompt)
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
            prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
    else:
        input_ids = tokenizer.encode(prompt, disallowed_special=())
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
            prompt = tokenizer.decode(input_ids)
            
    tries = 0
    if model in model_map:
        model = model_map[model]
        
    while tries < 5:
        tries += 1
        try:
            completion = client.chat.completions.create(
                model=model,
                # Modified: Added system prompt to ensure proper response structure
                messages=[
                    {"role": "system", "content": "You are a precise academic assistant. Provide the answer letter only. Do NOT explain your answer. Follow this rule strictly."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_new_tokens,
                # Modified: Added stop tokens to prevent unnecessary token generation
                stop=["\n", "Explanation:", "Reasoning:"]
            )
            return completion.choices[0].message.content
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print("Error Occurs: \"%s\"        Retry ..."%(str(e)))
            time.sleep(1)
    else:
        print("Max tries. Failed.")
        return ''

def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            # Modified: Additional robust fallback for short responses or different bracket styles
            match = re.search(r"[\(\[ ]([A-D])[\)\] ]", response)
            if not match:
                match = re.search(r"([A-D])", response)
            return match.group(1) if match else None

def get_pred(data, args, fout):
    model = args.model
    if "gpt" in model or "o1" in model:
        tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_map[model], trust_remote_code=True)
    client = OpenAI(
        base_url=URL,
        api_key=API_KEY
    )
    for item in tqdm(data):
        context = item['context']
        if args.rag > 0:
            template = template_rag
            retrieved = item["retrieved_context"][:args.rag]
            retrieved = sorted(retrieved, key=lambda x: x['c_idx'])
            context = '\n\n'.join([f"Retrieved chunk {idx+1}: {x['content']}" for idx, x in enumerate(retrieved)])
        elif args.no_context:
            template = template_no_context
        elif args.cot:
            template = template_0shot_cot
        else:
            template = template_0shot
        
        # Modified: Implementation of --inst_bottom flag to mitigate "lost in the middle"
        if args.inst_bottom and not (args.rag or args.no_context or args.cot):
            prompt = (
                f"Background document:\n{context.strip()}\n\n"
                f"Based on the document above, answer the following question:\n{item['question'].strip()}\n"
                f"Question: {item['question'].strip()}\n"
                f"(A) {item['choice_A']} (B) {item['choice_B']} (C) {item['choice_C']} (D) {item['choice_D']}\n\n"
                f"Follow this format strictly:\nAnswer: (X)\n\n"
                f"Where X is the letter of the correct choice. Provide ONLY the answer line. Do NOT explain your answer. Follow this rule strictly."
            )
        else:
            prompt = template.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip())
        
        # Modified: Temperature set to 0.0 for more deterministic output during standard inference
        if args.cot:
            output = query_llm(prompt, model, tokenizer, client, temperature=0.1, max_new_tokens=1024)
        else:
            output = query_llm(prompt, model, tokenizer, client, temperature=0.0, max_new_tokens=128)
            
        if output == '':
            continue
        if args.cot: # extract answer
            response = output.strip()
            item['response_cot'] = response
            prompt = template_0shot_cot_ans.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip()).replace('$COT$', response)
            output = query_llm(prompt, model, tokenizer, client, temperature=0.1, max_new_tokens=128)
            if output == '':
                continue
        
        # Modified: Changing to cleaner output file result structure
        response = output.strip()
        item['response'] = response
        item['pred'] = extract_answer(response)
        item['judge'] = item['pred'] == item['answer']
        
        # To keep the file clean, we remove the heavy context from the output line
        # but keep all other metadata for the final JSON result
        result_item = {
            "_id": item["_id"],
            "difficulty": item["difficulty"],
            "length": item["length"],
            "answer": item["answer"],
            "response": item["response"],
            "pred": item["pred"],
            "judge": item["judge"]
        }
        
        fout.write(json.dumps(result_item, ensure_ascii=False) + '\n')
        fout.flush()

def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    
    if args.rag > 0:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + f"_rag_{str(args.rag)}.jsonl")
    elif args.no_context:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + "_no_context.jsonl")
    elif args.cot:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + "_cot.jsonl")
    # Modified: Added inst_bottom to the original filename selection logic
    elif args.inst_bottom:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + ".inst_bottom.jsonl")
    else:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + ".jsonl")

    dataset = load_dataset('THUDM/LongBench-v2', split='train')
    data_all = [{"_id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"], "difficulty": item["difficulty"], "length": item["length"], "question": item["question"], "choice_A": item["choice_A"], "choice_B": item["choice_B"], "choice_C": item["choice_C"], "choice_D": item["choice_D"], "answer": item["answer"], "context": item["context"]} for item in dataset]

    # cache
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}
    fout = open(out_file, 'a', encoding='utf-8')
    data = []
    for item in data_all:
        if item["_id"] not in has_data:
            data.append(item)

    data_subsets = [data[i::args.n_proc] for i in range(args.n_proc)]
    processes = []
    for rank in range(args.n_proc):
        p = mp.Process(target=get_pred, args=(data_subsets[rank], args, fout))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="GLM-4-9B-Chat")
    parser.add_argument("--inst_bottom", action='store_true') # Modified: Added custom flag for research on bottom instruction placement
    parser.add_argument("--cot", "-cot", action='store_true') # set to True if using COT
    parser.add_argument("--no_context", "-nc", action='store_true') # set to True if using no context
    parser.add_argument("--rag", "-rag", type=int, default=0) # set to 0 if RAG is not used
    parser.add_argument("--n_proc", "-n", type=int, default=16)
    args = parser.parse_args()
    main()