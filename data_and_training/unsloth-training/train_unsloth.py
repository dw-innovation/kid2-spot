import os
import pandas as pd
import torch
from argparse import ArgumentParser
import jsonlines
from tqdm import tqdm
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer
from dotenv import load_dotenv


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


with open(f"data/zero_shot_cot_prompt.txt", 'r') as file:
    instruction_file = file.read()


def train(output_name, model_name, train_path, dev_path, epochs, lora_r,lora_alpha, random_state, early_stopping,
          eval_steps, save_steps, auto_batch_size, batch_size, learning_rate, weight_decay, lr_scheduler,
          max_seq_length, dtype, load_in_4bit):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    """We now add LoRA adapters so we only need to update 1 to 10% of all parameters!"""

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=lora_alpha, # 8, 16, 32
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        # random_state=random_state,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

    def formatting_prompts_func(examples):
        inputs = examples["sentence"]
        outputs = examples["query"]
        instructions = [instruction_file] * len(inputs)
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts, }


    train_ds = pd.read_csv(train_path, sep='\t')  # , delimiter=r"\s+")
    val_ds = pd.read_csv(dev_path, sep='\t')  # , delimiter=r"\s+")

    train_ds['sentence'] = train_ds['sentence'].apply(lambda x: x.lower())
    train_ds['query'] = train_ds['query'].apply(lambda x: x.lower())
    train_data = Dataset.from_pandas(train_ds)
    train_data = train_data.map(formatting_prompts_func, batched=True, )

    val_ds['sentence'] = val_ds['sentence'].apply(lambda x: x.lower())
    val_ds['query'] = val_ds['query'].apply(lambda x: x.lower())
    val_data = Dataset.from_pandas(val_ds)
    val_data = val_data.map(formatting_prompts_func, batched=True, )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=val_data,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping)],
            args=TrainingArguments(
            evaluation_strategy="steps",
            do_eval=True,
            save_strategy="steps",
            eval_steps=eval_steps,
            save_steps=save_steps,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            auto_find_batch_size=auto_batch_size,
            # gradient_accumulation_steps = 4,
            warmup_steps=int(len(train_ds) / 8),  # 5,
            # max_steps = 60,
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=weight_decay,
            lr_scheduler_type = lr_scheduler,
            # seed = random_state,
            output_dir=output_name,
            num_train_epochs=epochs,
            load_best_model_at_end=True,
        ),
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')

    # model.save_pretrained_merged(f'{output_name}_merged_4bit', tokenizer, save_method = "merged_4bit_forced", token=hf_token)
    # model.push_to_hub_merged(f'{output_name}_merged_4bit', tokenizer, save_method="merged_4bit_forced", token=hf_token)

    model.save_pretrained(f'models/{output_name}_local')  # Local saving
    tokenizer.save_pretrained(f'models/{output_name}_local')

    model.push_to_hub(f'{output_name}_lora', token=hf_token)  # Online saving
    tokenizer.push_to_hub(f'{output_name}_lora', token=hf_token)  # Online saving

    # model.save_pretrained_merged(f'{output_name}_lora', tokenizer, save_method = "lora", token=hf_token)
    # model.push_to_hub_merged(f'{output_name}_lora', tokenizer, save_method = "lora", token=hf_token)

    # # # for cpu code
    # quant_methods = ["q2_k", "q3_k_m", "q4_k_m", "q5_k_m", "q6_k", "q8_0"]
    # cpu_output_name = f'{output_name}_cpu'
    # for quant in quant_methods:
    #     model.save_pretrained_gguf(cpu_output_name, tokenizer, quantization_method=quant)
    #     model.push_to_hub_gguf(cpu_output_name, tokenizer, quant, token=hf_token)


def test(output_name, max_seq_length, dtype, load_in_4bit):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=f'models/{output_name}_local',  # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

    test_sentences = pd.read_csv('data/sentences.txt', sep='\t')
    test_sentences = test_sentences['sentence'].tolist()

    with open(f'test_results/predictions.jsonl', 'a') as outfile:
        results = []
        for sentence in tqdm(test_sentences, total=len(test_sentences)):
            sentence = sentence.lower()
            inputs = tokenizer(
                [
                    alpaca_prompt.format(
                        instruction_file,  # instruction
                        sentence,  # input
                        "",  # output - leave this blank for generation!
                    )
                ], return_tensors="pt").to("cuda")

            outputs = model.generate(
                **inputs,
                max_new_tokens=1048,
                use_cache=True,
                top_p=0.1,
                temperature=0.001,
            )
            outputs = tokenizer.batch_decode(outputs)[0]

            # input = outputs.split("### Input:")[1].split("### Response:")[0]
            respo = outputs.split("### Response:")[1].split("<|end_of_text|>")[0]

            results.append({
                "sentence": sentence,
                "model_result": respo
            })

    with jsonlines.open(f'test_results/{output_name}.jsonl', mode='w') as writer:
        for sample in results:
            writer.write(sample)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--dev_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--lora_r', type=int, required=True)
    parser.add_argument('--lora_alpha', type=int, required=True)
    parser.add_argument('--random_state', type=int, required=True)
    parser.add_argument('--early_stopping', type=int, required=True)
    parser.add_argument('--eval_steps', type=int, required=True)
    parser.add_argument('--save_steps', type=int, required=True)
    parser.add_argument('--auto_batch_size', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--weight_decay', type=float, required=True)
    parser.add_argument('--lr_scheduler', type=str, required=True)
    parser.add_argument('--max_seq_length', type=int, default=2048)
    parser.add_argument('--dtype', type=str, default=None)
    parser.add_argument('--load_in_4bit', type=int, default=True)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    output_name = args.output_name
    model_name = args.model_name
    train_path = args.train_path
    dev_path = args.dev_path
    epochs = args.epochs
    lora_r = args.lora_r
    lora_alpha = args.lora_alpha
    random_state = args.random_state
    early_stopping = args.early_stopping
    eval_steps = args.eval_steps
    save_steps = args.save_steps
    auto_batch_size = args.auto_batch_size
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    lr_scheduler = args.lr_scheduler
    max_seq_length = args.max_seq_length
    dtype = args.dtype
    load_in_4bit = args.load_in_4bit
    _train = args.train
    _test = args.test

    if dtype == "-1":
        dtype = None
    if load_in_4bit == 1:
        load_in_4bit = True
    else:
        load_in_4bit = False
    if auto_batch_size == 1:
        auto_batch_size = True
    else:
        auto_batch_size = False

    if _train:
        train(output_name, model_name, train_path, dev_path, epochs, lora_r,lora_alpha, random_state, early_stopping,
              eval_steps, save_steps, auto_batch_size, batch_size, learning_rate, weight_decay, lr_scheduler,
              max_seq_length, dtype, load_in_4bit)
    if _test:
        test(output_name, max_seq_length, dtype, load_in_4bit)