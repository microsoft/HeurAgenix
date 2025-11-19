import os
import json
from evaluator.generate_output import generate_output
from bert_score import score as bertscore


def compute_bert_score(
    model,
    tokenizer,
    test_dataset,
    max_new_tokens: int = 256,
    gen_batch_size: int = 4,
    bscore_batch_size: int = 64,
    enable_thinking: bool = False,
    bert_model_type: str = "roberta-large",
    language: str = "en",
    rescale_with_baseline: bool = False,
    idf: bool = False,
    output_dir: str = None,
    **kwargs
):
    if "message" in test_dataset[0]:
        system_prompts = [data["message"][0]["content"] for data in test_dataset]
        questions      = [data["message"][1]["content"] for data in test_dataset]
        references     = [data["message"][2]["content"] for data in test_dataset]
    elif "chosen_message" in test_dataset[0]:
        system_prompts = [data["chosen_message"][0]["content"] for data in test_dataset]
        questions      = [data["chosen_message"][1]["content"] for data in test_dataset]
        references     = [data["chosen_message"][2]["content"] for data in test_dataset]

    if os.path.exists(os.path.join(output_dir, "output.json")):
        with open(os.path.join(output_dir, "output.json"), "r", encoding="utf-8") as f:
            outputs = json.load(f)
    else:
        outputs = generate_output(
            model=model,
            tokenizer=tokenizer,
            test_dataset=test_dataset,
            max_new_tokens=max_new_tokens,
            batch_size=gen_batch_size,
            output_file=None,
            enable_thinking=enable_thinking,
        )

    cands = [o["output"].strip() for o in outputs]
    refs  = [r.strip() for r in references]

    P, R, F1 = bertscore(
        cands,
        refs,
        lang=language,
        model_type=bert_model_type,
        rescale_with_baseline=rescale_with_baseline,
        idf=idf,
        batch_size=bscore_batch_size,
    )

    P_list  = P.cpu().tolist()
    R_list  = R.cpu().tolist()
    F1_list = F1.cpu().tolist()

    avg_p  = float(sum(P_list) / max(len(P_list), 1))
    avg_r  = float(sum(R_list) / max(len(R_list), 1))
    avg_f1 = float(sum(F1_list) / max(len(F1_list), 1))

    results = {
        "avg_p": avg_p,
        "avg_r": avg_r,
        "avg_f1": avg_f1,
    }
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "bert_score_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    return results