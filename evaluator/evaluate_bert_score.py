from bert_score import score as bertscore

from evaluator.evaluate import generate_output
def compute_bert_score(
    model,
    tokenizer,
    test_dataset,
    max_new_tokens: int = 256,
    gen_batch_size: int = 4,
    bscore_batch_size: int = 64,
    enable_thinking: bool = False,
    model_type: str = "roberta-large",
    lang: str = "en",
    rescale_with_baseline: bool = False,
    idf: bool = False,
):

    if "message" in test_dataset[0]:
        system_prompts = [data["message"][0]["content"] for data in test_dataset]
        questions      = [data["message"][1]["content"] for data in test_dataset]
        references     = [data["message"][2]["content"] for data in test_dataset]
    elif "chosen_message" in test_dataset[0]:
        system_prompts = [data["chosen_message"][0]["content"] for data in test_dataset]
        questions      = [data["chosen_message"][1]["content"] for data in test_dataset]
        references     = [data["chosen_message"][2]["content"] for data in test_dataset]

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
        lang=lang,
        model_type=model_type,
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

    return {
        "avg_p": avg_p,
        "avg_r": avg_r,
        "avg_f1": avg_f1,
    }