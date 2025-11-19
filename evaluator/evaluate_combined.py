import os
from evaluator.generate_output import generate_standard_answer, generate_output
from evaluator.evaluate_bert_score import compute_bert_score
from evaluator.evaluate_ppl import compute_ppl

def evaluate_combined(
    model,
    tokenizer,
    test_dataset,
    max_new_tokens: int=256,
    batch_size: int=4,
    enable_thinking: bool=False,
    bert_model_type: str = "roberta-large",
    language: str = "en",
    rescale_with_baseline: bool = False,
    idf: bool = False,
    output_dir: str = None,
) -> dict:
    results = {}
    generate_standard_answer(
        test_dataset=test_dataset,
        output_dir=output_dir
    )
    results['standard'] = os.path.join(output_dir, "standard_answer.json")
    generate_output(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        enable_thinking=enable_thinking,
        output_dir=output_dir
    )
    results['output'] = os.path.join(output_dir, "output.json")
    ppl = compute_ppl(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        batch_size=batch_size,
        enable_thinking=enable_thinking,
        output_dir=output_dir
    )
    results['ppl'] = ppl

    bert_score = compute_bert_score(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        max_new_tokens=max_new_tokens,
        gen_batch_size=batch_size,
        bscore_batch_size=64,
        enable_thinking=enable_thinking,
        bert_model_type=bert_model_type,
        language=language,
        rescale_with_baseline=rescale_with_baseline,
        idf=idf,
        output_dir=output_dir
    )
    results['bert_score'] = bert_score

    return results