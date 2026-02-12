import os
import sys
from functools import partial
from pathlib import Path
from typing import Optional

import dspy
from dspy import Signature

from gepa_artifact.benchmarks import dspy_program
from gepa_artifact.benchmarks.hover.hover_program import search

sys.path.append(str(Path(__file__).parent.parent))

from gepa_artifact.utils.metric_logger import MetricWithLogger, CounterWithLock
from gepa_artifact.benchmarks.hotpotQA import benchmark as hotpot_b
from gepa_artifact.utils.capture_stream_logger import Logger

class HotpotMultiHop(dspy_program.LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self, create_query_hop2_prompt: str, final_answer_prompt: str, summarize1: str, summarize2: str):
        super().__init__()
        self.k = 7
        self.retrieve_k = partial(search, k=self.k) # dspy.Retrieve(k=self.k)
        self.create_query_hop2 = dspy.ChainOfThought(Signature("question,summary_1->query").with_instructions(create_query_hop2_prompt))
        self.final_answer = dspy.ChainOfThought(Signature("question,summary_1,summary_2->answer").with_instructions(final_answer_prompt))
        self.summarize1 = dspy.ChainOfThought(Signature("question,passages->summary").with_instructions(summarize1))
        self.summarize2 = dspy.ChainOfThought(Signature("question,context,passages->summary").with_instructions(summarize2))

    def forward(self, question):
        # HOP 1
        hop1_docs = self.retrieve_k(question).passages
        summary_1 = self.summarize1(
            question=question, passages=hop1_docs
        ).summary  # Summarize top k docs

        # HOP 2
        hop2_query = self.create_query_hop2(question=question, summary_1=summary_1).query
        hop2_docs = self.retrieve_k(hop2_query).passages
        summary_2 = self.summarize2(
            question=question, context=summary_1, passages=hop2_docs
        ).summary

        # HOP 3
        hop3_answer = self.final_answer(
            question=question, summary_1=summary_1, summary_2=summary_2
        ).answer

        return dspy.Prediction(answer=hop3_answer, hop1_docs=hop1_docs, hop2_docs=hop2_docs)


def create_lm(lm_config: dict):
    config = lm_config.copy()
    config['model'] = config.pop("new_model_name", config['model'])
    from dspy.clients.lm_local_arbor import ArborProvider
    provider = ArborProvider() if "openai/arbor" in config['model'] else None
    fixed_config = {
        "max_tokens": 16384,  # overriding the dspy defaults
        "num_retries": 0,
        "provider": provider,
    }
    config = {k:v for k, v in config.items() if k != "name"}
    return dspy.LM(**config, **fixed_config)


lm_for_optimizer = create_lm({
    'model': 'openai/gpt-4.1-mini',
    'temperature': 1
})
adapter = dspy.settings.adapter  # if "qwen" not in lm_name else XMLAdapter()
dspy.configure(lm=lm_for_optimizer, adapter=adapter)

def evaluate(prompt_path):
    # Read prompt
    text = open(prompt_path, "r").read()
    #     def __init__(self, create_query_hop2_prompt: str, final_answer_prompt: str, summarize1: str, summarize2: str):
    prompts: dict[str, Optional[list[str]]] = {"create_query_hop2": None, "final_answer": None, "summarize1": None, "summarize2": None}
    lines = text.splitlines()
    current_prompt = None
    for line in lines:
        if line.startswith(">>> PROMPT_START"):
            line = line.removeprefix(">>> PROMPT_START").strip()
            current_prompt = line
            continue
        if current_prompt is None:
            return {
                "combined_score": 0.0,
                "error": "Incorrectly formatted prompt list (missing initial >>> PROMPT_START)",
                "prompt_length": len(text)
            }
        if current_prompt not in prompts:
            return {
                "combined_score": 0.0,
                "error": f"Prompt \"{current_prompt}\" is not one of the possible prompts: {list(prompts.keys())}",
                "prompt_length": len(text)
            }
        if prompts[current_prompt] is None:
            prompts[current_prompt] = []
        prompts[current_prompt].append(line)

    for name, prompt in prompts.items():
        if prompt is None:
            return {
                "combined_score": 0.0,
                "error": f"No prompt for {name}",
                "prompt_length": len(text)
            }

    benchmark_meta = hotpot_b[0]
    benchmark_meta.program = [HotpotMultiHop("\n".join(prompts["create_query_hop2"]), "\n".join(prompts["final_answer"]), "\n".join(prompts["summarize1"]), "\n".join(prompts["summarize2"]))]
    benchmark = benchmark_meta.benchmark()
    final_eval_set = benchmark.test_set
    metric_counter = CounterWithLock()
    # optimizers = get_optimizers()
    # opt_idx = 4  # GEPA
    # optimizer_config = optimizers[opt_idx][1]
    program = benchmark_meta.program[0]


    with MetricWithLogger(
            metric_fn=benchmark_meta.metric,
            run_dir="runs/",
            counter_with_lock=metric_counter,
            train_dataset=benchmark.train_set,
            val_dataset=benchmark.val_set,
            test_dataset=benchmark.test_set,
            # log_example=True,
            log_prediction=True
    ) as metric_fn_with_logger, Logger("run_log.txt") as logger:
        evaluate_prog = dspy.Evaluate(
            devset=final_eval_set,
            metric=metric_fn_with_logger,
            num_threads=4,
            display_progress=True,
            max_errors=len(final_eval_set) * 10,
            provide_traceback=True,
            return_outputs=True
        )
        score, outputs = evaluate_prog(program)
        # print(score)
        return {
            "combined_score": score / 100,
            "prompt_length": len(text)
        }

if __name__ == "__main__":
    print(evaluate("test-prompt.txt"))
