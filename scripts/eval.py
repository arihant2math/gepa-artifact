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

from litellm import completion


class HotpotMultiHop(dspy_program.LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7
        self.retrieve_k = partial(search, k=self.k) # dspy.Retrieve(k=self.k)
        self.final_answer = dspy.ChainOfThought("question,passages->answer")

    def forward(self, question):
        # HOP 1
        hop1_docs = self.retrieve_k(question).passages
        final_answer = self.final_answer(
            question=question, passages=hop1_docs
        ).answer

        return dspy.Prediction(answer=final_answer, hop1_docs=hop1_docs)

def generate_feedback_prompt(question, log):
    FEEDBACK_PROMPT_TEMPLATE = """**System behavior overview:**
- **First hop:** Documents are queried, with only the knowledge of the previous question.
- **Summary 1:** The agent condenses the information retrieved in the first hop, distilling the most critical information from the top retrieved passages in response to the initial question. 
- **Second hop:** This query aims to retrieve additional relevant documents not found in the first hop using both the question and the first summary as input.
- **Summary 2:** Input passages may not always contain every necessary detail, this module should aim to bridge any gaps by inferring or generalizing, drawing upon information from both the initial summary and new passages. It uses the question, first summary, and the second hop's data.
- **Final Answer:** This module uses the combined information from both summaries to answer the question in an accurate manner.

Provide feedback to improve the following attempt. Your feedback will be used by a prompt optimizing model to improve prompts:

Question: {{question}}

Correct Answer: {{correct_answer}}

First hop output: {{first_hop_output}}

Summary 1 output: {{summary_1_output}}

Second hop output: {{second_hop_output}}

Summary 2 output: {{summary_2_output}}

Final answer: {{final_answer_output}}
"""
    return (FEEDBACK_PROMPT_TEMPLATE
            .replace("{{question}}", log["question"])
            .replace("{{correct_answer}}", question["answer"])
            .replace("{{first_hop_output}}", str(log["hop1_output"].passages))
            .replace("{{summary_1_output}}", str(log["summary_1_output"].summary))
            .replace("{{second_hop_output}}", str(log["hop2_output"].passages))
            .replace("{{summary_2_output}}", str(log["summary_2_output"].summary))
            .replace("{{final_answer_output}}", str(log["final_answer_output"].answer)))


def generate_feedback(questions, logs, outputs):
    feedbacks = []
    count = 0
    for question, log, output in zip(questions, logs, outputs):
        if count > 5:
            break
        if output[2]:
            continue
        count += 1
        prompt = generate_feedback_prompt(question, log)
        response = completion(model="openai/gpt-4.1-mini", messages=[{"role": "user", "content": prompt}])
        feedbacks.append((log["question"], response.choices[0].message.content))
    CONDENSE_PROMPT = """**System behavior overview:**
- **First hop:** Documents are queried, with only the knowledge of the previous question.
- **Summary 1:** The agent condenses the information retrieved in the first hop, distilling the most critical information from the top retrieved passages in response to the initial question. 
- **Second hop:** This query aims to retrieve additional relevant documents not found in the first hop using both the question and the first summary as input.
- **Summary 2:** Input passages may not always contain every necessary detail, this module should aim to bridge any gaps by inferring or generalizing, drawing upon information from both the initial summary and new passages. It uses the question, first summary, and the second hop's data.
- **Final Answer:** This module uses the combined information from both summaries to answer the question in an accurate manner.

The following has been provided as feedback about the current function of the system. Extract repeating themes and issues and provide a condensed version of the feedback that can be executed on more efficiently by an prompt-optimizing agent.
"""
    condense_prompt = CONDENSE_PROMPT
    for question, feedback in feedbacks:
        condense_prompt += f"\nFeedback for \"{question}\":\n" + feedback
    response = completion(model="openai/gpt-4.1-mini", messages=[{"role": "user", "content": condense_prompt}])
    return response.choices[0].message.content


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
    benchmark_meta = hotpot_b[0]
    benchmark_meta.program = [HotpotMultiHop()]
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
        feedback = generate_feedback(final_eval_set, program.logs, outputs)
        return {
            "metrics": {
                "combined_score": score / 100,
            },
            "artifacts": {
                "feedback": feedback
            }
        }

if __name__ == "__main__":
    print(evaluate("initial_prompt.txt"))
