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

from openevolve.evaluation_result import EvaluationResult

from litellm import completion

class HotpotMultiHop(dspy_program.LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self, create_query_hop2_prompt: str, final_answer_prompt: str, summarize1: str, summarize2: str):
        super().__init__()
        self.k = 7
        self.retrieve_k = partial(search, k=self.k) # dspy.Retrieve(k=self.k)
        self.create_query_hop2 = dspy.ChainOfThought(Signature("question,summary_1->query").with_instructions(create_query_hop2_prompt))
        self.final_answer = dspy.ChainOfThought(Signature("question,summary_1,summary_2->answer").with_instructions(final_answer_prompt))
        self.summarize1 = dspy.ChainOfThought(Signature("question,passages->summary").with_instructions(summarize1))
        self.summarize2 = dspy.ChainOfThought(Signature("question,context,passages->summary").with_instructions(summarize2))
        self.logs = []

    def forward(self, question):
        log = {"question": question}
        # HOP 1
        hop1_docs = self.retrieve_k(question)
        log["hop1_input"] = question
        log["hop1_output"] = hop1_docs
        hop1_docs = hop1_docs.passages
        summary_1 = self.summarize1(
            question=question, passages=hop1_docs
        )
        log["summary_1_input"] = {"question": question, "passages": hop1_docs}
        log["summary_1_output"] = summary_1
        summary_1 =  summary_1.summary  # Summarize top k docs

        # HOP 2
        hop2_query = self.create_query_hop2(question=question, summary_1=summary_1).query
        hop2_docs = self.retrieve_k(hop2_query)
        log["hop2_input"] = {"question": question, "summary_1": summary_1}
        log["hop2_output"] = hop2_docs
        hop2_docs = hop2_docs.passages
        summary_2 = self.summarize2(
            question=question, context=summary_1, passages=hop2_docs
        )
        log["summary_2_input"] = {"question": question, "context": summary_1, "passages": hop2_docs}
        log["summary_2_output"] = summary_2
        summary_2 = summary_2.summary

        # HOP 3
        final_answer = self.final_answer(
            question=question, summary_1=summary_1, summary_2=summary_2
        )
        log["final_answer_input"] = {"question": question, "summary_1": summary_1, "summary_2": summary_2}
        log["final_answer_output"] = final_answer
        final_answer = final_answer.answer

        prediction = dspy.Prediction(answer=final_answer, hop1_docs=hop1_docs, hop2_docs=hop2_docs)
        log["prediction"] = prediction
        self.logs.append(log)
        return prediction

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
    for question, log, output in zip(questions, logs, outputs):
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
                "prompt_lengths": {}
            }
        if current_prompt not in prompts:
            return {
                "combined_score": 0.0,
                "error": f"Prompt \"{current_prompt}\" is not one of the possible prompts: {list(prompts.keys())}",
                "prompt_lengths": {}
            }
        if prompts[current_prompt] is None:
            prompts[current_prompt] = []
        prompts[current_prompt].append(line)

    for name, prompt in prompts.items():
        if prompt is None:
            return {
                "combined_score": 0.0,
                "error": f"No prompt for {name}",
                "prompt_lengths": {}
            }

    benchmark_meta = hotpot_b[0]
    benchmark_meta.program = [HotpotMultiHop("\n".join(prompts["create_query_hop2"]), "\n".join(prompts["final_answer"]), "\n".join(prompts["summarize1"]), "\n".join(prompts["summarize2"]))]
    benchmark = benchmark_meta.benchmark()
    final_eval_set = benchmark.test_set[10:20]
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
        return EvaluationResult(
            metrics={
                "combined_score": score / 100,
                "prompt_lengths": {k: len("\n".join(v)) for k,v in prompts.items()},
            },
            artifacts={
                "feedback": feedback
            }
        )

if __name__ == "__main__":
    print(evaluate("test-prompt.txt"))
