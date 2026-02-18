from pathlib import Path

import sys

import dspy
from dspy import Signature

sys.path.append(str(Path(__file__).parent.parent))

from gepa_artifact.benchmarks.hotpotQA import benchmark as hotpot_b

from openevolve.evaluation_result import EvaluationResult

from litellm import completion
import os
from functools import partial

import bm25s
import Stemmer


class DotDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{key}'"
            )

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{key}'"
            )


stemmer = None
retriever = None
corpus = None
initialized = False

from diskcache import Cache

import threading

init_lock = threading.Lock()


def initialize_bm25s_retriever_and_corpus(directory):
    from dspy.utils import download

    download(
        "https://huggingface.co/dspy/cache/resolve/main/wiki.abstracts.2017.tar.gz"
    )
    # !tar -xzvf wiki.abstracts.2017.tar.gz
    import tarfile

    with tarfile.open("wiki.abstracts.2017.tar.gz", "r:gz") as tar:
        tar.extractall(path=directory)

    import ujson

    corpus = []

    assert os.path.exists(os.path.join(directory, "wiki.abstracts.2017.jsonl")), (
        "Corpus file not found. Please ensure the corpus is downloaded and extracted correctly."
    )

    with open(os.path.join(directory, "wiki.abstracts.2017.jsonl")) as f:
        for line in f:
            line = ujson.loads(line)
            corpus.append(f"{line['title']} | {' '.join(line['text'])}")

    import bm25s
    import Stemmer

    stemmer = Stemmer.Stemmer("english")
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)

    retriever = bm25s.BM25(k1=0.9, b=0.4)
    retriever.index(corpus_tokens)

    retriever.save(os.path.join(directory, "bm25s_retriever"))
    assert os.path.exists(os.path.join(directory, "bm25s_retriever")), (
        "Retriever not saved correctly."
    )


def init_retriever():
    global retriever, stemmer, corpus, initialized
    if initialized:
        return
    with init_lock:
        if not initialized:
            if not os.path.exists(
                os.path.join(os.path.dirname(__file__), "bm25s_retriever")
            ) or not os.path.exists(
                os.path.join(os.path.dirname(__file__), "wiki.abstracts.2017.jsonl")
            ):
                initialize_bm25s_retriever_and_corpus(os.path.dirname(__file__))
            retriever = bm25s.BM25.load(
                os.path.join(os.path.dirname(__file__), "bm25s_retriever")
            )
            stemmer = Stemmer.Stemmer("english")
            import ujson

            corpus_data = []
            with open(
                os.path.join(os.path.dirname(__file__), "wiki.abstracts.2017.jsonl")
            ) as f:
                for line in f:
                    line = ujson.loads(line)
                    corpus_data.append(f"{line['title']} | {' '.join(line['text'])}")
            corpus = corpus_data
            initialized = True


# Initialize cache with a dedicated directory
cache = Cache(os.path.join(os.path.dirname(__file__), "retriever_cache"))


@cache.memoize()
def search(query: str, k: int) -> list[str]:
    init_retriever()
    tokens = bm25s.tokenize(query, stopwords="en", stemmer=stemmer, show_progress=False)
    results, scores = retriever.retrieve(tokens, k=k, n_threads=1, show_progress=False)
    run = {corpus[doc]: float(score) for doc, score in zip(results[0], scores[0])}
    return DotDict({"passages": list(run.keys())[:k]})


class LangProBeDSPyMetaProgram(dspy.Module):
    def setup_lm(self, lm, api_key=None, api_base=None):
        dspy.settings.experimental = True
        self.lm = dspy.LM(lm, api_key=api_key, api_base=api_base)
        self.set_lm(self.lm)

    def program_type(self):
        return "dspy"


class Predict(dspy.Predict, LangProBeDSPyMetaProgram):
    pass


class CoT(dspy.ChainOfThought, LangProBeDSPyMetaProgram):
    pass


class HotpotSingleHop(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self, prompt: str):
        super().__init__()
        self.k = 7
        self.retrieve_k = partial(search, k=self.k)  # dspy.Retrieve(k=self.k)
        self.final_answer = dspy.ChainOfThought(
            Signature("question,passages->answer").with_instructions(prompt)
        )

    def forward(self, question):
        # HOP 1
        hop1_docs = self.retrieve_k(question).passages
        final_answer = self.final_answer(question=question, passages=hop1_docs).answer

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
    return (
        FEEDBACK_PROMPT_TEMPLATE.replace("{{question}}", log["question"])
        .replace("{{correct_answer}}", question["answer"])
        .replace("{{first_hop_output}}", str(log["hop1_output"].passages))
        .replace("{{summary_1_output}}", str(log["summary_1_output"].summary))
        .replace("{{second_hop_output}}", str(log["hop2_output"].passages))
        .replace("{{summary_2_output}}", str(log["summary_2_output"].summary))
        .replace("{{final_answer_output}}", str(log["final_answer_output"].answer))
    )


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
        response = completion(
            model="openai/gpt-4.1-mini", messages=[{"role": "user", "content": prompt}]
        )
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
        condense_prompt += f'\nFeedback for "{question}":\n' + feedback
    response = completion(
        model="openai/gpt-4.1-mini",
        messages=[{"role": "user", "content": condense_prompt}],
    )
    return response.choices[0].message.content


def create_lm(lm_config: dict):
    config = lm_config.copy()
    config["model"] = config.pop("new_model_name", config["model"])
    from dspy.clients.lm_local_arbor import ArborProvider

    provider = ArborProvider() if "openai/arbor" in config["model"] else None
    fixed_config = {
        "max_tokens": 16384,  # overriding the dspy defaults
        "num_retries": 0,
        "provider": provider,
    }
    config = {k: v for k, v in config.items() if k != "name"}
    return dspy.LM(**config, **fixed_config)


lm_for_optimizer = create_lm({"model": "openai/gpt-4.1-mini", "temperature": 1})
adapter = dspy.settings.adapter  # if "qwen" not in lm_name else XMLAdapter()
dspy.configure(lm=lm_for_optimizer, adapter=adapter)


def evaluate(prompt_path):
    # Read prompt
    prompt = open(prompt_path, "r").read()

    benchmark_meta = hotpot_b[0]
    benchmark = benchmark_meta.benchmark()
    final_eval_set = benchmark.test_set
    # optimizers = get_optimizers()
    # opt_idx = 4  # GEPA
    # optimizer_config = optimizers[opt_idx][1]
    program = HotpotSingleHop(prompt)
    evaluate_prog = dspy.Evaluate(
        devset=final_eval_set,
        metric=metric_fn_with_logger,
        num_threads=4,
        display_progress=True,
        max_errors=len(final_eval_set) * 10,
        provide_traceback=True,
        return_outputs=True,
    )
    score, outputs = evaluate_prog(program)
    feedback = generate_feedback(final_eval_set, program.logs, outputs)
    return EvaluationResult(
        metrics={
            "combined_score": score / 100,
            "prompt_length": len(prompt),
        },
        artifacts={"feedback": feedback},
    )


if __name__ == "__main__":
    print(evaluate("initial_prompt.txt"))
