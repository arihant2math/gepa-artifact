from functools import partial
import dspy

from .. import dspy_program
from ..hover.hover_program import search

rm = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")
dspy.configure(rm=rm)

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

def answer_match_fn(prediction, answers, frac=1.0):
    """Returns True if the prediction matches any of the answers."""
    from dspy.dsp.utils import EM, F1

    if frac >= 1.0:
        return EM(prediction, answers)

    return F1(prediction, answers) >= frac

def get_textual_context(d):
    title_to_sentences = {title:sentences for title, sentences in zip(d['context']['title'], d['context']['sentences'])}
    text = ""

    useful_titles = set(d['supporting_facts']['title'])

    for title in useful_titles:
        text += title + " | " + "".join(title_to_sentences[title])

    return text

def answer_exact_match_with_feedback(example, pred, trace=None, frac=1.0):
    ans_match = None
    if isinstance(example.answer, str):
        ans_match = answer_match_fn(pred.answer, [example.answer], frac=frac)
    elif isinstance(example.answer, list):
        ans_match = answer_match_fn(pred.answer, example.answer, frac=frac)
    
    textual_context = ""
    if hasattr(pred, "feedback_text"):
        textual_context = pred.feedback_text + "\n\n"

    textual_context += get_textual_context(example)

    if ans_match:
        return dspy.Prediction(score=ans_match, feedback=f"The provided answer, '{pred.answer}' is correct. Here's some additional context behind the answer:\n{textual_context}")
    else:
        return dspy.Prediction(score=ans_match, feedback=f"The provided answer, '{pred.answer}' is incorrect. The correct answer is: {example.answer}. Here's some context behind the answer, and how you could have reasoned to get the correct answer:\n{textual_context}")

def provide_feedback_to_answer_module(
    predictor_output,
    predictor_inputs,
    module_inputs,
    module_outputs, 
    captured_trace
):
    prediction = answer_exact_match_with_feedback(module_inputs, module_outputs)
    return {
        "feedback_score": prediction.score,
        "feedback_text": prediction.feedback,
    }

feedback_fn_map = {
    'final_answer.predict': provide_feedback_to_answer_module,
}
