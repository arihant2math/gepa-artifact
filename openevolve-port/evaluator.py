import os
import sys
from pathlib import Path

import dspy

sys.path.append(str(Path(__file__).parent.parent))

from gepa_artifact.utils.metric_logger import MetricWithLogger, CounterWithLock
from gepa_artifact.benchmarks.hotpotQA import benchmark as hotpot_b
from gepa_artifact.utils.capture_stream_logger import Logger

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
    'model': 'openai/gpt-4-mini',
    'temperature': 1
})
adapter = dspy.settings.adapter  # if "qwen" not in lm_name else XMLAdapter()
dspy.configure(lm=lm_for_optimizer, adapter=adapter)

def evaluate(prompt_path):
    # Read prompt
    text = open(prompt_path, "r").read()

    benchmark_meta = hotpot_b[0]
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
        print(score, outputs)
        return {
            "combined_score": score,
            "prompt_length": len(text)
        }
