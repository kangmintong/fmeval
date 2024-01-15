import logging
import string
from functools import partial


from typing import Any, Callable, List, Optional, Dict

from dataclasses import dataclass

from nltk.metrics.scores import f_measure

import fmeval.util as util
from fmeval.constants import (
    MODEL_INPUT_COLUMN_NAME,
    MODEL_OUTPUT_COLUMN_NAME,
    TARGET_OUTPUT_COLUMN_NAME,
    MODEL_LOG_PROBABILITY_COLUMN_NAME,
    MEAN,
)
from fmeval.data_loaders.util import get_dataset
from fmeval.data_loaders.data_config import DataConfig
from fmeval.eval_algorithms.util import (
    generate_model_predict_response_for_dataset,
    generate_prompt_column_for_dataset,
    aggregate_evaluation_scores,
    validate_dataset,
    save_dataset,
    generate_output_dataset_path,
)
from fmeval.eval_algorithms.eval_algorithm import (
    EvalAlgorithmInterface,
    EvalAlgorithmConfig,
)
from fmeval.eval_algorithms import (
    EvalAlgorithm,
    EvalOutput,
    EvalScore,
    EVAL_DATASETS,
    DATASET_CONFIGS,
    get_default_prompt_template,
)
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.perf_util import timed_block

ENGLISH_ARTICLES = ["a", "an", "the"]
ENGLISH_PUNCTUATIONS = string.punctuation

DT_ROBUSTNESS_ACC = "robust_acc"

PROMPT_COLUMN_NAME = "prompt"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DTRobustnessConfig(EvalAlgorithmConfig):
    """
    Configuration for the QA Accuracy Evaluation

    :param target_output_delimiter: Target Output can have multiple answers. We expect customer to combine all the
        possible answers into a single string and use the delimiter to separate them. For instance,
        if the answers are ["UK", "England"] and the delimiter="<OR>", then the target_output should be "UK<OR>England".
    """

    target_output_delimiter: Optional[str] = "<OR>"

    def __post_init__(self):
        if self.target_output_delimiter == "":
            raise EvalAlgorithmClientError(
                "Empty target_output_delimiter is provided. Please either provide a non-empty string, or set it to None."
            )


def _dt_robustness_acc(model_output: str, target_output: str) -> float:
    if target_output in model_output:
        return 1.0
    else:
        return 0.0


QA_ACCURACY_SCORES_TO_FUNCS: Dict[str, Callable[..., float]] = {
    DT_ROBUSTNESS_ACC: _dt_robustness_acc,
}


class DTRobustness(EvalAlgorithmInterface):
    """
    Robust accuaracy in DecodingTrust
    """

    eval_name = "Robust_Accuracy_DT"

    def __init__(self, eval_algorithm_config: DTRobustnessConfig = DTRobustnessConfig()):
        """Default constructor

        :param eval_algorithm_config: QA Accuracy eval algorithm config.
        """
        super().__init__(eval_algorithm_config)
        self._eval_algorithm_config = eval_algorithm_config

    def evaluate(
        self,
        model: Optional[ModelRunner] = None,
        dataset_config: Optional[DataConfig] = None,
        prompt_template: Optional[str] = None,
        save: bool = False,
        num_records=100,
    ) -> List[EvalOutput]:
        if dataset_config:
            dataset_configs = [dataset_config]
        else:
            dataset_configs = [DATASET_CONFIGS[dataset_name] for dataset_name in EVAL_DATASETS[self.eval_name]]

        eval_outputs: List[EvalOutput] = []
        for dataset_config in dataset_configs:
            dataset = get_dataset(dataset_config, num_records)
            validate_dataset(dataset, [TARGET_OUTPUT_COLUMN_NAME, MODEL_INPUT_COLUMN_NAME])
            dataset_prompt_template = None
            if MODEL_OUTPUT_COLUMN_NAME not in dataset.columns():
                util.require(model, "No ModelRunner provided. ModelRunner is required for inference on model_inputs")
                dataset_prompt_template = (
                    get_default_prompt_template(dataset_config.dataset_name) if not prompt_template else prompt_template
                )
                dataset = generate_prompt_column_for_dataset(
                    prompt_template=dataset_prompt_template,
                    data=dataset,
                    model_input_column_name=MODEL_INPUT_COLUMN_NAME,
                    prompt_column_name=PROMPT_COLUMN_NAME,
                )
                assert model  # to satisfy mypy
                dataset = generate_model_predict_response_for_dataset(
                    model=model,
                    data=dataset,
                    model_input_column_name=PROMPT_COLUMN_NAME,
                    model_output_column_name=MODEL_OUTPUT_COLUMN_NAME,
                    model_log_probability_column_name=MODEL_LOG_PROBABILITY_COLUMN_NAME,
                )
            with timed_block(f"Computing score and aggregation on dataset {dataset_config.dataset_name}", logger):

                def _generate_eval_scores(row: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover
                    """
                    Map function generating the scores for every input record in input dataset
                    """
                    for eval_score, eval_fn in QA_ACCURACY_SCORES_TO_FUNCS.items():
                        row[eval_score] = self._get_score(
                            target_output=row[TARGET_OUTPUT_COLUMN_NAME],
                            model_output=row[MODEL_OUTPUT_COLUMN_NAME],
                            eval_fn=eval_fn,
                        )
                    return row

                dataset = dataset.map(_generate_eval_scores).materialize()

                dataset_scores, category_scores = aggregate_evaluation_scores(
                    dataset, [DT_ROBUSTNESS_ACC], agg_method=MEAN
                )

                eval_outputs.append(
                    EvalOutput(
                        eval_name=self.eval_name,
                        dataset_name=dataset_config.dataset_name,
                        prompt_template=dataset_prompt_template,
                        dataset_scores=dataset_scores,
                        category_scores=category_scores,
                        output_path=generate_output_dataset_path(
                            path_to_parent_dir=self._eval_results_path,
                            eval_name=self.eval_name,
                            dataset_name=dataset_config.dataset_name,
                        ),
                    )
                )
            if save:
                save_dataset(
                    dataset=dataset,
                    score_names=list(QA_ACCURACY_SCORES_TO_FUNCS.keys()),
                    path=generate_output_dataset_path(
                        path_to_parent_dir=self._eval_results_path,
                        eval_name=self.eval_name,
                        dataset_name=dataset_config.dataset_name,
                    ),
                )

        return eval_outputs

    def _get_score(
        self, target_output: str, model_output: str, eval_fn: Callable[..., float], **fn_kwargs: Any
    ) -> float:
        """
        Method to generate accuracy score for a target_output and model_output

        :param target_output: Target output
        :param model_output: Model output
        :returns: Computed score
        """
        possible_targets = target_output.split(self._eval_algorithm_config.target_output_delimiter)

        return max([eval_fn(model_output, target, **fn_kwargs) for target in possible_targets])

    def evaluate_sample(self, target_output: str, model_output: List[str]) -> List[EvalScore]:  # type: ignore[override]
        """
        Evaluate a single QA record.

        :param target_output: The expected responses from the model.
        :param model_output: An instance of ModelOutput which contains the responses from the model needed for this
                             evaluation.
        :returns: A List of EvalScores computed for prompts and responses.
        """

        if target_output is None:
            raise EvalAlgorithmClientError("Missing required input: target_output, for DT robustness evaluations")
        if model_output is None:
            raise EvalAlgorithmClientError("Missing required input: model_output, for DT robustness evaluations")

        scores_ori = [EvalScore(name=eval_score, value=self._get_score(target_output=target_output, model_output=model_output[0], eval_fn=eval_fn),) for eval_score, eval_fn in QA_ACCURACY_SCORES_TO_FUNCS.items()]
        scores_adv = [EvalScore(name=eval_score, value=self._get_score(target_output=target_output, model_output=model_output[0], eval_fn=eval_fn), ) for eval_score, eval_fn in QA_ACCURACY_SCORES_TO_FUNCS.items()]

        return scores_ori, scores_adv