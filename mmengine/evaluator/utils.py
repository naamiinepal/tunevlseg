# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any


def get_metric_value(indicator: str, metrics: dict) -> Any:
    """Get the metric value specified by an indicator, which can be either a
    metric name or a full name with evaluator prefix.

    Args:
        indicator (str): The metric indicator, which can be the metric name
            (e.g. 'AP') or the full name with prefix (e.g. 'COCO/AP')
        metrics (dict): The evaluation results output by the evaluator

    Returns:
        Any: The specified metric value
    """

    if "/" in indicator:
        # The indicator is a full name
        if indicator in metrics:
            return metrics[indicator]
        raise ValueError(
            f'The indicator "{indicator}" can not match any metric in '
            f"{list(metrics.keys())}"
        )
    # The indicator is metric name without prefix
    matched = [k for k in metrics if k.split("/")[-1] == indicator]

    if not matched:
        raise ValueError(
            f"The indicator {indicator} can not match any metric in "
            f"{list(metrics.keys())}"
        )
    if len(matched) > 1:
        raise ValueError(
            f'The indicator "{indicator}" matches multiple ' f"metrics {matched}"
        )
    return metrics[matched[0]]
