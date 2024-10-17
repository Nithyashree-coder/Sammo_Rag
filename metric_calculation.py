import re
import string
import collections
from typing import List
from sammo.base import EvaluationScore
from sammo.data import DataTable

class TextProcessor:
    """
    Class for processing text data.
    """
    @staticmethod
    def get_tokens(s: str) -> List[str]:
        '''
        Get tokens from the input string.

        Parameters:
        ----------
        s : str
            Input string.

        Returns:
        -------
        List[str]:
            List of tokens extracted from the input string.
        '''
        if not s:
            return []
        return TextProcessor.normalize_answer(s).split()

    @staticmethod
    def normalize_answer(s: str) -> str:
        """
        Lower text and remove punctuation, articles, and extra whitespace.

        Parameters:
        ----------
        s : str
            Input string.

        Returns:
        -------
        str:
            Normalized string.
        """
        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

class F1ScoreCalculator:
    """
    Class for calculating F1 score.
    """
    @staticmethod
    def compute_f1(a_gold: str, a_pred: str) -> float:
        """
        Compute F1 score between two strings.

        Parameters:
        ----------
        a_gold : str
            True answer string.
        a_pred : str
            Predicted answer string.

        Returns:
        -------
        float:
            F1 score.
        """
        gold_toks = TextProcessor.get_tokens(a_gold)
        pred_toks = TextProcessor.get_tokens(a_pred)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())

        if len(gold_toks) == 0 or len(pred_toks) == 0:
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0.0

        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    @staticmethod
    def compute_f1_score(y_true: 'DataTable', y_pred: 'DataTable') -> 'EvaluationScore':
        """
        Compute F1 score for a set of true and predicted values.

        Parameters:
        ----------
        y_true : DataTable
            True values.
        y_pred : DataTable
            Predicted values.

        Returns:
        -------
        EvaluationScore:
            Evaluation score containing the computed F1 score.
        """
        y_true_normalized = y_true.outputs.normalized_values()
        y_pred_normalized = y_pred.outputs.normalized_values()

        if len(y_true_normalized) != len(y_pred_normalized):
            raise ValueError("The lengths of true and predicted values do not match.")

        total_f1 = 0.0
        for true, pred in zip(y_true_normalized, y_pred_normalized):
            total_f1 += F1ScoreCalculator.compute_f1(true, pred)

        avg_f1 = total_f1 / len(y_true_normalized)
        return EvaluationScore(avg_f1)