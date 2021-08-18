import gc
import json
import logging
from abc import ABCMeta, abstractmethod
from typing import Any, ClassVar, Dict

from qp2q.models import pecosq2q

LOGGER = logging.getLogger(__name__)


class WorkerMeta(ABCMeta):
    """Collects all eval_helpers in cls.subtypes automatically,
    when AbstractWorker is inherited.

    Notes:
    -----
    This is a meta-class:
    https://docs.python.org/3/library/abc.html#abc.ABCMeta
    """

    subtypes: ClassVar[
        Dict[str, Any]
    ] = {}  # static variable for storing classnames->class for eval_helpers

    def __new__(cls, name, bases, attr):
        cls = super().__new__(cls, name, bases, attr)
        if cls.__name__ != "AbstractWorker":
            WorkerMeta.subtypes[cls.__name__] = cls
        return cls


class AbstractWorker(metaclass=WorkerMeta):
    """
    An abstract class for different evaluators for different types of models.

    Any class that inherits AbstractWorker will be available in EVAL_HELPER_DICT,
    with key as class name and value as the class itself.
    """

    def __init__(self, topk):
        """
        Initialize AbstractWorker class.

        Parameters
        ----------
        topk : int
            number of suggestions to return
        """
        if (not topk) or (not isinstance(topk, int)) or (topk < 0):
            raise TypeError("`topk` should be a positive integer")

        self.topk = topk

    @abstractmethod
    def get_suggestions(self, **kwargs):
        """
        Abstract method to get suggestions.

        Not Implemented

        """
        raise NotImplementedError()

    @staticmethod
    def _return_results(suggestions_with_score):
        """
        Return topk results.

        Parameters
        ----------
        suggestions_with_score : iterator
            iterator of tuples where first item suggestion
            and the second item is its score
        Returns
        -------
        list
           A list of suggestions

        """
        return [suggestion_with_score[0] for suggestion_with_score in suggestions_with_score]


class PecosSuggester(AbstractWorker):
    """Generate Suggestions using PECOS Auto-complete suggestion model"""

    def __init__(
        self,
        model_path,
        beam_size,
        topk,
    ):
        """
        Initialize suggestor.

        Parameters
        ----------
        model_path: str
            path containing saved model file(s)
        beam_size: int
            beam_size to use in generating suggestions with PECOS models
        topk: int
            number of suggestions to return

        """
        super(PecosSuggester, self).__init__(topk=topk)
        self.beam_size = beam_size
        self.pecos_model = pecosq2q.PecosQP2QModel.load(model_path, realtime=True)
        gc.collect()

    def get_suggestions(self, request):
        """
        Return next query suggestions

        Parameters
        ----------
        request: dict with keys = prefix and prev_query

        Returns
        -------
        list
            list of next query suggestions

        """
        prefix = request["prefix"]
        prev_query = request["prev_query"]
        responses = self.pecos_model.get_suggestions(
            prev_query=prev_query,
            prefix=prefix,
            topk=self.topk,
            beam_size=self.beam_size,
        )

        responses = [(r[0], r[1]) for r in responses[: self.topk]]
        return self._return_results(responses)


class PrefFreqSuggester(AbstractWorker):
    """
    Generate Suggestions by returning most frequent queries matching the prefix
    It uses a dictionary to retrieve pre-computed list of queries matching a prefix
    If there are not enough queries matching the prefix then it fetches queries
    matching a smaller prefix (removing one character at a time).
    Queries retrieve using smaller prefix are appended to list of queries that exactly match the prefix.
    """

    def __init__(self, model_path, topk):
        """
        Initialize suggestor.

        Parameters
        ----------
        model_path: str
            path containing saved model file
        topk: int
            number of suggestions to return
        """
        super(PrefFreqSuggester, self).__init__(topk=topk)
        LOGGER.info("Loading model from file")
        with open(model_path, "r") as f:
            self.pref_to_topk = json.load(f)
        LOGGER.info("Finished loading model from file")

    def _get_suggestions(self, prefix):
        """

        Parameters
        ----------
        prefix

        Returns a list of 2-tuple where first value is label and second is its frequency
        -------

        """
        if prefix in self.pref_to_topk:
            if len(self.pref_to_topk[prefix]) >= self.topk:
                return self.pref_to_topk[prefix][: self.topk]
            elif len(prefix) > 0:
                more_labels = self._get_suggestions(prefix=prefix[:-1])
                perfect_match_labels, _ = zip(*self.pref_to_topk[prefix])
                perfect_match_labels = set(perfect_match_labels)
                all_labels = self.pref_to_topk[prefix] + [
                    (l, f) for (l, f) in more_labels[: self.topk] if l not in perfect_match_labels
                ]
                return all_labels[: self.topk]
            else:
                return self.pref_to_topk[prefix][: self.topk]
        elif len(prefix) > 0:
            """
            In case prefix does not have top-k suggestions already computed, and  prefix is greater than 1 char,
            then suggest using prefix[:-1],  i.e after removing 1 char from prefix
            """
            return self._get_suggestions(prefix=prefix[:-1])
        else:
            return []

    def get_suggestions(self, request):
        """
        Return next query suggestions

        Parameters
        ----------
        request: dict with keys = prefix and prev_query

        Returns
        -------
        list
            list of next query suggestions
        """
        prefix = request["prefix"]
        responses = self._get_suggestions(prefix=prefix)

        return self._return_results(responses)
