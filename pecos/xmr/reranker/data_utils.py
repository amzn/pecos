import bisect
import os
import random
from collections import OrderedDict
from typing import List, Tuple, Callable

import numpy as np
import pyarrow.parquet as pq
from datasets import load_dataset

import pecos


def get_pairwise_batch(
    batch,
    inp_col,
    lbl_col,
    rel_col,
    group_size,
    pos_label_sampling="weighted",
):
    """
    Convert listwise batch into pairwise batch by sampling two entrees
    with distict rel_col per input

    Example:
        batch = {
            "inp_col": ["q1", "q2"],
            "lbl_col": [
                ["p11", "p12", "p13"],
                ["p21", "p22", "p23", "p24"],
            ],
            "rel_col": [
                [1.0, 0.5, 0.1],
                [0.9, 0.9, 0.7, 0.6],
            ],
        }
        pairwise_batch = {
            "inp_col": ["q1", "q1", "q2", "q2"],
            "lbl_col": ["a11", "a13", "a22", "a23"],
            "rel_col": [1.0, 0.1, 0.9, 0.7],
        }

    """

    def sample_unequal_pair_from_array(data):
        "sample a pair of indices from a decent sorted array"
        non_min_sum = len(data) - 1
        while data[non_min_sum] == data[-1]:
            non_min_sum -= 1
        # assert non_min_sum > 0, (non_min_sum, data)

        if pos_label_sampling == "weighted":
            psum = np.sum(data[: (non_min_sum + 1)])
            pos_idx = np.random.choice(range(non_min_sum + 1), p=data[: (non_min_sum + 1)] / psum)
        elif pos_label_sampling == "uniform":
            pos_idx = np.random.randint(non_min_sum + 1)
        else:
            raise NotImplementedError(f"Unknown positive sampling method {pos_label_sampling}")

        # neg_num = np.sum(data < data[pos_idx])
        neg_start = pos_idx
        while data[neg_start] == data[pos_idx]:
            neg_start += 1

        # assert neg_num > 0, (neg_num, data)
        neg_idx = np.random.randint(neg_start, len(data))
        return pos_idx, neg_idx

    if group_size != 2:
        raise ValueError(f"pairwise batch sampling assumes group_size==2")

    pair_indices = [sample_unequal_pair_from_array(scores) for scores in batch[rel_col]]
    pos_lbls = [alist[pa[0]] for alist, pa in zip(batch[lbl_col], pair_indices)]
    neg_lbls = [alist[pa[1]] for alist, pa in zip(batch[lbl_col], pair_indices)]
    pos_rels = [slist[pa[0]] for slist, pa in zip(batch[rel_col], pair_indices)]
    neg_rels = [slist[pa[1]] for slist, pa in zip(batch[rel_col], pair_indices)]

    return {
        inp_col: np.repeat(batch[inp_col], group_size),
        lbl_col: np.vstack([pos_lbls, neg_lbls]).T.flatten(),
        rel_col: np.vstack([pos_rels, neg_rels]).T.flatten(),
    }


def get_listwise_batch(
    batch,
    inp_col,
    lbl_col,
    rel_col,
    group_size,
    pos_label_sampling="weighted",
    neg_rel_val=0.0,
):
    """
    Convert listwise batch into sampled listwise batch
    we assume the group_size be smaller then len(lbl_col[i]), for all i

    Example:
        batch = {
            "inp_col": ["q1", "q2"],
            "lbl_col": [
                ["p11", "p12", "p13", "p14"],
                ["p21", "p22", "a23", "p24", "p25"],
            ],
            "rel_col": [
                [0.8, 0.5, 0.0, 0.0],
                [0.9, 0.0, 0.0, 0.0, 0.0],
            ],
        }

        sampled listwise batch (group_size=3)
        listwise_batch = {
            "inp_col": ["q1", "q1", "q1", "q2", "q2", "q2"],
            "lbl_col": ["p12", "p14", "p13", "p21", "q23", "q25"],
            "rel_col": [0.5, 0.0, 0.0, 0.9, 0.0, 0.0],
        }
    """
    if group_size < 2:
        raise ValueError("listwise batch sampling assumes group_size >= 2")

    all_lbl_list, all_rel_list = [], []
    for lbl_arr, rel_arr in zip(batch[lbl_col], batch[rel_col]):
        # note that bisect assumes lbl_arr/rel_arr are sorted ascendingly (by values in rel_arr)
        # so we add negative sign to flip the order from descendingly to ascendingly,
        # and find rightmost index (i.e., pos_ptr) whose value less than -neg_label_val.
        # see https://docs.python.org/3/library/bisect.html
        pos_ptr = bisect.bisect_left(-rel_arr, -neg_rel_val)

        # sample 1 positive
        indices = np.random.randint(0, high=pos_ptr, size=1).tolist()

        # smaple group_size - 1 negatives
        num_true_neg = min(len(lbl_arr) - pos_ptr, group_size - 1)
        if num_true_neg > 0:
            indices += np.random.randint(pos_ptr, high=len(lbl_arr), size=num_true_neg).tolist()
        num_rand_neg = (group_size - 1) - num_true_neg
        if num_rand_neg > 0:
            assert NotImplementedError(f"within batch negative not support for Listwise Ranking!")

        all_lbl_list.append(lbl_arr[indices].tolist())
        all_rel_list.append(rel_arr[indices].tolist())
    # end for loop
    return {
        inp_col: np.repeat(batch[inp_col], group_size),
        lbl_col: np.vstack(all_lbl_list).flatten(),
        rel_col: np.vstack(all_rel_list).flatten(),
    }


class RankingDataUtils(pecos.BaseClass):
    """
    Utility class for handling data related tasks
    """

    @classmethod
    def remap_ordereddict(cls, od: OrderedDict, keymap_fn: Callable):
        """
        Function to remap the keys of an ordered Dictionary
        Args:
            od: The ordered dictionary to remap
            keymap_fn: The function to map the keys
        """
        new_od = OrderedDict()
        for k, v in od.items():
            new_od[keymap_fn(k)] = v
        return new_od

    @classmethod
    def _format_sample(
        cls,
        inp_text: str,
        lbl_contents: List[str],
        inp_prefix: str = "...",
        passage_prefix: str = "...",
        content_sep=" ",
    ) -> str:
        """
        Function to convert the text fields into a formatted string
        that the model understands.
        Args:
            inp_text: The input text
            lbl_contents: The list of content fields
            inp_prefix: The input prefix
            passage_prefix: The passage prefix
            content_sep: The separator between the content fields
        Returns: The formatted string
        """
        # Convention from rankllama is to replace hyphens in the title
        lbl_contents[0] = lbl_contents[0].replace("-", " ").strip()
        return f"{inp_prefix} {inp_text} {passage_prefix} {content_sep.join(lbl_contents)}".strip()

    @classmethod
    def _create_sample(
        cls,
        inp_id: int,
        ret_idxs: List[int],
        scores: List[float],
        table_stores,
        group_size: int,
        inp_prefix: str,
        passage_prefix: str,
        keyword_col_name: str,
        content_col_names: List[str],
        content_sep,
    ) -> Tuple[List[str], List[float]]:
        """
        Function to create a sample for training.
        Args:
            inp_id: The input id
            ret_idxs: The retrieved indices
            scores: Scores for the retrieved indices
            table_stores: Dictionary of table stores for input and label data
            group_size: The number of passages used to train for each query
            inp_prefix: The input prefix
            passage_prefix: The passage prefix
            keyword_col_name: The column name for the query text
            content_col_names: The column names for the content fields
            content_sep: The separator between the content fields
        Returns: A tuple of formatted samples and scores
        """
        qid = inp_id
        pidxs = ret_idxs

        input_store = table_stores["input"]
        label_store = table_stores["label"]

        # get the values of the query
        query = input_store[qid][keyword_col_name]
        mean_score = np.mean(scores)

        # get idxs for positive items
        pos_idxs = [(x, pid) for x, pid in zip(scores, pidxs) if x > mean_score]
        neg_idxs = [(x, pid) for x, pid in zip(scores, pidxs) if x <= mean_score]
        random.shuffle(pos_idxs)
        random.shuffle(neg_idxs)

        num_positives = group_size // 2

        all_selections = pos_idxs[:num_positives]
        num_positives = len(all_selections)
        num_negatives = group_size - num_positives
        all_selections.extend(neg_idxs[:num_negatives])

        if len(all_selections) < group_size:
            all_selections.extend(random.choices(neg_idxs, k=group_size - len(all_selections)))

        all_scores = [s for s, _ in all_selections]
        all_pids = [pid for _, pid in all_selections]

        # get the values for the retrieved items
        ret_info = [label_store[i] for i in all_pids]

        formated_pair = []
        for info in ret_info:
            formated_pair.append(
                cls._format_sample(
                    query,
                    [info[c] for c in content_col_names],
                    inp_prefix,
                    passage_prefix,
                    content_sep,
                )
            )
        return formated_pair, all_scores

    @classmethod
    def get_parquet_rows(cls, folder_path: str) -> int:
        """
        Returns the count of rows in parquet files by reading the
        metadata
        Args:
            folder_path: The folder containing the parquet files
        Returns: The count of rows in the parquet files
        """
        file_list = os.listdir(folder_path)
        file_list = [os.path.join(folder_path, x) for x in file_list]
        cumulative_rowcount = sum([pq.read_metadata(fp).num_rows for fp in file_list])

        return cumulative_rowcount

    @classmethod
    def get_sorted_data_files(cls, filenames: List[str], idx_colname) -> List[str]:
        """
        Returns the list of files sorted by the id in the first row of each file
        Args:
            filenames: The list of filenames
            idx_colname: The column name of the id
        Returns: The sorted list of filenames
        """
        # Load the datasets in streaming format and read the first id
        fn_ordered = []  # this containes tuples with (idx, filename)
        for fn in filenames:
            tmp_ds = load_dataset("parquet", data_files=fn, streaming=True, split="train")
            row = next(iter(tmp_ds.take(1)))
            fn_ordered.append((row[idx_colname], fn))
            del tmp_ds
        fn_ordered = sorted(fn_ordered, key=lambda x: x[0])

        return [x[1] for x in fn_ordered]
