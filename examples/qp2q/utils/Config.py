import json
import random
import os
import numpy as np
from pathlib import Path
import logging

LOGGER = logging.getLogger(__name__)


class Config(object):
    def __init__(self, filename=None):

        self.config_name = filename

        self.base_res_dir = "../results"  # Directory where result folder is created
        self.exp_id = "0_Debug"
        self.seed = 0

        # Data specific params
        self.data_type = "aol"
        self.fdir = "../data/aol/train"  # Directory where training data files are stored
        self.f_compressed = False  # Are training data files compressed or not?

        self.misc = ""  # Suffix to append to result dir

        self.mode = "train"
        self.n_jobs = 8

        # Model specific params
        self.query_prefix_delimiter = (
            "<@@>"  # Delimiter used when concatenating query and prefix in the input
        )
        self.pref_vectorizer = "poswgtd_c-tfidf"  # Type of prefix vectorizer
        self.prefix_vect_data = "train_data"  # What data to use to train vectorizer for prefix

        self.use_label_feat = False  # Append label text features instead of PIFA from prefix when creating label embedding for indexing
        self.label_vectorizer = "use_prefix_vectorizer"  # Type of label vectorizer
        self.label_vect_data = "train_data"  # data used to train a label vectorizer

        self.indexer_type = "hierarchicalkmeans"  # Type of label indexing method to use.
        self.depth = (
            -1
        )  # This is used for depth of Trie Index, or together with must-link-constraints etc

        self.imb_depth = 100  # imbalance_depth param used for Hierarchical kmeans
        self.imb_ratio = 0.0  # imbalance_ratio param used for Hierarchical kmeans
        self.nr_splits = 2  # Branching factor param for Hierarchical kmeans

        if filename is not None:
            self.__dict__.update(json.load(open(filename)))

        self.np_seed = None
        self.update_random_seeds(self.seed)

    def to_json(self):
        return json.dumps(filter_json(self.__dict__), indent=4, sort_keys=True)

    def save_config(self, exp_dir, filename="train_config.json"):
        Path(exp_dir).mkdir(exist_ok=True, parents=True)
        with open(os.path.join(exp_dir, filename), "w") as fout:
            fout.write(self.to_json())
            fout.write("\n")

    def __getstate__(self):
        state = dict(self.__dict__)

        return state

    @property
    def result_dir(self):

        # Update model name using selected config params
        model_name = "d={d}_v={v}_i={i}_s={s}{m}".format(
            d=self.data_type,
            v=self.pref_vectorizer,
            i=self.indexer_type.lower(),
            s=self.seed,
            m="_{}".format(self.misc) if self.misc != "" else "",
        )

        LOGGER.info("Model name is = {}".format(model_name))
        result_dir = "{base}/{exp_id}/{model}".format(
            base=self.base_res_dir, exp_id=self.exp_id, model=model_name
        )

        LOGGER.info("Updated res dir is = {}".format(result_dir))
        return result_dir

    def update_random_seeds(self, random_seed):

        self.seed = random_seed
        random.seed(random_seed)

        self.np_seed = random.randint(0, 1000)
        np.random.seed(self.np_seed)


def filter_json(the_dict):
    res = {}
    for k in the_dict.keys():
        if (
            type(the_dict[k]) is str
            or type(the_dict[k]) is float
            or type(the_dict[k]) is int
            or type(the_dict[k]) is list
            or type(the_dict[k]) is bool
            or the_dict[k] is None
        ):
            res[k] = the_dict[k]
        elif type(the_dict[k]) is dict:
            res[k] = filter_json(the_dict[k])
    return res
