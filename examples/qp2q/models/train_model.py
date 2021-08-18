import gc
import sys
import logging
import argparse
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

from qp2q.preprocessing.sparse_data_processing import SparseDataFrame
from qp2q.preprocessing.session_data_processing import parallel_get_qp2q_sparse_data
from qp2q.models.pecosq2q import PecosQP2QModel
from qp2q.models.vectorizers import TfidfQueryOnly, TfidfQueryPrefix, PositionProductTfidf
from qp2q.utils.Config import Config


logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, config):
        assert isinstance(config, Config)
        self.config = config

    def train(self):

        self.config.save_config(self.config.result_dir)

        # Create sparse data frame with tranining data
        i2r, i2c, smat = parallel_get_qp2q_sparse_data(
            fdir=self.config.fdir, compressed=self.config.f_compressed, n_jobs=self.config.n_jobs
        )
        sdf = SparseDataFrame(data_matrix=smat, columns=i2c, rows=i2r)

        # Need to create sdf where col are sorted by their strings.
        # This is need only when using TrieIndexer or HybridIndexer
        sorted_labels = sorted(sdf.c2i.keys())
        sdf = sdf[:, sorted_labels]

        # Create vectorizer to vectorizer the data that is fed into PECOS models
        input_vectorizer = self.get_input_vectorizer(sdf=sdf)

        LOGGER.info("Starting to train model")
        model = PecosQP2QModel(
            vectorizer=input_vectorizer,
            query_prefix_delimiter=self.config.query_prefix_delimiter,
            load_trained_vectorizer=True,
            indexer_type=self.config.indexer_type,
        )

        LOGGER.info("Create X and Y matrices")
        X, y = model.transform(sparse_data_frame=sdf)

        label_features = None
        dim_for_PIFA = None
        if self.config.use_label_feat:

            query_vectorizer = input_vectorizer.model_query
            prefix_vectorizer = (
                input_vectorizer.model_prefix if hasattr(input_vectorizer, "model_prefix") else None
            )
            label_vectorizer = self.get_label_vectorizer(
                sdf=sdf, prefix_vectorizer=prefix_vectorizer
            )
            label_features = label_vectorizer.transform(
                [sdf.i2c[i] for i in sorted(sdf.i2c.keys())]
            )

            LOGGER.info("Label features shape:{}".format(label_features.shape))
            LOGGER.info("X query feats:{}".format(len(query_vectorizer.get_feature_names())))
            LOGGER.info("X all shape:{}".format(X.shape))

            LOGGER.info("Using only label text embedding for indexing labels (no query pifa)")
            dim_for_PIFA = 0

        # Fit a model on training data
        model = model.fit(
            X=X,
            y=y,
            label_text_features=label_features,
            dim_for_PIFA=dim_for_PIFA,
            depth=self.config.depth,
            imbalanced_depth=self.config.imb_depth,
            imbalanced_ratio=self.config.imb_ratio,
            nr_splits=self.config.nr_splits,
        )

        LOGGER.info("Finished training...")
        model.save("{}/model".format(self.config.result_dir))
        LOGGER.info(f"Successfully saved model at {self.config.result_dir}")

    def get_input_vectorizer(self, sdf):
        """
        Train input vectorizer.
        Parameters:
        ----------
        sdf: SparseDataFrame

        Returns:
        -------
        A vectorizer for vectorizing input for PECOS model
        """

        LOGGER.info("Training input vectorizer")
        ######################## Train Query Vectorizer ####################################
        query_vectorizer_config = {
            "strip_accents": "unicode",
            "ngram_range": (1, 1),
            "analyzer": "word",
            "dtype": np.float32,
        }

        query_vectorizer = TfidfVectorizer(**query_vectorizer_config)
        query_vectorizer.fit(
            list(set([q.split(self.config.query_prefix_delimiter)[0] for q in sdf.i2r.values()]))
        )
        gc.collect()
        LOGGER.info("Finished Training query vectorizer")

        if self.config.pref_vectorizer.lower() == "":
            return TfidfQueryOnly(
                model_query=query_vectorizer, delim=self.config.query_prefix_delimiter
            )
        elif (
            self.config.pref_vectorizer.lower() == "c-tfidf"
            or self.config.pref_vectorizer.lower() == "poswgtd_c-tfidf"
        ):

            ######################## Train Prefix} Vectorizer ####################################
            LOGGER.info("Training prefix vectorizer")
            prefix_vectorizer_config = {
                "strip_accents": "unicode",
                "ngram_range": (1, 3),
                "analyzer": "char",
                "dtype": np.float32,
                "use_idf": True,
            }

            if self.config.pref_vectorizer.lower() == "c-tfidf":
                prefix_vectorizer = TfidfVectorizer(**prefix_vectorizer_config)
            elif self.config.pref_vectorizer.lower() == "poswgtd_c-tfidf":
                prefix_vectorizer = PositionProductTfidf(**prefix_vectorizer_config)
            else:
                raise Exception("Invalid vectorizer {}".format(self.config.pref_vectorizer))

            if self.config.prefix_vect_data == "label_text":
                LOGGER.info("Using label text for training prefix vectorizer")
                pref_vect_train_data = sdf.i2c.values()
            elif self.config.prefix_vect_data == "label_text_all_pref":
                LOGGER.info(
                    "Using all possible prefixes of label lext for training prefix vectorizer"
                )
                pref_vect_train_data = (
                    label[:i] for label in sdf.i2c.values() for i in range(1, len(label) + 1)
                )
            elif self.config.prefix_vect_data == "train_data":
                LOGGER.info("Using prefixes in train data for training prefix vectorizer")
                pref_vect_train_data = (
                    q.split(self.config.query_prefix_delimiter)[-1] for q in sdf.i2r.values()
                )
            else:
                raise Exception(
                    f"Invalid option for vectorizing prefix = {self.config.prefix_vect_data}.\n "
                    f"Choose from label_text, label_text_all_pref, train_data"
                )

            prefix_vectorizer.fit(pref_vect_train_data)
            gc.collect()

            ############ Combine Query and Prefix Vectorizer ####################################
            joint_vectorizer = TfidfQueryPrefix(
                model_query=query_vectorizer,
                model_prefix=prefix_vectorizer,
                delim=self.config.query_prefix_delimiter,
            )
            return joint_vectorizer
        else:
            raise Exception(
                "Invalid arg for config.pref_vectorizer = {}".format(self.config.pref_vectorizer)
            )

    def get_label_vectorizer(self, sdf, prefix_vectorizer):
        """
        Get labels vectorizer.
        Parameters:
        ----------
        sdf: SparseDataFrame
        prefix_vectorizer : A trained prefix vectorizer.
                            Only used when self.config.label_vectorizer == "use_prefix_vectorizer"
        Returns:
        -------
        A vectorizer for vectorizing labels in labels space of PECOS model
        """

        if self.config.label_vectorizer == "use_prefix_vectorizer":
            LOGGER.info("Reusing prefix vectorizer to get label embeddings")
            label_vectorizer = prefix_vectorizer
        else:
            LOGGER.info(
                "Training new {} vectorizer on {} for vectorizing labels".format(
                    self.config.label_vectorizer, self.config.label_vect_data
                )
            )
            label_vectorizer_config = {
                "strip_accents": "unicode",
                "ngram_range": (1, 3),
                "analyzer": "char",
                "dtype": np.float32,
                "use_idf": True,
            }

            ############################### Choose a vectorizer class ##################################################
            if self.config.label_vectorizer.lower() == "c-tfidf":
                label_vectorizer = TfidfVectorizer(**label_vectorizer_config)
            elif self.config.label_vectorizer.lower() == "poswgtd_c-tfidf":
                label_vectorizer = PositionProductTfidf(**label_vectorizer_config)
            else:
                raise Exception(
                    "Label vectorizer = {} not supported".format(self.config.label_vectorizer)
                )

            ############################### Choose data to train the vectorizer on #############################
            if self.config.label_vect_data == "train_data":
                LOGGER.info("Using prefixes in train data for training label vectorizer")
                vect_train_data = (
                    q.split(self.config.query_prefix_delimiter)[-1] for q in sdf.i2r.values()
                )
            elif self.config.label_vect_data == "label_text":
                LOGGER.info("Using label text for training label vectorizer")
                vect_train_data = sdf.i2c.values()
            elif self.config.label_vect_data == "label_text_all_pref":
                LOGGER.info(
                    "Using all possible prefixes of label lext for training label vectorizer"
                )
                vect_train_data = (
                    label[:i] for label in sdf.i2c.values() for i in range(1, len(label) + 1)
                )
            else:
                raise Exception(
                    "Label vectorizer opt = {} not implemented ".format(self.config.label_vect_data)
                )

            label_vectorizer.fit(vect_train_data)
            gc.collect()

        return label_vectorizer


def main():
    parser = argparse.ArgumentParser(
        description="Train XMC models for next query predictions tasks"
    )
    parser.add_argument("--config", type=str, help="Train Config file")

    temp_config = Config()
    ################################## OPTIONAL ARGUMENTS TO OVERWRITE CONFIG FILE ARGS ################################
    for config_arg in temp_config.__dict__:
        def_val = temp_config.__getattribute__(config_arg)
        arg_type = type(def_val) if def_val is not None else str
        parser.add_argument(
            "--{}".format(config_arg),
            type=arg_type,
            default=None,
            help="If not specified then value from config file will be used",
        )
    ####################################################################################################################

    args = parser.parse_args()

    assert args.config is not None
    config = Config(args.config)
    for config_arg in temp_config.__dict__:
        def_val = getattr(args, config_arg)
        if def_val is not None:
            old_val = config.__dict__[config_arg]
            config.__dict__.update({config_arg: def_val})
            new_val = config.__dict__[config_arg]
            LOGGER.info(
                "Updating Config.{} from {} to {} using arg_val={}".format(
                    config_arg, old_val, new_val, def_val
                )
            )

    Path(config.result_dir).mkdir(
        parents=True, exist_ok=True
    )  # Create resultDir directory if not already present
    config.update_random_seeds(config.seed)
    config.save_config(config.result_dir)

    trainer = Trainer(config)
    if config.mode == "train":
        trainer.train()
    else:
        raise Exception("Invalid mode = {}.".format(config.mode))


if __name__ == "__main__":
    main()
