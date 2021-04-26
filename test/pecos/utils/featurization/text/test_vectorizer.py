#  Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
#  with the License. A copy of the License is located at
#
#  http://aws.amazon.com/apache2.0/
#
#  or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
#  OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
#  and limitations under the License.
import pytest  # noqa: F401; pylint: disable=unused-variable

import numpy as np
from pecos.utils import smat_util
from pecos.utils.featurization.text.vectorizers import Vectorizer
from pytest import approx


def check_result_of_vectorizer_config(corpus, X, config, error_msg=""):
    vect = Vectorizer.train(corpus, config=config)
    Xp = vect.predict(corpus).toarray()
    assert X == approx(Xp), error_msg


def test_tfidf_vectorizer_tf():
    corpus = [
        "midas amazon pecos",
        "amazon pecos pecos pecos",
        "iphone",
    ]

    # config: {'use_idf': False, 'smooth_idf': False, norm: 'l2'}
    # ground TF matrix and IDF array manually computed by
    # tf(t,d) = f_{t,d} / sum_{t'} f_{t',d}
    # idf(t,D) = log(N/n_t)
    # more details at https://en.wikipedia.org/wiki/Tf%E2%80%93idf

    # feat_dim = {0: "iphone", 1: "midas", 2: "amazon", 3: "pecos"}
    X_1 = np.array(
        [
            [0.00000000, 0.57735026, 0.57735026, 0.57735026],
            [0.00000000, 0.00000000, 0.31622776, 0.94868326],
            [1.00000000, 0.00000000, 0.00000000, 0.00000000],
        ]
    )
    # feat_dim = {0: "midas amazon", 1: "pecos pecos", 2: "amzon pecos"}
    X_2 = np.array(
        [
            [0.70710677, 0.00000000, 0.70710677],
            [0.00000000, 0.89442720, 0.44721360],
            [0.00000000, 0.00000000, 0.00000000],
        ]
    )
    # stack X_1 and X_2 column-wise
    X_3 = np.array(
        [
            [0.00000000, 0.40824829, 0.40824829, 0.40824829, 0.50000000, 0.00000000, 0.50000000],
            [0.00000000, 0.00000000, 0.22360680, 0.67082038, 0.00000000, 0.63245555, 0.31622777],
            [1.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
        ]
    )

    # ==== test case 1 ====
    # build from c++ tfidf word unigram (TfidfBaseVectorizerParam)
    config = {
        "type": "tfidf",
        "kwargs": {
            "use_idf": False,
            "smooth_idf": False,
            "norm": "l2",
        },
    }
    check_result_of_vectorizer_config(corpus, X_1, config, error_msg="word unigram config")

    # ==== test case 2 ====
    # build from c++ tfidf word bigram (TfidfVectorizerParam)
    config = {
        "type": "tfidf",
        "kwargs": {
            "norm_p": 2,
            "threads": 4,
            "base_vect_configs": [
                {
                    "ngram_range": [2, 2],
                    "use_idf": False,
                    "smooth_idf": False,
                    "norm": "l2",
                    "analyzer": "word",
                }
            ],
        },
    }
    check_result_of_vectorizer_config(corpus, X_2, config, error_msg="word bigram config")

    # ==== test case 3 ====
    # build from c++ tfidf word bigram (TfidfVectorizerParam)
    config = {
        "type": "tfidf",
        "kwargs": {
            "norm_p": 2,
            "threads": 4,
            "base_vect_configs": [
                {
                    "ngram_range": [1, 1],
                    "use_idf": False,
                    "smooth_idf": False,
                    "norm": "l2",
                    "analyzer": "word",
                },
                {
                    "ngram_range": [2, 2],
                    "use_idf": False,
                    "smooth_idf": False,
                    "sublinear_tf": False,
                    "norm": "l2",
                    "analyzer": "word",
                },
            ],
        },
    }
    check_result_of_vectorizer_config(
        corpus, X_3, config, error_msg="word [unigram, bigram] config"
    )

    # ==== test case 4, online prediction ====
    config = {
        "type": "tfidf",
        "kwargs": {
            "norm_p": 2,
            "threads": 4,
            "base_vect_configs": [
                {
                    "ngram_range": [1, 1],
                    "use_idf": False,
                    "smooth_idf": False,
                    "norm": "l2",
                    "analyzer": "word",
                },
                {
                    "ngram_range": [2, 2],
                    "use_idf": False,
                    "smooth_idf": False,
                    "norm": "l2",
                    "analyzer": "word",
                },
            ],
        },
    }
    vect = Vectorizer.train(corpus, config=config)
    Xp_4 = [vect.predict([line]) for line in corpus]
    Xp_4 = smat_util.vstack_csr(Xp_4).toarray()
    assert X_3 == approx(Xp_4), "online prediction failed"


def test_tfidf_vectorizer_idf():
    corpus = [
        "midas amazon pecos",
        "amazon pecos pecos pecos",
        "iphone",
    ]

    # config: {'use_idf': True, 'smooth_idf': False, norm: 'l2'}
    # ground TF matrix and IDF array manually computed by
    # tf(t,d) = f_{t,d} / sum_{t'} f_{t',d}
    # idf(t,D) = log(N/n_t)
    # more details at https://en.wikipedia.org/wiki/Tf%E2%80%93idf

    # feat_dim = {0: "iphone", 1: "midas", 2: "amazon", 3: "pecos"}
    X_1 = np.array(
        [
            [0.00000000, 0.88651040, 0.32718456, 0.32718456],
            [0.00000000, 0.00000000, 0.31622776, 0.94868330],
            [1.00000000, 0.00000000, 0.00000000, 0.00000000],
        ]
    )
    # feat_dim = {0: "midas amazon", 1: "pecos pecos", 2: "amzon pecos"}
    X_2 = np.array(
        [
            [0.93814546, 0.00000000, 0.34624156],
            [0.00000000, 0.98339630, 0.18147115],
            [0.00000000, 0.00000000, 0.00000000],
        ]
    )
    # stack X_1 and X_2 column-wise
    X_3 = np.array(
        [
            [0.00000000, 0.62685746, 0.23135442, 0.23135442, 0.66336906, 0.00000000, 0.24482976],
            [0.00000000, 0.00000000, 0.22360680, 0.67082040, 0.00000000, 0.69536614, 0.12831947],
            [1.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
        ]
    )

    # ==== test case 1 ====
    # build from c++ tfidf word unigram (TfidfBaseVectorizerParam)
    config = {
        "type": "tfidf",
        "kwargs": {
            "use_idf": True,
            "smooth_idf": False,
            "norm": "l2",
        },
    }
    check_result_of_vectorizer_config(corpus, X_1, config, error_msg="word unigram config")

    # ==== test case 2 ====
    # build from c++ tfidf word bigram (TfidfVectorizerParam)
    config = {
        "type": "tfidf",
        "kwargs": {
            "norm_p": 2,
            "threads": 4,
            "base_vect_configs": [
                {
                    "ngram_range": [2, 2],
                    "use_idf": True,
                    "smooth_idf": False,
                    "norm": "l2",
                    "analyzer": "word",
                }
            ],
        },
    }
    check_result_of_vectorizer_config(corpus, X_2, config, error_msg="word bigram config")

    # ==== test case 3 ====
    # build from c++ tfidf word bigram (TfidfVectorizerParam)
    config = {
        "type": "tfidf",
        "kwargs": {
            "norm_p": 2,
            "threads": 4,
            "base_vect_configs": [
                {
                    "ngram_range": [1, 1],
                    "use_idf": True,
                    "smooth_idf": False,
                    "norm": "l2",
                    "analyzer": "word",
                },
                {
                    "ngram_range": [2, 2],
                    "use_idf": True,
                    "smooth_idf": False,
                    "norm": "l2",
                    "analyzer": "word",
                },
            ],
        },
    }
    check_result_of_vectorizer_config(
        corpus, X_3, config, error_msg="word [unigram, bigram] config"
    )

    # ==== test case 4, online prediction ====
    config = {
        "type": "tfidf",
        "kwargs": {
            "norm_p": 2,
            "threads": 4,
            "base_vect_configs": [
                {
                    "ngram_range": [1, 1],
                    "use_idf": True,
                    "smooth_idf": False,
                    "norm": "l2",
                    "analyzer": "word",
                },
                {
                    "ngram_range": [2, 2],
                    "use_idf": True,
                    "smooth_idf": False,
                    "norm": "l2",
                    "analyzer": "word",
                },
            ],
        },
    }
    vect = Vectorizer.train(corpus, config=config)
    Xp_4 = [vect.predict([line]) for line in corpus]
    Xp_4 = smat_util.vstack_csr(Xp_4).toarray()
    assert X_3 == approx(Xp_4), "online prediction failed"
