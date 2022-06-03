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
from pytest import approx


def test_importable():
    import pecos.xmc.xlinear  # noqa: F401
    from pecos.xmc.xlinear import XLinearModel  # noqa: F401
    from pecos.xmc import Indexer  # noqa: F401


def test_cost_sensitive(tmpdir):
    from pecos.utils import smat_util
    from pecos.xmc import MLProblem, MLModel

    train_X_file = "test/tst-data/xmc/xlinear/X.npz"
    train_Y_file = "test/tst-data/xmc/xlinear/Y.npz"

    X = smat_util.load_matrix(train_X_file)
    Y = smat_util.load_matrix(train_Y_file)

    Cp = 2.0
    R = smat_util.binarized(Y)
    R.data = Cp * R.data

    # test MLModel
    # Cp=2.0 and R=none should equiv to Cp=1.0 and R=2.0
    model_v0 = MLModel.train(
        MLProblem(X, Y, C=None, M=None, R=None),
        train_params=MLModel.TrainParams(Cp=Cp),
    )

    model_v1 = MLModel.train(
        MLProblem(X, Y, C=None, M=None, R=R),
        train_params=MLModel.TrainParams(Cp=1.0),
    )
    assert model_v0.W.todense() == approx(model_v1.W.todense(), abs=1e-9)

    # test XLinearModel
    # test data has one positve label per instance
    # therefore Cp=2.0 and R=none should still equiv to Cp=1.0 and R=2.0
    from pecos.xmc.xlinear import XLinearModel
    from pecos.xmc import Indexer, LabelEmbeddingFactory

    label_feat = LabelEmbeddingFactory.create(Y, X, method="pifa")
    cluster_chain = Indexer.gen(label_feat, max_leaf_size=2)
    xlm_v0 = XLinearModel.train(
        X,
        Y,
        C=cluster_chain,
        R=R,
        train_params={"rel_norm": "no-norm", "rel_mode": "induce"},
    )
    xlm_v1 = XLinearModel.train(
        X,
        Y,
        C=cluster_chain,
        train_params={"hlm_args": {"model_chain": {"Cp": Cp}}},
    )

    for d in range(len(cluster_chain)):
        assert xlm_v0.model.model_chain[d].W.todense() == approx(
            xlm_v1.model.model_chain[d].W.todense()
        )

    # test CLI
    import subprocess
    import shlex
    from pecos.xmc import Indexer, LabelEmbeddingFactory
    from pecos.utils import smat_util

    model_folder = str(tmpdir.join("save_model"))
    rel_mat_file = str(tmpdir.join("R.npz"))
    smat_util.save_matrix(rel_mat_file, R)

    cmd = []
    cmd += ["python3 -m pecos.xmc.xlinear.train"]
    cmd += ["-x {}".format(train_X_file)]
    cmd += ["-y {}".format(train_Y_file)]
    cmd += ["-m {}".format(model_folder)]
    cmd += ["-r {}".format(rel_mat_file)]
    cmd += ["--max-leaf-size {}".format(2)]
    cmd += ["--rel-norm {}".format("no-norm")]
    cmd += ["--rel-mode {}".format("induce")]
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.returncode == 0, " ".join(cmd)
    xlm_v2 = XLinearModel.load(model_folder)

    for d in range(len(cluster_chain)):
        assert xlm_v0.model.model_chain[d].W.todense() == approx(
            xlm_v2.model.model_chain[d].W.todense()
        )


def test_predict_consistency_between_python_and_cpp(tmpdir):
    import subprocess
    import shlex
    from pecos.xmc import PostProcessor
    from pecos.xmc.xlinear import XLinearModel as py_xlm

    train_X_file = "test/tst-data/xmc/xlinear/X.npz"
    train_Y_file = "test/tst-data/xmc/xlinear/Y.npz"
    test_X_file = "test/tst-data/xmc/xlinear/Xt.npz"
    model_folder = str(tmpdir.join("save_model"))

    models = []

    # Obtain a xlinear model
    cmd = []
    cmd += ["python3 -m pecos.xmc.xlinear.train"]
    cmd += ["-x {}".format(train_X_file)]
    cmd += ["-y {}".format(train_Y_file)]
    cmd += ["-m {}".format(model_folder)]
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.returncode == 0, " ".join(cmd)
    models += [model_folder]

    # Obtain xlinear models with vairous number of splits
    for splits in [2, 4]:
        model_folder_local = f"{model_folder}-{splits}"
        cmd = []
        cmd += ["python3 -m pecos.xmc.xlinear.train"]
        cmd += [f"-x {train_X_file}"]
        cmd += [f"-y {train_Y_file}"]
        cmd += [f"--nr-splits {splits}"]
        cmd += [f"--max-leaf-size 2"]
        cmd += [f"-m {model_folder_local}"]
        process = subprocess.run(
            shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert process.returncode == 0, " ".join(cmd)
        models += [model_folder_local]

    X = py_xlm.load_feature_matrix(test_X_file)
    for model in models:
        py_m = py_xlm.load(model)
        py_bin_search_m = py_xlm.load(
            model, is_predict_only=True, weight_matrix_type="BINARY_SEARCH_CHUNKED"
        )
        py_hash_m = py_xlm.load(model, is_predict_only=True, weight_matrix_type="HASH_CHUNKED")
        py_csc_m = py_xlm.load(model, is_predict_only=True, weight_matrix_type="CSC")

        for pp in PostProcessor.valid_list():
            kwargs = {"post_processor": pp, "beam_size": 2}

            # in batch mode

            py_pred = py_m.predict(X, **kwargs).todense()

            # Sparse inputs
            # Test csr_t x bin_search_chunked_matrix_t
            py_bin_search_pred = py_bin_search_m.predict(X, **kwargs).todense()
            # Test csr_t x hash_chunked_matrix_t
            py_hash_pred = py_hash_m.predict(X, **kwargs).todense()
            # Test csr_t x csc_t
            py_csc_pred = py_csc_m.predict(X, **kwargs).todense()

            # Dense inputs
            # Test drm_ x binary search chunked
            py_bin_search_dense_pred = py_bin_search_m.predict(X.todense(), **kwargs).todense()
            # Test drm_t x hash chunked
            py_hash_chunked_dense_pred = py_hash_m.predict(X.todense(), **kwargs).todense()
            # Test drm_t x csc_t
            py_csc_dense_pred = py_csc_m.predict(X.todense(), **kwargs).todense()

            assert py_bin_search_pred == approx(
                py_pred, abs=1e-6
            ), f"model:{model} (sparse, bin-search) post_processor:{pp}"
            assert py_hash_pred == approx(
                py_pred, abs=1e-6
            ), f"model:{model} (sparse, hash) post_processor:{pp}"
            assert py_csc_pred == approx(
                py_pred, abs=1e-6
            ), f"model:{model} (sparse, csc) post_processor:{pp}"

            assert py_bin_search_dense_pred == approx(
                py_pred, abs=1e-6
            ), f"model:{model} (dense, bin-search) post_processor:{pp}"
            assert py_hash_chunked_dense_pred == approx(
                py_pred, abs=3e-6
            ), f"model:{model} (dense, hash) post_processor:{pp}"
            assert py_csc_dense_pred == approx(
                py_pred, abs=1e-6
            ), f"model:{model} (dense, csc) post_processor:{pp}"

            # in realtime mode
            for i in range(X.shape[0]):
                query_slice = X[[i], :]
                # Some versions of Scipy don't maintain sortedness when slicing
                query_slice.sort_indices()

                py_pred = py_m.predict(query_slice, **kwargs).todense()

                # Sparse Inputs
                # Test csr_t x bin_search_chunked_matrix_t
                py_bin_search_pred = py_bin_search_m.predict(query_slice, **kwargs).todense()
                # Test csr_t x hash_chunked_matrix_t
                py_hash_pred = py_hash_m.predict(query_slice, **kwargs).todense()
                # Test csr_t x csc_t
                py_csc_pred = py_csc_m.predict(query_slice, **kwargs).todense()

                # Dense Inputs
                # Test drm_ x binary search chunked
                py_bin_search_dense_pred = py_bin_search_m.predict(
                    query_slice.todense(), **kwargs
                ).todense()
                # Test drm_t x hash chunked
                py_hash_chunked_dense_pred = py_hash_m.predict(
                    query_slice.todense(), **kwargs
                ).todense()
                # Test csr_t x csc_t
                py_csc_dense_pred = py_csc_m.predict(query_slice.todense(), **kwargs).todense()

                assert py_bin_search_pred == approx(
                    py_pred, abs=1e-6
                ), f"model:{model} (sparse, bin-search) post_processor:{pp}, inst:{i}"
                assert py_hash_pred == approx(
                    py_pred, abs=1e-6
                ), f"model:{model} (sparse, hash) post_processor:{pp}, inst:{i}"
                assert py_csc_pred == approx(
                    py_pred, abs=1e-6
                ), f"model:{model} (sparse, csc) post_processor:{pp}, inst:{i}"

                assert py_bin_search_dense_pred == approx(
                    py_pred, abs=1e-6
                ), f"model:{model} (dense, bin-search) post_processor:{pp}, inst:{i}"
                assert py_hash_chunked_dense_pred == approx(
                    py_pred, abs=3e-6
                ), f"model:{model} (dense, hash) post_processor:{pp}, inst:{i}"
                assert py_csc_dense_pred == approx(
                    py_pred, abs=1e-6
                ), f"model:{model} (dense, csc) post_processor:{pp}, inst:{i}"


def test_consistency_of_primal(tmpdir):
    import subprocess
    import shlex
    import numpy as np
    from pecos.utils import smat_util

    train_sX_file = "test/tst-data/xmc/xlinear/X.npz"
    train_dX_file = str(tmpdir.join("X.trn.npy"))
    train_Y_file = "test/tst-data/xmc/xlinear/Y.npz"
    test_sX_file = "test/tst-data/xmc/xlinear/Xt.npz"
    test_dX_file = str(tmpdir.join("X.tst.npy"))
    test_Y_file = "test/tst-data/xmc/xlinear/Yt.npz"
    true_Y_pred_file = "test/tst-data/xmc/xlinear/Yt_primal_pred.npz"
    test_Y_pred_file = str(tmpdir.join("Yt_pred_test.npz"))
    model_folder = str(tmpdir.join("save_model"))

    np.save(train_dX_file, smat_util.load_matrix(train_sX_file).toarray(), allow_pickle=False)
    np.save(test_dX_file, smat_util.load_matrix(test_sX_file).toarray(), allow_pickle=False)

    for solver_type in ["L2R_L2LOSS_SVC_PRIMAL"]:
        for train_X, test_X in [(train_sX_file, test_sX_file), (train_dX_file, test_dX_file)]:
            # Training
            cmd = []
            cmd += ["python3 -m pecos.xmc.xlinear.train"]
            cmd += ["-x {}".format(train_X)]
            cmd += ["-y {}".format(train_Y_file)]
            cmd += ["-m {}".format(model_folder)]
            cmd += ["-s {}".format(solver_type)]
            cmd += ["--max-leaf-size {}".format(10)]
            process = subprocess.run(
                shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            assert process.returncode == 0, " ".join(cmd)

            # Batch Inference
            cmd = []
            cmd += ["python3 -m pecos.xmc.xlinear.predict"]
            cmd += ["-x {}".format(test_X)]
            cmd += ["-y {}".format(test_Y_file)]
            cmd += ["-o {}".format(test_Y_pred_file)]
            cmd += ["-m {}".format(model_folder)]
            process = subprocess.run(
                shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            assert process.returncode == 0, " ".join(cmd)
            true_Yt_pred = smat_util.load_matrix(true_Y_pred_file)
            Yt_pred = smat_util.load_matrix(test_Y_pred_file)
            assert Yt_pred.todense() == approx(true_Yt_pred.todense(), abs=1e-6)

            # Select Inference
            cmd = []
            cmd += ["python3 -m pecos.xmc.xlinear.predict"]
            cmd += ["-x {}".format(test_X)]
            cmd += ["-y {}".format(test_Y_file)]
            cmd += ["-so {}".format(true_Y_pred_file)]
            cmd += ["-o {}".format(test_Y_pred_file)]
            cmd += ["-m {}".format(model_folder)]
            process = subprocess.run(
                shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            assert process.returncode == 0, " ".join(cmd)
            true_Yt_pred = smat_util.load_matrix(true_Y_pred_file)
            Yt_selected_pred = smat_util.load_matrix(test_Y_pred_file)
            assert Yt_selected_pred.todense() == approx(true_Yt_pred.todense(), abs=1e-6)


def test_cli(tmpdir):
    import subprocess
    import shlex
    import numpy as np
    import scipy.sparse as smat
    from pecos.xmc.xlinear import XLinearModel as xlm
    from pecos.xmc import Indexer, LabelEmbeddingFactory
    from pecos.utils import smat_util

    train_sX_file = "test/tst-data/xmc/xlinear/X.npz"
    train_dX_file = str(tmpdir.join("X.trn.npy"))
    train_Y_file = "test/tst-data/xmc/xlinear/Y.npz"
    test_sX_file = "test/tst-data/xmc/xlinear/Xt.npz"
    test_dX_file = str(tmpdir.join("X.tst.npy"))
    test_Y_file = "test/tst-data/xmc/xlinear/Yt.npz"
    true_Y_pred_file = "test/tst-data/xmc/xlinear/Yt_pred.npz"
    true_Y_pred_with_man_file = "test/tst-data/xmc/xlinear/Yt_pred_with_tfn+man.npz"
    true_Yt_pred_with_splits = {
        2: "test/tst-data/xmc/xlinear/P_nr_splits=2.npz",
        4: "test/tst-data/xmc/xlinear/P_nr_splits=4.npz",
    }
    test_Y_pred_file = str(tmpdir.join("Yt_pred_test.npz"))
    code_file = str(tmpdir.join("codes.npz"))
    cluster_chain_folder = str(tmpdir.join("cluster_chain"))
    match_file = str(tmpdir.join("M.npz"))
    model_folder = str(tmpdir.join("save_model"))

    np.save(train_dX_file, smat_util.load_matrix(train_sX_file).toarray(), allow_pickle=False)
    np.save(test_dX_file, smat_util.load_matrix(test_sX_file).toarray(), allow_pickle=False)

    for train_X, test_X in [(train_sX_file, test_sX_file), (train_dX_file, test_dX_file)]:
        # Training
        cmd = []
        cmd += ["python3 -m pecos.xmc.xlinear.train"]
        cmd += ["-x {}".format(train_X)]
        cmd += ["-y {}".format(train_Y_file)]
        cmd += ["-m {}".format(model_folder)]
        cmd += ["--max-leaf-size {}".format(10)]
        process = subprocess.run(
            shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert process.returncode == 0, " ".join(cmd)

        # Batch Inference
        cmd = []
        cmd += ["python3 -m pecos.xmc.xlinear.predict"]
        cmd += ["-x {}".format(test_X)]
        cmd += ["-y {}".format(test_Y_file)]
        cmd += ["-o {}".format(test_Y_pred_file)]
        cmd += ["-m {}".format(model_folder)]
        process = subprocess.run(
            shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert process.returncode == 0, " ".join(cmd)
        true_Yt_pred = smat_util.load_matrix(true_Y_pred_file)
        Yt_pred = smat_util.load_matrix(test_Y_pred_file)
        assert Yt_pred.todense() == approx(true_Yt_pred.todense(), abs=1e-6)

        # Select Inference
        cmd = []
        cmd += ["python3 -m pecos.xmc.xlinear.predict"]
        cmd += ["-x {}".format(test_X)]
        cmd += ["-y {}".format(test_Y_file)]
        cmd += ["-so {}".format(true_Y_pred_file)]
        cmd += ["-o {}".format(test_Y_pred_file)]
        cmd += ["-m {}".format(model_folder)]
        process = subprocess.run(
            shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert process.returncode == 0, " ".join(cmd)
        true_Yt_pred = smat_util.load_matrix(true_Y_pred_file)
        Yt_selected_pred = smat_util.load_matrix(test_Y_pred_file)
        assert Yt_selected_pred.todense() == approx(true_Yt_pred.todense(), abs=1e-6)

        # Evaluate
        cmd = []
        cmd += ["python3 -m pecos.xmc.xlinear.evaluate"]
        cmd += ["-y {}".format(test_Y_file)]
        cmd += ["-p {}".format(true_Y_pred_file)]
        process = subprocess.run(
            shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert process.returncode == 0, " ".join(cmd)
        std_output = b"==== evaluation results ====\nprec   = 100.00 50.00 33.33 25.00 20.00 16.67 14.29 12.50 11.11 10.00\nrecall = 100.00 100.00 100.00 100.00 100.00 100.00 100.00 100.00 100.00 100.00\n"
        assert process.stdout == std_output

        # Training with Existing Clustering
        X = xlm.load_feature_matrix(train_X)
        Y = xlm.load_label_matrix(train_Y_file)
        label_feat = LabelEmbeddingFactory.create(Y, X, method="pifa")

        # Training with cluster chain stored in a cluster folder
        cluster_chain = Indexer.gen(label_feat, max_leaf_size=10)
        cluster_chain.save(cluster_chain_folder)
        cmd = []
        cmd += ["python3 -m pecos.xmc.xlinear.train"]
        cmd += ["-x {}".format(train_X)]
        cmd += ["-y {}".format(train_Y_file)]
        cmd += ["-c {}".format(cluster_chain_folder)]
        cmd += ["-m {}".format(model_folder)]
        process = subprocess.run(
            shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert process.returncode == 0, " ".join(cmd)

        # Training with last layer code matrix stored in a scipy.sparse matrix
        C = cluster_chain[-1]
        smat.save_npz(code_file, C)
        cmd = []
        cmd += ["python3 -m pecos.xmc.xlinear.train"]
        cmd += ["-x {}".format(train_X)]
        cmd += ["-y {}".format(train_Y_file)]
        cmd += ["-c {}".format(code_file)]
        cmd += ["-m {}".format(model_folder)]
        process = subprocess.run(
            shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert process.returncode == 0, " ".join(cmd)

        # Batch Inference
        cmd = []
        cmd += ["python3 -m pecos.xmc.xlinear.predict"]
        cmd += ["-x {}".format(test_X)]
        cmd += ["-y {}".format(test_Y_file)]
        cmd += ["-o {}".format(test_Y_pred_file)]
        cmd += ["-m {}".format(model_folder)]
        process = subprocess.run(
            shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert process.returncode == 0, " ".join(cmd)
        true_Yt_pred = smat_util.load_matrix(true_Y_pred_file)
        Yt_pred = smat_util.load_matrix(test_Y_pred_file)
        assert Yt_pred.todense() == approx(true_Yt_pred.todense(), abs=1e-6)

        # Select Inference
        cmd = []
        cmd += ["python3 -m pecos.xmc.xlinear.predict"]
        cmd += ["-x {}".format(test_X)]
        cmd += ["-y {}".format(test_Y_file)]
        cmd += ["-so {}".format(true_Y_pred_file)]
        cmd += ["-o {}".format(test_Y_pred_file)]
        cmd += ["-m {}".format(model_folder)]
        process = subprocess.run(
            shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert process.returncode == 0, " ".join(cmd)
        true_Yt_pred = smat_util.load_matrix(true_Y_pred_file)
        Yt_selected_pred = smat_util.load_matrix(test_Y_pred_file)
        assert Yt_selected_pred.todense() == approx(true_Yt_pred.todense(), abs=1e-6)

        # Training with User Supplied Negative
        M = (Y * C).tocsc()
        smat.save_npz(match_file, M)
        cmd = []
        cmd += ["python3 -m pecos.xmc.xlinear.train"]
        cmd += ["-x {}".format(train_X)]
        cmd += ["-y {}".format(train_Y_file)]
        cmd += ["-c {}".format(code_file)]
        cmd += ["-m {}".format(model_folder)]
        cmd += ["-um {}".format(match_file)]
        cmd += ["-uy {}".format(train_Y_file)]
        process = subprocess.run(
            shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert process.returncode == 0, " ".join(cmd)

        # Batch Inference
        cmd = []
        cmd += ["python3 -m pecos.xmc.xlinear.predict"]
        cmd += ["-x {}".format(test_X)]
        cmd += ["-y {}".format(test_Y_file)]
        cmd += ["-o {}".format(test_Y_pred_file)]
        cmd += ["-m {}".format(model_folder)]
        process = subprocess.run(
            shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert process.returncode == 0, " ".join(cmd)
        true_Yt_pred = smat_util.load_matrix(true_Y_pred_file)
        Yt_pred = smat_util.load_matrix(test_Y_pred_file)
        assert Yt_pred.todense() == approx(true_Yt_pred.todense(), abs=1e-6)

        # Select Inference
        cmd = []
        cmd += ["python3 -m pecos.xmc.xlinear.predict"]
        cmd += ["-x {}".format(test_X)]
        cmd += ["-y {}".format(test_Y_file)]
        cmd += ["-so {}".format(true_Y_pred_file)]
        cmd += ["-o {}".format(test_Y_pred_file)]
        cmd += ["-m {}".format(model_folder)]
        process = subprocess.run(
            shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert process.returncode == 0, " ".join(cmd)
        true_Yt_pred = smat_util.load_matrix(true_Y_pred_file)
        Yt_pred = smat_util.load_matrix(test_Y_pred_file)
        assert Yt_pred.todense() == approx(true_Yt_pred.todense(), abs=1e-6)

        # Evaluate
        cmd = []
        cmd += ["python3 -m pecos.xmc.xlinear.evaluate"]
        cmd += ["-y {}".format(test_Y_file)]
        cmd += ["-p {}".format(true_Y_pred_file)]
        process = subprocess.run(
            shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert process.returncode == 0, " ".join(cmd)
        std_output = b"==== evaluation results ====\nprec   = 100.00 50.00 33.33 25.00 20.00 16.67 14.29 12.50 11.11 10.00\nrecall = 100.00 100.00 100.00 100.00 100.00 100.00 100.00 100.00 100.00 100.00\n"
        assert process.stdout == std_output

        # Training with Matcher Aware Negatives
        cmd = []
        cmd += ["python3 -m pecos.xmc.xlinear.train"]
        cmd += ["-x {}".format(train_X)]
        cmd += ["-y {}".format(train_Y_file)]
        cmd += ["-m {}".format(model_folder)]
        cmd += ["--max-leaf-size {}".format(10)]
        cmd += ["-pp noop"]
        cmd += ["-b 2"]
        cmd += ["-ns tfn+man"]
        process = subprocess.run(
            shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert process.returncode == 0, " ".join(cmd)

        # Batch Inference
        cmd = []
        cmd += ["python3 -m pecos.xmc.xlinear.predict"]
        cmd += ["-x {}".format(test_X)]
        cmd += ["-y {}".format(test_Y_file)]
        cmd += ["-o {}".format(test_Y_pred_file)]
        cmd += ["-m {}".format(model_folder)]
        cmd += ["-pp sigmoid"]
        cmd += ["-b 4"]
        process = subprocess.run(
            shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert process.returncode == 0, " ".join(cmd)
        true_Yt_pred_with_man = smat_util.load_matrix(true_Y_pred_with_man_file)
        Yt_pred = smat_util.load_matrix(test_Y_pred_file)
        assert Yt_pred.todense() == approx(true_Yt_pred_with_man.todense(), abs=1e-6)

        # Select Inference
        cmd = []
        cmd += ["python3 -m pecos.xmc.xlinear.predict"]
        cmd += ["-x {}".format(test_X)]
        cmd += ["-y {}".format(test_Y_file)]
        cmd += ["-so {}".format(true_Y_pred_with_man_file)]
        cmd += ["-o {}".format(test_Y_pred_file)]
        cmd += ["-m {}".format(model_folder)]
        cmd += ["-pp sigmoid"]
        cmd += ["-b 4"]
        process = subprocess.run(
            shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert process.returncode == 0, " ".join(cmd)
        true_Yt_pred_with_man = smat_util.load_matrix(true_Y_pred_with_man_file)
        Yt_selected_pred = smat_util.load_matrix(test_Y_pred_file)
        assert Yt_selected_pred.todense() == approx(true_Yt_pred_with_man.todense(), abs=1e-6)

        # Training with Matcher Aware Negatives
        cmd = []
        cmd += ["python3 -m pecos.xmc.xlinear.train"]
        cmd += ["-x {}".format(train_X)]
        cmd += ["-y {}".format(train_Y_file)]
        cmd += ["-m {}".format(model_folder)]
        cmd += ["--max-leaf-size {}".format(10)]
        cmd += ["-pp noop"]
        cmd += ["-b 2"]
        cmd += ["-ns tfn+man"]
        process = subprocess.run(
            shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert process.returncode == 0, " ".join(cmd)

        # Batch Inference
        cmd = []
        cmd += ["python3 -m pecos.xmc.xlinear.predict"]
        cmd += ["-x {}".format(test_X)]
        cmd += ["-y {}".format(test_Y_file)]
        cmd += ["-o {}".format(test_Y_pred_file)]
        cmd += ["-m {}".format(model_folder)]
        cmd += ["-pp sigmoid"]
        cmd += ["-b 4"]
        process = subprocess.run(
            shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert process.returncode == 0, " ".join(cmd)
        true_Yt_pred_with_man = smat_util.load_matrix(true_Y_pred_with_man_file)
        Yt_pred = smat_util.load_matrix(test_Y_pred_file)
        assert Yt_pred.todense() == approx(true_Yt_pred_with_man.todense(), abs=1e-6)

        # Select Inference
        cmd = []
        cmd += ["python3 -m pecos.xmc.xlinear.predict"]
        cmd += ["-x {}".format(test_X)]
        cmd += ["-y {}".format(test_Y_file)]
        cmd += ["-so {}".format(true_Y_pred_with_man_file)]
        cmd += ["-o {}".format(test_Y_pred_file)]
        cmd += ["-m {}".format(model_folder)]
        cmd += ["-pp sigmoid"]
        cmd += ["-b 4"]
        process = subprocess.run(
            shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert process.returncode == 0, " ".join(cmd)
        true_Yt_pred_with_man = smat_util.load_matrix(true_Y_pred_with_man_file)
        Yt_selected_pred = smat_util.load_matrix(test_Y_pred_file)
        assert Yt_selected_pred.todense() == approx(true_Yt_pred_with_man.todense(), abs=1e-6)

        # Training with various number of splits to construct hierarchy
        for splits in [2, 4]:
            model_folder_local = f"{model_folder}-{splits}"

            cmd = []
            cmd += ["python3 -m pecos.xmc.xlinear.train"]
            cmd += [f"-x {train_X}"]
            cmd += [f"-y {train_Y_file}"]
            cmd += [f"--nr-splits {splits}"]
            cmd += [f"--max-leaf-size 2"]
            cmd += [f"-m {model_folder_local}"]

            process = subprocess.run(
                shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            assert process.returncode == 0, " ".join(cmd)

            cmd = []
            cmd += ["python3 -m pecos.xmc.xlinear.predict"]
            cmd += [f"-x {test_X}"]
            cmd += [f"-y {test_Y_file}"]
            cmd += [f"-m {model_folder_local}"]
            cmd += [f"-o {test_Y_pred_file}"]
            cmd += [f"-B 2"]

            process = subprocess.run(
                shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            assert process.returncode == 0, " ".join(cmd)

            true_Yt_pred = smat_util.load_matrix(true_Yt_pred_with_splits[splits])
            Yt_pred = smat_util.load_matrix(test_Y_pred_file)
            assert Yt_pred.todense() == approx(true_Yt_pred.todense(), abs=1e-6)

            cmd = []
            cmd += ["python3 -m pecos.xmc.xlinear.predict"]
            cmd += [f"-x {test_X}"]
            cmd += [f"-y {test_Y_file}"]
            cmd += [f"-so {true_Yt_pred_with_splits[splits]}"]
            cmd += [f"-m {model_folder_local}"]
            cmd += [f"-o {test_Y_pred_file}"]
            cmd += [f"-B 2"]

            process = subprocess.run(
                shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            assert process.returncode == 0, " ".join(cmd)

            true_Yt_pred = smat_util.load_matrix(true_Yt_pred_with_splits[splits])
            Yt_selected_pred = smat_util.load_matrix(test_Y_pred_file)
            assert Yt_selected_pred.todense() == approx(true_Yt_pred.todense(), abs=1e-6)


def test_split_model_at_depth():
    import numpy as np
    import scipy.sparse as smat
    from pecos.xmc.xlinear.model import XLinearModel
    from pecos.xmc import MLModel, HierarchicalMLModel

    c_matrix_1 = smat.csc_matrix([[1], [1]], dtype=np.float32)
    w_matrix_1 = smat.csc_matrix(np.random.normal(size=(10, 2)), dtype=np.float32)
    c_matrix_2 = smat.csc_matrix([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
    w_matrix_2 = smat.csc_matrix(np.random.normal(size=(10, 4)), dtype=np.float32)
    model_chain = [MLModel(C=c_matrix_1, W=w_matrix_1), MLModel(C=c_matrix_2, W=w_matrix_2)]
    xlm = XLinearModel(HierarchicalMLModel(model_chain))
    model_group = xlm.split_model_at_depth(given_depth=1, reindex=True)
    parent_model = model_group["parent_model"]
    child_models = model_group["child_models"]
    assert len(parent_model.model.model_chain) == 1
    assert len(child_models) == 2
    assert len(child_models[0][0].model.model_chain) == 1
    assert (parent_model.model.model_chain[0].C != c_matrix_1).nnz == 0
    assert (parent_model.model.model_chain[0].W != w_matrix_1).nnz == 0
    assert (child_models[0][0].model.model_chain[0].C != c_matrix_1).nnz == 0
    assert (child_models[0][0].model.model_chain[0].W != w_matrix_2[:, 0:2]).nnz == 0
    assert (child_models[1][0].model.model_chain[0].C != c_matrix_1).nnz == 0
    assert (child_models[1][0].model.model_chain[0].W != w_matrix_2[:, 2::]).nnz == 0
    assert child_models[0][1][0] == 0
    assert child_models[0][1][1] == 1
    assert child_models[1][1][0] == 2
    assert child_models[1][1][1] == 3

    c_matrix_1 = smat.csc_matrix([[1], [1]], dtype=np.float32)
    w_matrix_1 = smat.csc_matrix(np.random.normal(size=(10, 2)), dtype=np.float32)
    c_matrix_2 = smat.csc_matrix([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
    w_matrix_2 = smat.csc_matrix(np.random.normal(size=(10, 4)), dtype=np.float32)
    model_chain = [MLModel(C=c_matrix_1, W=w_matrix_1), MLModel(C=c_matrix_2, W=w_matrix_2)]
    xlm = XLinearModel(HierarchicalMLModel(model_chain))
    model_group = xlm.split_model_at_depth(given_depth=1, reindex=False)
    parent_model = model_group["parent_model"]
    child_models = model_group["child_models"]
    assert (parent_model.model.model_chain[0].C != c_matrix_1[:, [0]]).nnz == 0
    assert (parent_model.model.model_chain[0].W != w_matrix_1).nnz == 0
    assert child_models[0][0].model.model_chain[0].C.shape == (4, 1)
    assert child_models[0][0].model.model_chain[0].W.shape == (10, 4)
    assert len(child_models[0][0].model.model_chain[0].W.data) == 20


def test_reconstruct_model():
    import numpy as np
    import scipy.sparse as smat
    from pecos.xmc.xlinear.model import XLinearModel
    from pecos.xmc import MLModel, HierarchicalMLModel

    c_matrix_1 = smat.csc_matrix([[1], [1]], dtype=np.float32)
    w_matrix_1 = smat.csc_matrix(np.random.normal(size=(10, 2)), dtype=np.float32)
    c_matrix_2 = smat.csc_matrix([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
    w_matrix_2 = smat.csc_matrix(np.random.normal(size=(10, 4)), dtype=np.float32)
    model_chain = [MLModel(C=c_matrix_1, W=w_matrix_1), MLModel(C=c_matrix_2, W=w_matrix_2)]
    xlm = XLinearModel(HierarchicalMLModel(model_chain))
    model_group = xlm.split_model_at_depth(given_depth=1, reindex=True)
    parent_model = model_group["parent_model"]
    child_models_with_ids = model_group["child_models"]
    child_models = [child_model_with_ids[0] for child_model_with_ids in child_models_with_ids]

    parent_leaf_nr = parent_model.model.model_chain[-1].C.shape[0]
    child_leaf_nr = child_models[0].model.model_chain[-1].C.shape[0]
    Y_ids_of_child_models = np.array_split(
        np.arange(int(parent_leaf_nr * child_leaf_nr)), parent_leaf_nr
    )
    new_xlm = XLinearModel.reconstruct_model(parent_model, child_models, Y_ids_of_child_models)

    assert len(new_xlm.model.model_chain) == 2
    assert new_xlm.model.model_chain[0].C.shape == (2, 1)
    assert new_xlm.model.model_chain[0].W.shape == (10, 2)
    assert new_xlm.model.model_chain[1].C.shape == (4, 2)
    assert new_xlm.model.model_chain[1].W.shape == (10, 4)
    assert (new_xlm.model.model_chain[0].C != xlm.model.model_chain[0].C).nnz == 0
    assert (new_xlm.model.model_chain[0].W != xlm.model.model_chain[0].W).nnz == 0
    assert (new_xlm.model.model_chain[1].C != xlm.model.model_chain[1].C).nnz == 0
    assert (new_xlm.model.model_chain[1].W != xlm.model.model_chain[1].W).nnz == 0

    # different c_matrix_2
    c_matrix_1 = smat.csc_matrix([[1], [1]], dtype=np.float32)
    w_matrix_1 = smat.csc_matrix(np.random.normal(size=(10, 2)), dtype=np.float32)
    c_matrix_2 = smat.csc_matrix([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32)
    w_matrix_2 = smat.csc_matrix(np.random.normal(size=(10, 4)), dtype=np.float32)
    model_chain = [MLModel(C=c_matrix_1, W=w_matrix_1), MLModel(C=c_matrix_2, W=w_matrix_2)]
    xlm = XLinearModel(HierarchicalMLModel(model_chain))
    model_group = xlm.split_model_at_depth(given_depth=1, reindex=True)
    parent_model = model_group["parent_model"]
    child_models_with_ids = model_group["child_models"]
    child_models = [child_model_with_ids[0] for child_model_with_ids in child_models_with_ids]
    Y_ids_of_child_models = [
        child_model_with_ids[1] for child_model_with_ids in child_models_with_ids
    ]
    new_xlm = XLinearModel.reconstruct_model(parent_model, child_models, Y_ids_of_child_models)

    assert len(new_xlm.model.model_chain) == 2
    assert new_xlm.model.model_chain[0].C.shape == (2, 1)
    assert new_xlm.model.model_chain[0].W.shape == (10, 2)
    assert new_xlm.model.model_chain[1].C.shape == (4, 2)
    assert new_xlm.model.model_chain[1].W.shape == (10, 4)
    assert (new_xlm.model.model_chain[0].C != xlm.model.model_chain[0].C).nnz == 0
    assert (new_xlm.model.model_chain[0].W != xlm.model.model_chain[0].W).nnz == 0
    assert (new_xlm.model.model_chain[1].C != xlm.model.model_chain[1].C).nnz == 0
    assert (new_xlm.model.model_chain[1].W != xlm.model.model_chain[1].W).nnz == 0


def test_manual_init(tmpdir):
    import numpy as np
    from pecos.xmc.xlinear.model import XLinearModel
    from pecos.xmc import MLModel, HierarchicalMLModel
    from pecos.utils import smat_util

    train_X = smat_util.load_matrix("test/tst-data/xmc/xlinear/X.npz").astype(np.float32)
    train_Y = smat_util.load_matrix("test/tst-data/xmc/xlinear/Y.npz").astype(np.float32)
    test_X = smat_util.load_matrix("test/tst-data/xmc/xlinear/Xt.npz").astype(np.float32)

    xlm = XLinearModel.train(train_X, train_Y, bias=1.0)
    cluster_chain = [model.C for model in xlm.model.model_chain]
    weight_chain = [model.W for model in xlm.model.model_chain]

    # Initialize XLinearModel using weight and clustering matrices
    model_chain = [MLModel(C=C, W=W, bias=1.0) for C, W in zip(cluster_chain, weight_chain)]
    xlm_manual_init = XLinearModel(HierarchicalMLModel(model_chain))

    Yt_pred = xlm.predict(test_X)
    Yt_pred_manual = xlm_manual_init.predict(test_X)
    assert Yt_pred.todense() == approx(Yt_pred_manual.todense(), abs=1e-6)


def test_batch_prediction_mode(tmpdir):
    import numpy as np
    from pecos.xmc.xlinear.model import XLinearModel
    from pecos.utils import smat_util

    train_X = smat_util.load_matrix("test/tst-data/xmc/xlinear/X.npz").astype(np.float32)
    train_Y = smat_util.load_matrix("test/tst-data/xmc/xlinear/Y.npz").astype(np.float32)
    test_X = smat_util.load_matrix("test/tst-data/xmc/xlinear/Xt.npz").astype(np.float32)

    xlm = XLinearModel.train(train_X, train_Y, bias=1.0)

    Yt_pred = xlm.predict(test_X)
    Yt_pred_batch = xlm.predict(test_X, batch_size=2)
    assert Yt_pred.todense() == approx(Yt_pred_batch.todense(), abs=1e-6)


def test_matcher_ranker_mode():
    from pecos.utils import smat_util
    from pecos.xmc.xlinear.model import XLinearModel
    from pecos.xmc import Indexer, LabelEmbeddingFactory

    X = smat_util.load_matrix("test/tst-data/xmc/xlinear/X.npz")
    Y = smat_util.load_matrix("test/tst-data/xmc/xlinear/Y.npz")
    test_X = smat_util.load_matrix("test/tst-data/xmc/xlinear/Xt.npz")
    pred_kwargs = {"post_processor": "noop"}
    label_feat = LabelEmbeddingFactory.create(Y, X, method="pifa")
    cluster_chain = Indexer.gen(label_feat, max_leaf_size=2, nr_splits=2)
    xlmatcher = XLinearModel.train(
        X,
        Y,
        C=cluster_chain,
        ranker_level=3,
        mode="matcher",
        negative_sampling_scheme="tfn+man",
        pred_kwargs=pred_kwargs,
    )
    M_pred = xlmatcher.predict(test_X, only_topk=2)
    xlranker = XLinearModel.train(
        X,
        Y,
        C=cluster_chain,
        ranker_level=3,
        mode="ranker",
        user_supplied_negatives={3: M_pred},
        negative_sampling_scheme="usn+tfn+man",
        pred_kwargs=pred_kwargs,
    )
    Y_pred = xlranker.predict(test_X, only_topk=2)
    xlfull = XLinearModel.train(
        X,
        Y,
        C=cluster_chain,
        mode="full-model",
        negative_sampling_scheme="tfn+man",
        pred_kwargs=pred_kwargs,
    )
    Y_pred_full = xlfull.predict(test_X, only_topk=2)
    assert Y_pred.todense() == approx(Y_pred_full.todense(), abs=1e-6)


def test_ova_shallow_mode(tmpdir):
    from pecos.utils import smat_util
    from pecos.xmc.xlinear.model import XLinearModel
    from pecos.xmc import Indexer, LabelEmbeddingFactory
    import subprocess
    import shlex

    X = smat_util.load_matrix("test/tst-data/xmc/xlinear/X.npz")
    Y = smat_util.load_matrix("test/tst-data/xmc/xlinear/Y.npz")
    test_X = smat_util.load_matrix("test/tst-data/xmc/xlinear/Xt.npz")
    label_feat = LabelEmbeddingFactory.create(Y, X, method="pifa")
    cluster_chain = Indexer.gen(label_feat, max_leaf_size=10)
    print(cluster_chain[:])
    xlova = XLinearModel.train(
        X,
        Y,
        C=None,
    )
    ova_pred = str(tmpdir.join("P_ova.npz"))
    smat_util.save_matrix(ova_pred, xlova.predict(test_X))

    xlshallow = XLinearModel.train(
        X,
        Y,
        C=cluster_chain[-1],
        shallow=True,
    )
    shallow_pred = str(tmpdir.join("P_shallow.npz"))
    smat_util.save_matrix(shallow_pred, xlshallow.predict(test_X))
    # Evaluate
    cmd = []
    cmd += ["python3 -m pecos.xmc.xlinear.evaluate"]
    cmd += ["-y {}".format("test/tst-data/xmc/xlinear/Yt.npz")]
    cmd += ["-p {}".format(ova_pred)]
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.returncode == 0, " ".join(cmd)
    std_output = b"==== evaluation results ====\nprec   = 100.00 50.00 33.33 25.00 20.00 16.67 14.29 12.50 11.11 10.00\nrecall = 100.00 100.00 100.00 100.00 100.00 100.00 100.00 100.00 100.00 100.00\n"
    assert process.stdout == std_output

    cmd = []
    cmd += ["python3 -m pecos.xmc.xlinear.evaluate"]
    cmd += ["-y {}".format("test/tst-data/xmc/xlinear/Yt.npz")]
    cmd += ["-p {}".format(shallow_pred)]
    process = subprocess.run(
        shlex.split(" ".join(cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.returncode == 0, " ".join(cmd)
    assert process.stdout == std_output


def test_set_output_constraint(tmpdir):
    from pecos.utils import smat_util
    from pecos.xmc.xlinear import XLinearModel
    from pecos.xmc import Indexer, LabelEmbeddingFactory

    train_X_file = "test/tst-data/xmc/xlinear/X.npz"
    train_Y_file = "test/tst-data/xmc/xlinear/Y.npz"
    test_X_file = "test/tst-data/xmc/xlinear/Xt.npz"
    model_folder = str(tmpdir.join("save_model"))
    X = smat_util.load_matrix(train_X_file)
    Y = smat_util.load_matrix(train_Y_file)
    label_feat = LabelEmbeddingFactory.create(Y, X, method="pifa")
    Xt = smat_util.load_matrix(test_X_file)
    label_size = Y.shape[1]

    model_folder_list = []
    # Obtain xlinear models with vairous number of splits
    for splits in [2, 4]:
        model_folder_local = f"{model_folder}-{splits}"
        cluster_chain = Indexer.gen(label_feat, nr_splits=splits, max_leaf_size=2)
        py_model = XLinearModel.train(X, Y, C=cluster_chain)
        py_model.save(model_folder_local)
        model_folder_list.append(model_folder_local)

    # Obtain the OVA model
    py_model_ova = XLinearModel.train(X, Y, C=None)
    model_folder_local_ova = model_folder + "-ova"
    py_model_ova.save(model_folder_local_ova)
    model_folder_list.append(model_folder_local_ova)

    labels_to_keep_list = [[]]  # empty list
    labels_to_keep_list.append(
        list(set(map(int, [0, label_size / 3, label_size / 2, label_size * 2 / 3, label_size - 1])))
    )  # symmetric label indices to keep
    labels_to_keep_list.append(
        list(set(map(int, [0, label_size / 3, label_size / 2])))
    )  # asymmetric label indices to keep

    beam_size_list = [2, label_size]

    def obtain_ground_truth_pred(model, X, pruned_model, beam_size=None, post_processor=None):
        pred_csr = None
        default_kwargs = {"beam_size": 10, "only_topk": 20, "post_processor": "l3-hinge"}
        if post_processor is None:
            post_processor = default_kwargs.get("post_processor")

        if beam_size is None:
            beam_size = default_kwargs.get("beam_size")

        for d in range(model.depth):
            cur_model = model.model_chain[d]

            pred_csr = cur_model.predict(
                X, only_topk=cur_model.C.shape[0], csr_codes=pred_csr, post_processor=post_processor
            )
            kept_children = pruned_model.model_chain[d].C.indices

            for i_nnz in range(pred_csr.nnz):
                if pred_csr.indices[i_nnz] not in kept_children:
                    pred_csr.data[i_nnz] = 0
            pred_csr.eliminate_zeros()
            pred_csr = smat_util.sorted_csr(pred_csr, only_topk=beam_size)
        return pred_csr

    for model_folder_local in model_folder_list:
        for labels_to_keep in labels_to_keep_list:
            for beam_size in beam_size_list:
                py_model = XLinearModel.load(model_folder_local)
                py_model.set_output_constraint(labels_to_keep)
                model_folder_local_with_constraint = model_folder_local + "-constraint"
                py_model.save(model_folder_local_with_constraint)

                py_model_full = XLinearModel.load(model_folder_local)
                pred_ground_truth = obtain_ground_truth_pred(
                    py_model_full.model, Xt, py_model.model, beam_size
                ).todense()

                py_model_with_constraint = XLinearModel.load(model_folder_local_with_constraint)
                pred_with_constraint = py_model_with_constraint.predict(
                    X, beam_size=beam_size, only_topk=beam_size
                ).todense()

                py_model_with_constraint_predict_only = XLinearModel.load(
                    model_folder_local_with_constraint, is_predict_only=True
                )
                pred_with_constraint_predict_only = py_model_with_constraint_predict_only.predict(
                    X, beam_size=beam_size, only_topk=beam_size
                ).todense()

                assert pred_ground_truth[:, labels_to_keep] == approx(
                    pred_with_constraint[:, labels_to_keep], abs=1e-6
                ), f"prediction result for label constraints {labels_to_keep}, beam_size={beam_size}, model={model_folder_local_with_constraint} and XLinearModel.load is not correct"

                assert pred_ground_truth[:, labels_to_keep] == approx(
                    pred_with_constraint_predict_only[:, labels_to_keep], abs=1e-6
                ), f"prediction result for label constraints {labels_to_keep}, beam_size={beam_size}, model={model_folder_local_with_constraint} and XLinearModel.load in predict-only mode is not correct"


def test_get_submodel():
    import numpy as np
    import scipy.sparse as smat
    from pecos.xmc import MLModel

    c_matrix = smat.csc_matrix([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1]], dtype=np.float32)
    w_matrix = smat.csc_matrix(np.eye(4), dtype=np.float32)
    ml_model = MLModel(C=c_matrix, W=w_matrix)
    out = ml_model.get_submodel(selected_codes=[1, 2], reindex=True)
    assert len(out["active_labels"]) == 3
    assert len(out["active_codes"]) == 2
    assert 0 in out["active_labels"]
    assert 1 in out["active_labels"]
    assert 3 in out["active_labels"]
    new_C = ml_model.C[out["active_labels"], :]
    new_C = new_C[:, [1, 2]]
    assert (out["model"].C != new_C).nnz == 0  # check for equality of sparse matrices
    assert (out["model"].W != ml_model.W[:, out["active_labels"]]).nnz == 0

    c_matrix = smat.csc_matrix([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1]], dtype=np.float32)
    w_matrix = smat.csc_matrix(np.random.normal(size=(10, 4)), dtype=np.float32)
    ml_model = MLModel(C=c_matrix, W=w_matrix)
    out = ml_model.get_submodel(selected_codes=[1, 2], reindex=False)
    assert out["model"].C.shape == c_matrix.shape
    assert out["model"].W.shape == w_matrix.shape
    for r in range(c_matrix.shape[0]):
        for c in range(c_matrix.shape[1]):
            if r in [0, 1, 3] and c in [1, 2]:
                assert out["model"].C[r, c] == c_matrix[r, c]
            else:
                assert out["model"].C[r, c] == 0
    for r in range(w_matrix.shape[0]):
        for c in range(w_matrix.shape[1]):
            if c in [0, 1, 3]:
                assert out["model"].W[r, c] == w_matrix[r, c]
            else:
                assert out["model"].W[r, c] == 0
    assert len(out["active_labels"]) == 3
    assert len(out["active_codes"]) == 2
    assert 0 in out["active_labels"]
    assert 1 in out["active_labels"]
    assert 3 in out["active_labels"]


def test_predict_consistency_between_topk_and_selected(tmpdir):
    from pecos.xmc import PostProcessor, Indexer, LabelEmbeddingFactory
    from pecos.xmc.xlinear import XLinearModel

    train_X_file = "test/tst-data/xmc/xlinear/X.npz"
    train_Y_file = "test/tst-data/xmc/xlinear/Y.npz"
    test_X_file = "test/tst-data/xmc/xlinear/Xt.npz"
    Xt = XLinearModel.load_feature_matrix(train_X_file)
    Yt = XLinearModel.load_feature_matrix(train_Y_file)
    model_folder = str(tmpdir.join("save_model"))
    label_feat = LabelEmbeddingFactory.create(Yt, Xt, method="pifa")

    model_folder_list = []
    # Obtain xlinear models with vairous number of splits
    for splits in [2, 4]:
        model_folder_local = f"{model_folder}-{splits}"
        cluster_chain = Indexer.gen(label_feat, nr_splits=splits)
        py_model = XLinearModel.train(Xt, Yt, C=cluster_chain)
        py_model.save(model_folder_local)
        model_folder_list.append(model_folder_local)

    X = XLinearModel.load_feature_matrix(test_X_file)

    def test_on_model(model, X):
        for pp in PostProcessor.valid_list():
            # Batch mode topk
            py_sparse_topk_pred = model.predict(X, post_processor=pp)
            py_dense_topk_pred = model.predict(X.todense(), post_processor=pp)

            # Sparse Input
            py_selected_sparse_topk_pred = model.predict(
                X, selected_outputs_csr=py_sparse_topk_pred, post_processor=pp
            )
            # Dense Input
            py_selected_dense_topk_pred = model.predict(
                X.todense(), selected_outputs_csr=py_dense_topk_pred, post_processor=pp
            )

            assert py_sparse_topk_pred.todense() == approx(
                py_selected_sparse_topk_pred.todense(), abs=1e-6
            ), f"model:{model_folder_local} (batch, sparse, topk) post_processor:{pp})"
            assert py_dense_topk_pred.todense() == approx(
                py_selected_dense_topk_pred.todense(), abs=1e-6
            ), f"model:{model_folder_local} (batch, dense, topk) post_processor:{pp})"

            # Realtime mode topk
            for i in range(X.shape[0]):
                query_slice = X[[i], :]
                query_slice.sort_indices()

                py_sparse_realtime_pred = model.predict(query_slice, post_processor=pp)
                py_dense_realtime_pred = model.predict(query_slice.todense(), post_processor=pp)

                # Sparse Input
                py_selected_sparse_realtime_pred = model.predict(
                    query_slice, selected_outputs_csr=py_sparse_realtime_pred, post_processor=pp
                )
                # Dense input
                py_selected_dense_realtime_pred = model.predict(
                    query_slice.todense(),
                    selected_outputs_csr=py_dense_realtime_pred,
                    post_processor=pp,
                )

                assert py_sparse_realtime_pred.todense() == approx(
                    py_selected_sparse_realtime_pred.todense(), abs=1e-6
                ), f"model:{model_folder_local} (realtime, sparse, topk) post_processor:{pp}"
                assert py_dense_realtime_pred.todense() == approx(
                    py_selected_dense_realtime_pred.todense(), abs=1e-6
                ), f"model:{model_folder_local} (realtime, dense, topk) post_processor:{pp}"

    for model_folder_local in model_folder_list:
        model_f = XLinearModel.load(model_folder_local, is_predict_only=False)
        model_t = XLinearModel.load(
            model_folder_local, is_predict_only=True, weight_matrix_type="CSC"
        )

        test_on_model(model_f, X)
        test_on_model(model_t, X)
