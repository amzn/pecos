
import sys
from pecos.utils import smat_util
from sentence_transformers import SentenceTransformer, LoggingHandler
import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def main(model_name_or_path, input_txt_path, output_emb_path):
    # load data
    corpus = [line.strip() for line in open(input_txt_path, "r")]
    logging.info("|corpus| = {}".format(len(corpus)))

    # load encoder
    # e.g., model_name_or_path = "sentence-transformers/nq-distilbert-base-v1"
    model = SentenceTransformer(model_name_or_path)
    logging.info("model_name_or_path {}".format(model_name_or_path))

    # Start the multi-process pool on all available CUDA devices
    # https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/computing-embeddings/computing_embeddings_mutli_gpu.py
    mp_pool = model.start_multi_process_pool()

    # encoding
    #emb_arr = model.encode(corpus, show_progress_bar=True)
    emb_arr = model.encode_multi_process(corpus, mp_pool, batch_size=256)
    logging.info("emb_arr {}".format(emb_arr.shape))

    # saving output as npy
    smat_util.save_matrix(output_emb_path, emb_arr)

    #Optional: Stop the proccesses in the pool
    model.stop_multi_process_pool(mp_pool)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("python encoder.py [model_name_or_path] [input_txt_path] [output_emb_path]")
        exit(0)
    model_name_or_path = sys.argv[1]
    input_txt_path = sys.argv[2]
    output_emb_path = sys.argv[3]
    main(model_name_or_path, input_txt_path, output_emb_path)
