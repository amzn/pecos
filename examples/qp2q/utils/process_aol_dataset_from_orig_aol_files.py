import os
import re
import csv
import sys
import json
import glob
import argparse
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s ",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def get_all_data(fname):
    """

    Parameters
    ----------
    fname: file containing raw aol data

    Returns dict with key = user_id
                        value =  list of (query, query_time) tuples
    -------

    """
    LOGGER.info("Starting reading all data from {}".format(fname))

    date_format = "%Y-%m-%d %H:%M:%S"
    with open(fname, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        all_data = defaultdict(list)
        header = next(reader, None)
        for ctr, row in enumerate(reader):

            user_id = row[0]
            q_time = row[2]
            query_tokens = [q_token for q_w_dot in row[1].split() for q_token in q_w_dot.split(".")]
            query_tokens = [
                re.sub(r"\W+", "", token) for token in query_tokens
            ]  # Remove non-alpha numberic characters

            query = " ".join(query_tokens)
            query = query.lower().strip()
            if len(query) == 0:  # Skip empty queries
                continue
            all_data[user_id] += [(query, datetime.strptime(q_time, date_format))]

    LOGGER.info("Finished reading all data from  {}".format(fname))
    return all_data


def create_sessions(all_data, sess_len):
    """

    Parameters
    ----------
    all_data: dict with key = user_id
                        value =  list of (query, query_time) tuples
    sess_len : int
            Minimum time difference between two queries required to put them in different sessions
    Returns list of (sessions, min_session_time, max_session_time) tuples where each session is a list of queries
    -------

    """
    LOGGER.info(
        "Splitting data into session using min query time diff = {} seconds".format(sess_len)
    )
    all_session_data = []
    for user_id in all_data:

        if len(all_data[user_id]) > 0:
            prev_query, prev_query_time = all_data[user_id][0]
            curr_session = [prev_query]
            curr_session_times = [prev_query_time]
            for curr_query, curr_query_time in all_data[user_id][1:]:
                diff = curr_query_time - prev_query_time
                if diff.total_seconds() > sess_len:  # Start new session
                    if len(curr_session) > 0:
                        all_session_data += [
                            (curr_session, min(curr_session_times), max(curr_session_times))
                        ]

                    # Start a new session
                    curr_session = [curr_query]
                    curr_session_times = [curr_query_time]
                else:
                    curr_session += [curr_query]
                    curr_session_times += [curr_query_time]
                prev_query_time = curr_query_time

            if len(curr_session) > 0:
                all_session_data += [
                    (curr_session, min(curr_session_times), max(curr_session_times))
                ]

    LOGGER.info(
        "Created data w/ {} sessions using min query time diff = {} seconds".format(
            len(all_session_data), sess_len
        )
    )
    return all_session_data


def deduplicate_session_data(all_session_data):
    """
    Removes duplicate copies of queries when they appear consecutively in a session
    Parameters
    ----------
    all_session_data: list of (sessions, min_session_time, max_session_time) tuples where each session is a list of queries

    Returns list of (session, min_session_time, max_session_time) tuples where session is deduplicated
    -------

    """
    # For each session, create prev query and next query pairs
    LOGGER.info(
        "Removing duplicate consecutive queries from {} sessions".format(len(all_session_data))
    )
    # Deduplicate session data
    dedup_sess_data = []
    for session, min_sess_time, max_sess_time in all_session_data:

        prev_query = session[0] if len(session) > 0 else None
        dedup_sess = [prev_query] if len(session) > 0 else []
        for curr_query in session:
            if curr_query == prev_query:  # Ignore duplicate consecutive queries
                continue
            else:
                dedup_sess += [curr_query]
            prev_query = curr_query

        if len(dedup_sess) > 1:  # Only consider sessions that have at least 2 queries
            dedup_sess_data += [(dedup_sess, min_sess_time, max_sess_time)]

    LOGGER.info(
        "Finished removing duplicate consecutive queries and we have {} sessions now ".format(
            len(dedup_sess_data)
        )
    )
    return dedup_sess_data


def split_sessions_into_train_test_val(all_sessions):
    """
    Splits session data into train/test/val lists based on session start time.
    1 March - 15 May -> Train
    16 May - 23 May -> Test
    24 May - 31 May -> Val
    Parameters
    ----------
    all_sessions: list of (session, min_session_time, max_session_time) tuples where session is deduplicated

    Returns train/test/val session lists
    -------

    """
    date_format = "%Y-%m-%d %H:%M:%S"
    train_time_range = datetime.strptime("2006-03-01 00:00:00", date_format), datetime.strptime(
        "2006-05-15 23:59:59", date_format
    )
    val_time_range = datetime.strptime("2006-05-16 00:00:00", date_format), datetime.strptime(
        "2006-05-23 23:59:59", date_format
    )
    test_time_range = datetime.strptime("2006-05-24 00:00:00", date_format), datetime.strptime(
        "2006-05-31 23:59:59", date_format
    )
    train_sessions, test_sessions, val_sessions = [], [], []
    for curr_session, min_sess_time, max_sess_time in all_sessions:

        assert (
            min_sess_time <= max_sess_time
        ), "Min session = {} time should be less than max session time = {}.".format(
            min_sess_time, max_sess_time
        )

        if train_time_range[0] <= min_sess_time <= train_time_range[1]:
            train_sessions.append(curr_session)
        elif val_time_range[0] <= min_sess_time <= val_time_range[1]:
            val_sessions.append(curr_session)
        elif test_time_range[0] <= min_sess_time <= test_time_range[1]:
            test_sessions.append(curr_session)
        else:
            raise Exception(
                "This session range is not handled = ({}, {})".format(min_sess_time, max_sess_time)
            )

    return train_sessions, test_sessions, val_sessions


def create_query_pairs(all_session_data):
    query_pairs = [
        (prev_query, next_query)
        for s_queries in all_session_data
        for prev_query, next_query in zip(s_queries[:-1], s_queries[1:])
    ]
    return query_pairs


def convert_to_train_data_format(query_pairs, out_fname):
    """
    Create training data by sampling prefix from next query in the query pairs and write it to a file
    Parameters
    ----------
    query_pairs: list of (prev_query, next_query) pairs
    out_fname : name of file where data is where the data is written

    Returns : None
    """
    np.random.seed(0)
    Path(os.path.dirname(out_fname)).mkdir(exist_ok=True, parents=True)
    with open(out_fname, "w") as writer:
        for prev_query, next_query in query_pairs:
            # Sample prefix len uniformly at random
            pref_len = int(np.random.choice(range(1, len(next_query) + 1), size=1)[0])
            new_datapoint = {
                "prev_query": prev_query,
                "prefix": next_query[:pref_len],
                "next_query": next_query,
            }
            writer.write(json.dumps(new_datapoint) + "\n")

    LOGGER.info("Created train dataset")


def convert_to_gt_data_format(query_pairs, out_fname):
    """
    Create test data by sampling prefix from next query in the query pairs and write it to a file
    Parameters
    ----------
    query_pairs: list of (prev_query, next_query) pairs
    out_fname : name of file where data is where the data is written

    Returns : None
    """
    np.random.seed(0)
    Path(os.path.dirname(out_fname)).mkdir(exist_ok=True, parents=True)
    with open(out_fname, "w") as writer:
        for prev_query, next_query in query_pairs:
            pref_len = int(
                np.random.choice(range(1, len(next_query) + 1), size=1)[0]
            )  # Uniformly sample a prefix
            prefix = next_query[:pref_len]
            output = {"prev_query": prev_query, "prefix": prefix, "next_query": next_query}
            writer.write(json.dumps(output) + "\n")

    LOGGER.info("Created test dataset")


def main():
    parser = argparse.ArgumentParser(description="Process AOL Dataset (from original AOL files)")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory where all raw AOL log files are stored",
    )
    parser.add_argument("--out_dir", type=str, required=True, help="Dir to store out file")
    parser.add_argument(
        "--sess_len",
        type=int,
        default=1800,
        help="Min time gap (in seconds) b/w two consecutive queries to put them in different sessions ",
    )

    np.random.seed(0)

    args = parser.parse_args()
    data_dir = args.data_dir
    out_dir = args.out_dir
    sess_len = args.sess_len

    Path(out_dir).mkdir(exist_ok=True, parents=True)

    os.system("cp {} {}/".format(sys.argv[0], out_dir))
    os.system(
        "echo {} >> {}/command_to_run.txt".format(" ".join([str(x) for x in sys.argv]), out_dir)
    )

    data_files = sorted([str(f) for f in glob.glob("{}/*.txt".format(data_dir))])
    LOGGER.info("Data files  = {}".format(data_files))
    train_sessions = []
    test_sessions = []
    val_sessions = []
    all_sessions = []
    for curr_file in data_files:
        LOGGER.info("\nProcessing file = {}\n".format(curr_file))
        all_data = get_all_data(fname=curr_file)
        dedup_sessions = deduplicate_session_data(
            all_session_data=create_sessions(all_data=all_data, sess_len=sess_len)
        )
        curr_train, curr_test, curr_val = split_sessions_into_train_test_val(dedup_sessions)

        all_sessions += dedup_sessions
        train_sessions += curr_train
        val_sessions += curr_val
        test_sessions += curr_test

        assert len(train_sessions) + len(val_sessions) + len(test_sessions) == len(
            all_sessions
        ), "{} + {} + {} != {}".format(
            len(train_sessions), len(val_sessions), len(test_sessions), len(all_sessions)
        )

    LOGGER.info("Total number of sessions = {}".format(len(all_sessions)))
    LOGGER.info(
        "Total number of query pairs = {}".format(
            len(create_query_pairs([sess for sess, _, _ in all_sessions]))
        )
    )
    LOGGER.info(
        "Total number of unique query pairs = {}".format(
            len(set(create_query_pairs([sess for sess, _, _ in all_sessions])))
        )
    )
    LOGGER.info(
        "Total number of queries = {}".format(len([q for sess, _, _ in all_sessions for q in sess]))
    )
    LOGGER.info(
        "Total number of unique queries = {}".format(
            len(set([q for sess, _, _ in all_sessions for q in sess]))
        )
    )

    LOGGER.info("Total number of train sessions = {}".format(len(train_sessions)))
    LOGGER.info(
        "Total number of train query pairs = {}".format(len(create_query_pairs(train_sessions)))
    )
    LOGGER.info(
        "Total number of train unique query pairs = {}".format(
            len(set(create_query_pairs(train_sessions)))
        )
    )
    LOGGER.info(
        "Total number of train queries = {}".format(
            len([q for sess in train_sessions for q in sess])
        )
    )
    LOGGER.info(
        "Total number of unique train queries = {}".format(
            len(set([q for sess in train_sessions for q in sess]))
        )
    )

    LOGGER.info("Total number of test sessions = {}".format(len(test_sessions)))
    LOGGER.info(
        "Total number of test query pairs = {}".format(len(create_query_pairs(test_sessions)))
    )
    LOGGER.info(
        "Total number of test unique query pairs = {}".format(
            len(set(create_query_pairs(test_sessions)))
        )
    )
    LOGGER.info(
        "Total number of test queries = {}".format(len([q for sess in test_sessions for q in sess]))
    )
    LOGGER.info(
        "Total number of unique test queries = {}".format(
            len(set([q for sess in test_sessions for q in sess]))
        )
    )

    LOGGER.info("Total number of val sessions = {}".format(len(val_sessions)))
    LOGGER.info(
        "Total number of val query pairs = {}".format(len(create_query_pairs(val_sessions)))
    )
    LOGGER.info(
        "Total number of val unique query pairs = {}".format(
            len(set(create_query_pairs(val_sessions)))
        )
    )
    LOGGER.info(
        "Total number of val queries = {}".format(len([q for sess in val_sessions for q in sess]))
    )
    LOGGER.info(
        "Total number of unique val queries = {}".format(
            len(set([q for sess in val_sessions for q in sess]))
        )
    )

    convert_to_train_data_format(
        query_pairs=create_query_pairs(train_sessions),
        out_fname="{}/train/train.json".format(out_dir),
    )
    convert_to_train_data_format(
        query_pairs=create_query_pairs(val_sessions), out_fname="{}/val/val.json".format(out_dir)
    )
    convert_to_gt_data_format(
        query_pairs=create_query_pairs(test_sessions), out_fname="{}/test/test.json".format(out_dir)
    )


if __name__ == "__main__":
    main()
