import pandas as pd
import sys, os
import re
import string

regex_punctuation = re.compile("[%s]" % re.escape(string.punctuation))
regex_space = re.compile(r"\s+")

def normalize(xx):
    xx = regex_punctuation.sub(" ", xx)
    xx = regex_space.sub(" ", xx).strip().lower()
    return xx


def process_json_data(dataname, part):
    raw_dir = f"{dataname}/raw/" 
    norm_dir = f"{dataname}/normalized/" 
    os.makedirs(norm_dir, exist_ok=True)

    assert part in ["trn", "tst", "lbl"]
    df = pd.read_json(f"{raw_dir}/{part}.json", lines=True)

    X_title = list(df.title)
    X_content = list(df.content)


    Y = list(df.target_ind)
    X_norm = []
    X = []

    if "titles" in dataname.lower():
        for xxx in X_title:
            X.append(xxx)
            X_norm.append(normalize(xxx))
    else:
        for i in range(len(X_title)):
            xxx = X_title[i] + " " + X_content[i]
            X.append(xxx)
            X_norm.append(normalize(xxx))
        
    # X_norm = [normalize(xxx) for xxx in X]
    

    if part in ["trn", "tst"]:
        # trn.txt tst.txt
        # l_1,l_2,...,l_k<TAB>xxxx xxxx xxxx
        with open(f"{raw_dir}/{part}.txt", 'w') as fout:
            for xx, yy in zip(X, Y):
                fout.write(",".join([str(y) for y in yy]) + '\t' + xx + '\n')

        with open(f"{norm_dir}/{part}.txt", 'w') as fout:
            for xx, yy in zip(X_norm, Y):
                fout.write(",".join([str(y) for y in yy]) + '\t' + xx + '\n')

        with open(f"{raw_dir}/X.{part}.txt", 'w') as fout:
            for xx in X:
                fout.write(xx + '\n')
        with open(f"{norm_dir}/X.{part}.txt", 'w') as fout:
            for xx in X_norm:
                fout.write(xx + '\n')
    else:
        with open(f"{raw_dir}/output-items.txt", 'w') as fout:
            for xx in X:
                fout.write(xx + '\n')

        with open(f"{norm_dir}/output-items.txt", 'w') as fout:
            for xx in X_norm:
                fout.write(xx + '\n')


dataname = f'./dataset/{sys.argv[1]}'
process_json_data(dataname, "tst")
process_json_data(dataname, "trn")
process_json_data(dataname, "lbl")
