
dataset=$1
if [ ${dataset} != "ogbn-arxiv" ] && [ ${dataset} != "ogbn-products" ]; then
    echo "dataset=${dataset} is not yet supported!"
    exit
fi

wget https://archive.org/download/pecos-dataset/giant-xrt/${dataset}.tar.gz
tar -zxvf ${dataset}.tar.gz
