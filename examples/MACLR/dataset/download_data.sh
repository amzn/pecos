dataset=$1
if [ ${dataset} != "LF-Amazon-131K" ] && [ ${dataset} != "LF-WikiSeeAlso-320K" ] && [ ${dataset} != "LF-Wikipedia-500K" ] && [ ${dataset} != "LF-Amazon-1M" ]; then
    echo "dataset=${dataset} is not yet supported!"
    exit
fi

wget https://archive.org/download/maclr-www22/datasets/${dataset}.tar.gz
tar -zxvf ${dataset}.tar.gz