#================= inputs =====================
dataset=$1 # This is th dataset name (i.e LF-Amazon-131K).
work_dir="."
mkdir -p dataset
cd dataset
echo "$(pwd)"

echo "Downloading ${dataset}"
if [[ $dataset == "LF-Amazon-131K" ]]
then
    rawtext_id="1WuquxCAg8D4lKr-eZXPv4nNw2S2lm7_E"
    BoW_id="1YNGEifTHu4qWBmCaLEBfjx07qRqw9DVW"
    BoW_dname="LF-Amazon-131K"
elif [[ $dataset == "LF-WikiSeeAlso-320K" ]]
then
    rawtext_id="1QZD4dFVxDpskCI2kGH9IbzgQR1JSZT-N"
    BoW_id="1N8C_RL71ErX6X92ew9h8qRuTWJ9LywE8"
    BoW_dname="LF-WikiSeeAlso-320K"
elif [[ $dataset == "LF-Amazon-1.3M" ]]
then
    rawtext_id="12zH4mL2RX8iSvH0VCNnd3QxO4DzuHWnK"
    BoW_id="1Davc6BIfoTIAS3mP1mUY5EGcGr2zN2pO"
    BoW_dname="LF-AmazonTitles-1.3M"
fi

echo $rawtext_id

gdown $rawtext_id
gdown $BoW_id

unzip $dataset.raw.zip
unzip -j $BoW_dname.bow.zip -d $dataset

data_dir="${dataset}"
mkdir -p ${data_dir}/normalized
mkdir -p ${data_dir}/raw

cd ${dataset}

mv *.json.gz raw

cd raw

gunzip *.gz 

echo "${dataset} downlowded and unzipped!!!"

