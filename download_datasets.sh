# CONLL 2003 NER
mkdir -p data/conll2003ner
cd data/conll2003ner
wget https://data.deepai.org/conll2003.zip
unzip conll2003.zip
rm conll2003.zip
cd ../..

# Github link below has been deleted because of copyright issues
# For detailed information, please refer to this link: https://github.com/huggingface/datasets/issues/3582
#wget https://raw.githubusercontent.com/davidsbatista/NER-datasets/master/CONLL2003/train.txt -O data/conll2003ner/train.txt
#wget https://raw.githubusercontent.com/davidsbatista/NER-datasets/master/CONLL2003/valid.txt -O data/conll2003ner/val.txt
#wget https://raw.githubusercontent.com/davidsbatista/NER-datasets/master/CONLL2003/test.txt -O data/conll2003ner/test.txt

# Bibtex 7395 1836 159 card: 2.4
mkdir -p data/bibtex_original
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1IgcJQs3__v6VXa5rxfU9nAVb2blgpNdy' -O data/bibtex_original/train.arff
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1FkDQBhRwUFqzhG8ywK-St35lNanfoyiT' -O data/bibtex_original/val.arff
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1sWhjC02Tf0qWFeMKAOYPsjYY63V0J96N' -O data/bibtex_original/test.arff
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1pEAPKR4IYr8oP6jY32P_bM6gp262hX4b' -O data/bibtex_original/train+val.arff

# required installation from below.
pip install gdown

#NYT
mkdir -p data/nyt
cd data/nyt
gdown "https://drive.google.com/uc?id=1ZI4GkHejbW1aSG7bRyNknSiLpBdiIZRD"
tar -xzvf nyt.tar.gz
rm nyt.tar.gz
cd ../..

#Blurb Genre Collection
mkdir -p data/bgc
cd data/bgc
gdown "https://drive.google.com/uc?id=1AuB1qHWhqcD1LL3lV-2KmAdLJHnl14Dh"
tar -xzvf bgc.tar.gz
rm bgc.tar.gz
cd ../..

# (num_instances, num_features, num_labels) commented.

mkdir -p data/cal500-stratified10folds-meka  # 502	68	174 card: 26.  (priority: cal500, eurlex, corel5k)
cd data/cal500-stratified10folds-meka
gdown https://drive.google.com/uc?id=1m3lKsN3KRI9UOwgRnmd_erVVzYV9vzSH
tar -xzvf cal500-stratified10folds-meka.tar.gz
rm cal500-stratified10folds-meka.tar.gz
cd ../..

mkdir -p data/corel5k-stratified10folds-meka # 5000	499	374 card: 3.522  (comparison bibtex: 7395 1836 159 card: 2.4) --> 0.001 active label ratio per (example * label dim).
cd data/corel5k-stratified10folds-meka
gdown https://drive.google.com/uc?id=1_29bDv6Hir5LMqxOOHJ1bpEmZoMlw0Pr
tar -xzvf corel5k-stratified10folds-meka.tar.gz
rm corel5k-stratified10folds-meka.tar.gz
cd ../..

mkdir -p data/delicious-stratified10folds-meka #16110	500	983 card: 19.02  -> 0.02  active label ratio per (example * label dim).
cd data/delicious-stratified10folds-meka
gdown https://drive.google.com/uc?id=1ks93YBs7MkHW8uBPmG5fSbtk3-ejZmaA
tar -xzvf delicious-stratified10folds-meka.tar.gz
rm delicious-stratified10folds-meka.tar.gz
cd ../..


mkdir -p data/eurlex-ev-stratified10folds-meka # 19350	5000	3993 continuous card: 5.31
cd data/eurlex-ev-stratified10folds-meka
gdown https://drive.google.com/uc?id=1Ncubmp4yiixL2Twcf-LiXqz-LfHEaCqG
tar -xzvf eurlex-ev-stratified10folds-meka.tar.gz
rm eurlex-ev-stratified10folds-meka.tar.gz
cd ../..


mkdir -p data/genbase-stratified10folds-meka  #662	1186	27  -> done.
cd data/genbase-stratified10folds-meka
gdown https://drive.google.com/uc?id=1ubdLXR5FvWz0pw8XZu5IWgiAF0D1yohW
tar -xzvf genbase-stratified10folds-meka.tar.gz
rm genbase-stratified10folds-meka.tar.gz
cd ../..


mkdir -p data/mediamill-stratified10folds-meka # 43910 120 101  continuous
cd data/mediamill-stratified10folds-meka
gdown https://drive.google.com/uc?id=1ZbQcyriJNXbvpTbSW2ZUF4-UDGetKR5x
tar -xzvf mediamill-stratified10folds-meka.tar.gz
rm mediamill-stratified10folds-meka.tar.gz
cd ../..



# mlc with taxonomy
mkdir -p data/expr_fun
cd data/expr_fun
gdown https://drive.google.com/uc?id=1w_I7gpdJeCaavPC8W9CkVzszR_PsgoBL
tar -xzvf expr_FUN.tar.gz
rm expr_FUN.tar.gz
cd ../..

mkdir -p data/expr_go
cd data/expr_go
gdown https://drive.google.com/uc?id=1m_xHwr6bPPfe3YX2-7E_8qYkPqryWOuX
tar -xzvf expr_GO.tar.gz
rm expr_GO.tar.gz
cd ../..

mkdir -p data/spo_fun
cd data/spo_fun
gdown https://drive.google.com/uc?id=1JKICdapuU9IuDskF72fmpE29E4savwXr
tar -xzvf spo_FUN.tar.gz
rm spo_FUN.tar.gz
cd ../..

mkdir -p data/spo_go
cd data/spo_go
gdown https://drive.google.com/uc?id=10X3osoI26vL8FN__oMZYcQkuk89BevuQ
tar -xzvf spo_GO.tar.gz
rm spo_GO.tar.gz
cd ../..





# Bibtex better stratified splits.
mkdir -p data/bibtex_stratified10folds_meka
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1V8OMs4HqxZCeZYoG2DbDK-8zGC-iL2wn' -O data/bibtex_stratified10folds_meka/Bibtex-fold1.arff
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1VCi72UEcpU0tGyN7xeaZbG0rxeiuTxP-' -O data/bibtex_stratified10folds_meka/Bibtex-fold2.arff
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1r0eFBbnEb15yCANBzBpqR7c7pJF53HI5' -O data/bibtex_stratified10folds_meka/Bibtex-fold3.arff
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=11LOW-yFq1EIM4sWy0XmMPPC7Rv2mjhFd' -O data/bibtex_stratified10folds_meka/Bibtex-fold4.arff
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1xo7IWMO6Cc0d6wkf051VvYm3W196Ns0P' -O data/bibtex_stratified10folds_meka/Bibtex-fold5.arff
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1meV9ldv9XRPL8Akb78G7TvsY_QcfmYqu' -O data/bibtex_stratified10folds_meka/Bibtex-fold6.arff
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=15stGx406I2jwySsfzHdUKvJhUxoDu6rH' -O data/bibtex_stratified10folds_meka/Bibtex-fold7.arff
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1AepSJKrQ_PA0yCUVsbU5gmlW6GRLDtCd' -O data/bibtex_stratified10folds_meka/Bibtex-fold8.arff
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ATz5E7GkZ0Kh5i6_uBCqAp7bdMEGg7an' -O data/bibtex_stratified10folds_meka/Bibtex-fold9.arff
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1rvmycPjHD4Juvnf6DjwEBFhhNquheivy' -O data/bibtex_stratified10folds_meka/Bibtex-fold10.arff
