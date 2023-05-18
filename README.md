# Graph Deep Learning @ Università della Svizzera Italiana
## S3GC Project SP 2023
Navdeep Singh Bedi, Sepehr Beheshti, Cristiano Colangelo


DataSet:
you can find all necessary datasets in this drive:
https://drive.google.com/drive/folders/17uN67e6uZiOlzFeloU7t_DoOZezBt0Yp

These datasets must be deposited on datasets folder one level back to S3GC folder. (../datasets/)

Don't worry, if you don't donwload or the code doesnt find the datasets, it will just donwload them.

### ###############  How to run different codes ############### ###
### ################## ################## ################## ######

### S3GC code
to run S3GC on different dataset, you need to run S3GC.py by adding the name of dataset at the end of command.

all datasets are: {'Cora', 'Citeseer', 'Pubmed','ogbn-arxiv' , 'reddit','ogbn-products'}

first go to the folder of S3GC in the repo then for different datasets please run this commands:


cora          : `./python3 S3GC.py cora`

citeseer      : `./python3 S3GC.py citeseer`

pubmed        : `./python3 S3GC.py pubmed`

ogbn-arxiv    : `./python3 S3GC.py ogbn-arxiv`

reddit        : `./python3 S3GC.py reddit`

ogbn-products : `./python3 S3GC.py ogbn-products`



### DGI Code

first go to the folder of DGI in the repo then for different datasets please run this commands:

cora          : `./python3 DGI.py cora`

citeseer      : `./python3 DGI.py citeseer`

pubmed        : `./python3 DGI.py pubmed`

ogbn-arxiv    : `./python3 DGI.py ogbn-arxiv`

reddit        : `./python3 DGI.py reddit`

ogbn-products : `./python3 DGI.py ogbn-products`

### Node2Vec code


first go to the folder of DGI in the repo then for different datasets please run this commands:

cora          : `./python3 node2vec.py cora`

citeseer      : `./python3 node2vec.py citeseer`

pubmed        : `./python3 node2vec.py pubmed`

ogbn-arxiv    : `./python3 node2vec.py ogbn-arxiv`

reddit        : `./python3 node2vec.py reddit`

ogbn-products : `./python3 node2vec.py ogbn-products`