# Perplexity-correlation-data 

## To do 

-[ ] Let's expand our pool to 30B tokens
    - Can we train on 3 subsets of size 3B:
        - Fully random
        - Preselect from that - can you save all the fasttext scores and plot a histogram of them
        - Random, but select same # from each domain as 2
-[ ] Evaluation: SciQ ARC-E ARC-C LogiQA OBQA HellaSwag PIQA WinoGrande LAMBADA RACE SciQ BBH
-[ ] histogram of pairwise similarities for each document for top 5 represented domains as well
-[ ] metadata map: mapping domain -> list of tuple of id, fasttext score

## note 
pip install numpy==1.23.5
