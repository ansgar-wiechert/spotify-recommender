from songify.recommender.transformer import Transformer

def return_transformer():
    trans = Transformer("pytorch")
    return trans.check_version()