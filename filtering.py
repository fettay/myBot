__author__ = 'raphaelfettaya'
import re
"""
Handling filters
"""
CORRESPONDING_SIZE = {'extra small': 'XS', 'small': 'S', 'medium': 'M', 'large': 'L', 'extra large': 'XL'}
FILTER_TYPES = ['price_plus', 'price_less', 'price_order', 'gender', 'size', 'collection', 'discount']
COLORS = ['blanc', 'kaki', 'argent', 'or','noir','gris','multicolor','jaune','bleu','rouge','beige','vert','rose',
          'marine','marron','orange','creme']
MASKS = {'price': "\d{1,10}(?=>\s{0,3}euros|\s{0,3}euro|\s{0,3}eur|\s{0,3}e|\s{0,3}€)", 'size': "taille\s{0,3}[1-5]|[3-5]\d\W(?!\s{0,3}e|\s{0,3}eur|\s{0,3}euros|\s{0,3}euro|\s{0,3}€)|\d{1,2} ans|\d{1,2} mois|(?<=\s)xs+(?=\s|$|\.|;|\n|\b|,)|extra.small|(?<=\s)s+(?=\s|$|\.|;|\n|\b|,)|small|\|(?<=\s)m+(?=\s|$|\.|;|\n|\b|,)|medium|(?<=\s)xl+(?=\s|$|\.|;|\n|\b|,)|extra.large|(?<=\s)l+(?=\s|$|\.|;|\n|\b|,)|large",
         'gender': "homme|femme|fille|garçon|garcon", 'collection':"nouv+(?:eau|elle|el|eaux|o)|printemps|été|hiver|automne",
         'discount':"discount|solde|soldé+(?=$|e|es|s|.|,|\n)|reduction",
         'color': "|".join(["(?<=\s){}(?=\s|$|\.|;|\n|\b|,)".format(col) for col in COLORS])
         }
TAILLE_UNIQUE = "T.U"
# PATTERN = """
#         (?P<a>{price})
#         |
#         (?P<b>{size})
#         |
#         (?P<c>{gender})
#         |
#         (?P<d>{collection})
#         |
#         (?P<e>{discount})
#     """.format(**MASKS)
PATTERNS = {k:re.compile(v) for k, v in MASKS.items()}


def generate_filters(sentence, filters_dict):
    result = {}
    for k, v in filters_dict.items():
        if k == 'price':
           if 'plus de' in sentence: # Maybe not needed
               result['price_plus'] = v[0]
           else:
               result['price_less'] = v[0]
        elif k == 'size':
            v_fix = [CORRESPONDING_SIZE[el] for el in v if el in CORRESPONDING_SIZE]
            result['size'] = [el.upper() if el in ['xs', 's', 'm', 'l', 'xl'] else el for el in v_fix]
            result['size'].append(TAILLE_UNIQUE)
        elif k == 'gender':
            if len(set(v).intersection(set(['fille', 'femme']))):
                result['gender'] = ['female']
            if len(set(v).intersection(set(['homme', 'garcon', 'garçon']))):
                result['gender'] = ['male']
            result['gender'].append('unisex')
        elif k == 'collection':
            pass
        elif k == 'discount':
            result['discount'] = True
        elif k == 'color':
            result['color'] = v
    return result


def match_sentence(sentence):
    results = {}
    for k, v in PATTERNS.items():
        res = v.findall(sentence)
        if len(res) > 0:
            results[k] = res
    return results


# TODO MAKE IT FASTER (ALL CONDITION IN ONE LOOP?)
def filter_results(df_data, res_indices, filters):
    df_res = df_data.iloc[res_indices]
    for k, v in filters.items():
        if k == 'price_plus':
            df_price = df_res['price'].apply(lambda x: float(re.findall("\d+\.\d+", x)[0]))
            df_res = df_res[df_price >= int(v)]
        elif k == 'price_less':
            df_price = df_res['price'].apply(lambda x: float(re.findall("\d+\.\d+", x)[0]))
            df_res = df_res[df_price <= int(v)]
        elif k == 'size':
            df_mask = df_res['size'].apply(lambda x: any([s in x for s in v]))
            df_res = df_res[df_mask]
        elif k == 'gender':
            df_mask = df_res['gender'].apply(lambda x: any([s == x for s in v]))
            df_res = df_res[df_mask]
        elif k == 'color':
            df_mask = df_res['color'].apply(lambda x: any([s == x for s in v]))
            df_res = df_res[df_mask]
        elif k == 'discount':
            pass #TODO handle it
            # if v:
            #     df_res = df_res['']
    return [int(elem) for elem in df_res.index]  # Convert to regular int

def filter_main(df_data, res_indices, sentence):
    filters = match_sentence(sentence)
    filters = generate_filters(sentence, filters)
    return filter_results(df_data, res_indices, filters)

if __name__ == '__main__':
    sentence = "Je veux un t-shirt bleu en extra small ou en 36 pour homme a moins de 100 eur et soldé mais nouveau."
    import pandas as pd
    PRODUCTS = pd.read_csv('Data/Lengow.csv').fillna('')
    print(filter_main(PRODUCTS, [i for i in range(len(PRODUCTS.index))], sentence))
