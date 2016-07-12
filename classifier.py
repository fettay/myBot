# -*- coding: utf-8 -*-

import keywords_fr
import handler
import numpy as np
import datefinder

DUMMY_VOCAB = {
    'product_price': ['prix', 'coute', 'coûte'],
    'shop_hours': ['heure', 'ouvre', 'horaires', 'ferme', '*dates', 'boutique', 'magasin'],
    'shop_location': ['où', 'adresse', 'boutique', 'magasin'],
    'shop_telephone': ['numero', 'telephone', 'numéro', 'téléphone', 'boutique', 'magasin'],
    'product_view': ['photos', 'photo']
}

DEFAULT_TO_KEEP = ['où']


def dummy_classifier(sentence, dict_df, opt_list):
    """
    Compare result with other heuristics values
    :param sentence:
    :param dict_df:
    :param opt_list:
    :return: class, categorie_of_result, list of relevant result indice
    """

    def compare_kw(keywords_set, list_data):
        common = [set(prod.split(' ')).intersection(keywords_set) for prod in list_data]
        scores = [len(com) for com in common]
        args_val = np.argwhere(scores == np.amax(scores)).flatten().tolist()
        return common, scores, args_val

    def compute_score(kw_compare_res):
        common, scores, args_val = kw_compare_res
        return scores[args_val[0]] if len(args_val)/len(scores) < .1 else 0


    # FIND PROD AND SHOPS
    keywords_set = keywords_fr.extract(sentence, to_keep=DEFAULT_TO_KEEP)
    scores_classes = {k: 0 for k in opt_list}
    df_comparitions = {k: compare_kw(keywords_set, v['Words']) for k, v in dict_df.items()}
    df_scores = {k: compute_score(df_comparitions[k]) for k, v in dict_df.items()}
    dict_comparitions = {k: compare_kw(keywords_set, DUMMY_VOCAB[k]) for k in opt_list}
    for k in scores_classes:
        scores_classes[k] += df_scores[handler.DATA_CONTAINERS[k][0]]
        kw_res = dict_comparitions[k]
        scores_classes[k] += 3 * kw_res[1][kw_res[2][0]]

    # FIND DATES
    dates = datefinder.find_dates(sentence)
    if len(dates) > 0:
        for k in scores_classes:
            if '*dates' in DUMMY_VOCAB[k]:
                scores_classes[k] += 2

    max_class = max(scores_classes, key=scores_classes.get)
    max_scores = scores_classes[max_class]
    max_res = df_comparitions[handler.DATA_CONTAINERS[max_class][0]]


    # GET CATEGORIE
    if max_scores == 0:  # No class found
        return None, -1, []
    if len(max_res[2]) > 1 and len(max_res[2]) < .1 * len(max_res[1]) :  # List of relevant items
        return max_class, 1, max_res[2]
    else:  # All good
        return max_class, 0, max_res[2]