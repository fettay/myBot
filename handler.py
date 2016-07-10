# -*- coding: utf-8 -*-
import keywords_fr
import json
import numpy as np
import pandas as pd
import datefinder
import datetime
import classifier

DEFAULT_ANSWER = "Desole je n'ai pas compris votre demande."
DATA_LOC = 'Data/'
data = json.load(open(DATA_LOC + 'Prod_prix.json'))
list_prod = data[0]
list_prix = data[1]
PRODUCTS = pd.read_csv(DATA_LOC + 'Product.csv')
SHOPS = pd.read_csv(DATA_LOC + 'Shops.csv')
df_shop = pd.read_csv(DATA_LOC + 'mag.csv')
DAY_TRANSLATATION = {'Monday':'lundi', 'Tuesday': 'mardi', 'Wednesday': 'mercredi', 'Thursday': 'jeudi',
                     'Friday': 'vendredi', 'Saturday': 'samedi', 'Sunday': 'dimanche'}
ALL_OPT = ['product_price', 'shop_hours', 'shop_location', 'shop_telephone']
DATA_CONTAINERS = {'product_price': ('PRODUCTS', 'price'), 'shop_hours': ('SHOPS', '*day'),
                   'shop_location': ('SHOPS', 'Adresse'), 'shop_telephone': ('SHOPS', 'telephone')}



class Handler(object):
    """
    Receive a sentence and return the relevant answer
    """
    def __init__(self, opt_list, products=None, shops=None):

        if any([opt not in ALL_OPT for opt in opt_list]):
            raise ValueError("Unknown option.")

        self.OPT_LIST = opt_list
        self.PRODUCTS = products
        self.SHOPS = shops
        self.format_files()

    def format_files(self):
        """
        Only add a column with a bag of words in all dataframes
        :return: None
        """
        if self.SHOPS is not None:
            def filter_shops(x):
                res = " ".join([elem for elem in x['Adresse'].split(" ") if not elem.isdigit()])
                try:
                    res += " " + x['city']
                except TypeError:
                    pass
                return res.lower()
            self.SHOPS["Words"] = self.SHOPS.apply(filter_shops, axis=1)
        if self.PRODUCTS is not None:
            self.PRODUCTS["Words"] = self.PRODUCTS.apply(lambda x: x["product"].lower(), axis=1)

    def classify(self, sentence):
        list_df = {'SHOPS': self.SHOPS, 'PRODUCTS': self.PRODUCTS}
        return classifier.dummy_classifier(sentence, list_df, self.OPT_LIST)


    def responses_formatter(self, result, sentence):
        """
        :param result: Result from classifier
        :return: Formatted string
        """
        class_, status, all_prod = result
        if status == 0:
            res_lines = getattr(self, DATA_CONTAINERS[class_][0]).iloc[all_prod]
            target = res_lines.iloc[0]
            if class_ == 'product_price':
                return "Le prix du {} est de {} euros".format(target["product"], target["price"])
            elif class_ == 'shop_hours':
                date = datefinder.find_dates(sentence)
                if len(date) == 1:
                    day_of_week = date[0].strftime('%A')
                elif len(date) > 1:
                    return "Je ne sais pas gérer plusieurs dates à la fois."
                else:
                    day_of_week = datetime.datetime.now().strftime('%A')
                res = target[day_of_week]
                if isinstance(res, float):
                    return "Le magasin est fermé le {}".format(DAY_TRANSLATATION[day_of_week])
                return "Le magasin ouvre le {} à {} jusqu'à {}." .format(DAY_TRANSLATATION[day_of_week], res.split('-')[0],
                                                                         res.split('-')[1])
            elif class_ == 'shop_telephone':
                return "Le numero du magasin est le: {}" % target['telephone']
            elif class_ == 'place':
                return "Le magasin se trouve au {}" % target["Adresse"]
            return DEFAULT_ANSWER
        elif status == 1:
            res_lines = getattr(self, DATA_CONTAINERS[class_][0]).iloc[all_prod]
            if DATA_CONTAINERS[class_][0] == 'SHOPS':
                list_res = (res_lines["city"] + res_lines["Adresse"]).values
                return "De quelle boutique parlez vous? \n{}".format("\n".join(list_res))
            elif DATA_CONTAINERS[class_][0] == 'PRODUCTS':
                return "De quel article parlez vous? \n{}".format("\n".join(res_lines["product"].values))
            return DEFAULT_ANSWER
        else:  # -1
            return DEFAULT_ANSWER


def get_response(sentence):
    keywords = keywords_fr.extract(sentence)
    result_product = check_prices(keywords)
    if result_product['product'] == 0:
        result_shop = check_shops(keywords, sentence)
        result_shop = handle_shop_results(result_shop)
        if result_shop == 0:
            return "Désolé je ne comprends pas ce que vous voulez dire."
        else:
            return result_shop
    if result_product['found']:
        return "Le prix du %s est de %s euros." % (result_product["product"], result_product["price"])
    else:
        return "De quel %s parlez vous? \n%s" % (result_product["product"], result_product["products"])


def check_prices(keywords):
    keywords_set = keywords
    common = [set(prod).intersection(keywords_set) for prod in list_prod]
    scores = [len(com) for com in common]
    args_val = np.argwhere(scores == np.amax(scores)).flatten().tolist()
    max_elem = args_val[0]
    if len(args_val) == 1:
        return {'product': " ".join(list_prod[max_elem]).title(), 'price': list_prix[max_elem], 'found': True}
    # Check if all common are the same
    elif scores[max_elem] > 0 and any([" ".join(common[max_elem]) == " ".join(common[i]) for i in args_val]):
        list_comm = [" ".join(list_prod[i]) for i in args_val]
        return {'product': " ".join(common[max_elem]), 'price': 0, 'found': False,
                'products': "\n".join(list_comm).title()}
    else:
        return {'product': 0, 'price': 0, 'found': False}


def check_shops(keywords, sentence):
    keywords_set = set(keywords)
    all_words = [keywords_fr.extract(elem) for elem in df_shop["Words"]]
    common = [set(mag).intersection(keywords_set) for mag in all_words]
    scores = [len(com) for com in common]
    args_val = np.argwhere(scores == np.amax(scores)).flatten().tolist()
    max_elem = args_val[0]
    if len(args_val) == 1:
        if len(keywords_set.intersection({'num', 'telephone', 'numero'})) > 0:
            return {'type_': 'number', 'product': df_shop["Code de magasin"].iloc[max_elem],
                    'result': df_shop[u"Numéro de téléphone principal"].iloc[max_elem]}
        dates = datefinder.find_dates(sentence)
        if len(dates) == 1:
            day_of_week = dates[0].strftime('%A')
            return {'type_': 'hours', 'product': df_shop["Code de magasin"].iloc[max_elem],
                                'result': df_shop[day_of_week].iloc[max_elem], 'additional_data':
                        {'day_of_week': DAY_TRANSLATATION[day_of_week]}}
        # Default send today's
        elif len(keywords_set.intersection({'horaires', 'heure', 'ouvre', 'ferme'})) > 0:
            day_of_week = datetime.datetime.now().strftime('%A')
            return {'type_': 'hours', 'product': df_shop["Code de magasin"].iloc[max_elem],
                    'result': df_shop[day_of_week].iloc[max_elem], 'additional_data':
                        {'day_of_week': DAY_TRANSLATATION[day_of_week]}}
        elif len(keywords_set.intersection({'où', 'se trouve', 'ou est', 'adresse'})) > 0:
            return {'type_': 'place', 'product': df_shop["Code de magasin"].iloc[max_elem],
                'result': df_shop["Ligne d'adresse 1"].iloc[max_elem]}


def handle_shop_results(results):
    if results is not None:
        if results['type_'] == 'hours':
            if isinstance(results['result'], float):
                return "Le magasin est fermé le %s." % results['additional_data']['day_of_week']
            return "Le magasin ouvre le %s à %s jusqu'à %s." % (results['additional_data']['day_of_week'],
                                                                 results['result'].split('-')[0],
                                                                 results['result'].split('-')[1])
        if results['type_'] == 'place':
            return "Le magasin se trouve au %s" % results['result']
        if results['type_'] == 'number':
            return "Le numero du magasin est le: " % results['result']
    return 0


if __name__ == '__main__':
    assert(keywords_fr.extract("Je suis juif") == set(["juif"]))
    # print(get_response("Hello quel est prix du sac candide Large Cuir"))
    # print(get_response("Hello a quelle heure ouvre le magasin rue francois 1er"))
    # print(get_response("Hello a quelle heure ouvre le magasin rue francois 1er demain"))
    # print(get_response("Hello quand ouvre le magasin de cannes rue d'antibes 1er demain"))
    # print(get_response("Hello"))
    # print get_response(u'La boutique de Lille rue de la monnaie est elle ouverte le 16 Juin prochain ?')
    hdl = Handler(opt_list=ALL_OPT, shops=SHOPS, products=PRODUCTS)
    while True:
        sentence = raw_input("What do you want to test? ")
        cls_result = hdl.classify(sentence)
        print(hdl.responses_formatter(cls_result, sentence))