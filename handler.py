# -*- coding: utf-8 -*-
import keywords_fr
import json
import numpy as np
import pandas as pd
import datefinder
import datetime
import classifier
from unidecode import unidecode
import utils

DEFAULT_ANSWER = "Desole je n'ai pas compris votre demande."
DAY_TRANSLATATION = {'Monday':'lundi', 'Tuesday': 'mardi', 'Wednesday': 'mercredi', 'Thursday': 'jeudi',
                     'Friday': 'vendredi', 'Saturday': 'samedi', 'Sunday': 'dimanche'}
ALL_OPT = ['product_price', 'shop_hours', 'shop_location', 'shop_telephone', 'product_view']
DATA_CONTAINERS = {'product_price': ('PRODUCTS', 'price'), 'shop_hours': ('SHOPS', '*day'),
                   'shop_location': ('SHOPS', 'Adresse'), 'shop_telephone': ('SHOPS', 'telephone'),
                   'product_view':('PRODUCTS', 'price')}


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
                res = " ".join([unidecode(elem) for elem in x['Adresse'].split(" ") if not elem.isdigit()])
                # res = " ".join([unidecode(elem.decode('utf-8')) for elem in x['Adresse'].split(" ") if not elem.isdigit()]) #PY2
                try:
                    res += " " + unidecode(x['city'])
                    # res += " " + unidecode(x['city'].decode('utf-8')) #PY2
                except TypeError:
                    pass
                return res.lower()
            self.SHOPS["Words"] = self.SHOPS.apply(filter_shops, axis=1)

        if self.PRODUCTS is not None:
            def filter_products(x):
                words = x["product"].lower().split()
                words.extend([utils.plural(word) for word in words])  # Add plurals
                return " ".join(words)

            self.PRODUCTS["Words"] = self.PRODUCTS.apply(filter_products, axis=1)

    def classify(self, sentence, class_=None):
        """
        :param sentence: sentence received
        :param class_: if a class_ is already computed
        :return: formatted result see classifier
        """
        list_df = {'SHOPS': self.SHOPS, 'PRODUCTS': self.PRODUCTS}
        if class_ is None:
            return classifier.log_classifier(sentence, list_df)
        else:
            return classifier.item_finder(sentence, list_df, class_)

    def responses_formatter(self, result, sentence):
        """
        :param result: Result from classifier
        :return:Tuple API action, Formatted string example: (send_message, "haha")
        """
        class_, status, all_prod = result
        if status == 0:
            res_lines = getattr(self, DATA_CONTAINERS[class_][0]).iloc[all_prod]
            target = res_lines.iloc[0]
            if class_ == 'product_price':
                return "send_message", "Le prix du {} est de {} euros".format(target["product"], target["price"])
            elif class_ == 'shop_hours':
                day_of_week = utils.extract_day(sentence)
                res = target[day_of_week]
                if isinstance(res, float):
                    return "send_message", "Le magasin est fermé le {}".format(DAY_TRANSLATATION[day_of_week])
                return "send_message", "Le magasin ouvre le {} à {} jusqu'à {}." .format(DAY_TRANSLATATION[day_of_week],
                                                                                  res.split('-')[0], res.split('-')[1])
            elif class_ == 'shop_telephone':
                return "send_message", "Le numero du magasin est le: {}" % target['telephone']
            elif class_ == 'shop_location':
                return "send_message", "Le magasin se trouve au {}" % target["Adresse"]
            return "send_message", DEFAULT_ANSWER
        elif status == 1:
            res_lines = getattr(self, DATA_CONTAINERS[class_][0]).iloc[all_prod]
            # if class_ == "product_view":
            if DATA_CONTAINERS[class_][0] == 'PRODUCTS':
                return "send_carousel", format_carousel(res_lines)
            if DATA_CONTAINERS[class_][0] == 'SHOPS':
                desc = res_lines.apply(lambda x:  x["city"] + "".join(el for el in x["Adresse"]
                                                                      if not el.isdigit()) + ", ", axis=1)
                data_col = DATA_CONTAINERS[class_][1]
                if class_ == 'shop_hours':
                    data_col = utils.extract_day(sentence)
                list_res = (desc + res_lines[data_col]).values
                return "send_message", "Voilà mes infos: \n{}".format("\n".join(list_res))
                # return "send_message", "De quel article parlez vous? \n{}".format("\n".join(res_lines["product"].values))
            return "send_message", DEFAULT_ANSWER
        else:  # -1
            if DATA_CONTAINERS[class_][0] == 'PRODUCTS':
                return "send_message", "Désolé je ne connais pas cette article"
            elif DATA_CONTAINERS[class_][0] == 'SHOPS':
                return "Désolé je ne connais pas cette boutique"
            return "send_message", "Désolé je ne comprend pas ce que vous voulez dire."
        # return "send_message", DEFAULT_ANSWER


def format_carousel(product_list):
    data = {
        "message": {
            "attachment": {
                "type": "template",
                "payload": {
                    "template_type": "generic",
                    "elements": []
                }
            }
        }
    }
    for _, p in product_list.iterrows():
        format_value = {
            "title": p["product"],
            "image_url": p["picture"],
            "subtitle": "%d€" % p["price"],
            "buttons": [
                {
                    "type": "web_url",
                    "url": "www.zadigetvoltaire.com",
                    "title": "Voir le produit sur site"
                }
            ]
        }
        data["message"]["attachment"]["payload"]["elements"].append(format_value)
    return data

if __name__ == '__main__':
    DATA_LOC = 'Data/'
    PRODUCTS = pd.read_csv(DATA_LOC + 'Product.csv').fillna('')
    SHOPS = pd.read_csv(DATA_LOC + 'Shops.csv').fillna('')
    assert(keywords_fr.extract("Je suis juif") == set(["juif"]))
    hdl = Handler(opt_list=ALL_OPT, shops=SHOPS, products=PRODUCTS)
    while True:
        sentence = raw_input("What do you want to test? ")
        cls_result = hdl.classify(sentence)
        print(hdl.responses_formatter(cls_result, sentence))