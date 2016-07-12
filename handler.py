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
        :return:Tuple API action, Formatted string example: (send_message, "haha")
        """
        class_, status, all_prod = result
        if status == 0:
            res_lines = getattr(self, DATA_CONTAINERS[class_][0]).iloc[all_prod]
            target = res_lines.iloc[0]
            if class_ == 'product_price':
                return "send_message", "Le prix du {} est de {} euros".format(target["product"], target["price"])
            elif class_ == 'shop_hours':
                date = datefinder.find_dates(sentence)
                if len(date) == 1:
                    day_of_week = date[0].strftime('%A')
                elif len(date) > 1:
                    return "send_message", "Je ne sais pas gérer plusieurs dates à la fois."
                else:
                    day_of_week = datetime.datetime.now().strftime('%A')
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
            if class_ == "product_view":
                return "send_carousel", format_carousel(res_lines)
            if DATA_CONTAINERS[class_][0] == 'SHOPS':
                list_res = (res_lines["city"] + res_lines["Adresse"]).values
                return "send_message", "De quelle boutique parlez vous? \n{}".format("\n".join(list_res))
            elif DATA_CONTAINERS[class_][0] == 'PRODUCTS':
                return "send_message", "De quel article parlez vous? \n{}".format("\n".join(res_lines["product"].values))
            return "send_message", DEFAULT_ANSWER
        else:  # -1
            return "send_message", DEFAULT_ANSWER


def format_carousel(product_list):
    data = {
        "message": {
            "attachement": {
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
            "subtitle": p["price"] + "€",
            "buttons": [
                {
                    "type": "web_url",
                    "url": "www.zadigetvoltaire.com",
                    "title": "Voir le produit sur site"
                }
            ]
        }
        data["message"]["attachement"]["payload"]["elements"].append(format_value)
    return data

if __name__ == '__main__':
    assert(keywords_fr.extract("Je suis juif") == set(["juif"]))

    hdl = Handler(opt_list=ALL_OPT, shops=SHOPS, products=PRODUCTS)
    while True:
        sentence = raw_input("What do you want to test? ")
        cls_result = hdl.classify(sentence)
        print(hdl.responses_formatter(cls_result, sentence))