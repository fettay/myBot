import re
from string import punctuation
from unidecode import unidecode

RGX_HOUR = '([0-9]{1,2}[h|H|:][0-9]{0,2})'
RPL_HOUR = 'XXhXX'
CURRENCIES = 'eur|e|usd|gbp|euro|euros|eu|dollar|dollars'
RGX_PRICE = '([0-9]{1,}\s*)(%s)(\W|\s|$)|(\s)(%s)(\s*[0-9]{1,})' % (CURRENCIES, CURRENCIES)
MONTHS = 'janvier|fevrier|mars|avril|mai|juin|juillet|aout|septembre|octobre|novembre|decembre|jan|fev|mar|avr|sep|sept|oct|nov|dec'
RGX_DATA = '([0-9]{1,}\s*)(%s)(\s*[0-9]{2,4})|([0-9]{1,}\s*)(%s)' % (MONTHS, MONTHS)


def replace_time(string_):
    times_ = re.findall(RGX_HOUR, string_)
    for i in times_:
        string_ = string_.replace(i, 'XXTIMEXX')
    return string_


def replace_date(string_):
    dates_ = re.findall(RGX_DATA, string_)
    dates_ = [''.join(x) for x in dates_]
    for i in dates_:
        string_ = string_.replace(i, 'XXDATEXX')
    return string_


def replace_price(string_):
    prices_ = re.findall(RGX_PRICE, string_)
    prices_ = [''.join(x) for x in prices_]
    for i in prices_:
        string_ = string_.replace(i, 'XXPRICEXX')
    return string_


def remove_symbols(string_):
    # special case for the French word `aujourd'hui`
    ajd = "aujourd'hui"
    string_ = string_.replace(ajd, 'aujourdhui')
    for punctuation_ in punctuation:
        string_ = string_.replace(punctuation_, ' ')
    return string_


def clean_string(string_):
    return remove_symbols(
                replace_date(
                    replace_time(
                        replace_price(
                            unidecode(
                                (string_.lower()))))))
