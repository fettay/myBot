__author__ = 'raphaelfettaya'
import csv
import unicodecsv
import pandas as pd
import csvImporter
def unicode_csv_reader(utf8_data, dialect=csv.excel, **kwargs):
    csv_reader = csv.reader(utf8_data, dialect=dialect, **kwargs)
    for row in csv_reader:
        yield [cell.decode('utf-8') for cell in row]


if __name__ == '__main__':
    data = unicodecsv.reader(open('mag2.txt'), encoding='utf-8')
    for elm in data:
        print(elm)
    data = unicode_csv_reader(open('mag2.txt'))
    for elm in data:
        print(elm)
print pd.read_csv('mag2.txt')['Words']
