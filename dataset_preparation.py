import sys
from Dataset.preprocessor import Dataset_preparation

if __name__ == '__main__':
    if sys.argv[1].lower().startswith('jhu'):
        Dataset_preparation.JHU()
        print('jhu++ dataset is prepared.')
    elif (sys.argv[1].lower().find('shtc') >= 0 or sys.argv[1].lower().find(
            'shanghai') >= 0) and sys.argv[1].lower().find('a') >= 0:
        Dataset_preparation.SHTCA()
        print('SHTCA dataset is prepared.')
    elif (sys.argv[1].lower().find('shtc') >= 0 or sys.argv[1].lower().find(
            'shanghai') >= 0) and sys.argv[1].lower().find('b') >= 0:
        Dataset_preparation.SHTCB()
        print('SHTCB dataset is prepared.')
    elif sys.argv[1].lower().find('qnrf') >= 0:
        Dataset_preparation.QNRF()
        print('QNRF dataset is prepared.')
    elif sys.argv[1].lower().find('nwpu') >= 0:
        Dataset_preparation.NWPU()
        print('NWPU dataset is prepared.')
    else:
        raise NotImplementedError