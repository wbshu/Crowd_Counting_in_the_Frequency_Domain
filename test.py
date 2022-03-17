from Utils import model_test as mt
from Model import vgg_19
import sys


if __name__ == '__main__':
    d = mt.Chf_Model_Test(vgg_19.vgg19(),sys.argv[2])
    if sys.argv[1].lower().find('qnrf') >= 0:
        d.evaluate_on_test_set(sys.argv[1], sys.argv[1]+'_result.txt', max_size=1536)
    elif sys.argv[1].lower().find('nwpu') >= 0:
        d.evaluate_on_NWPU_test(file='nwpu_result.txt')
    else:
        d.evaluate_on_test_set(sys.argv[1], sys.argv[1]+'_result.txt')
