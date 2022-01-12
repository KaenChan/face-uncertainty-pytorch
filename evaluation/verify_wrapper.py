import os
import numpy as np
# from evaluation import verification
from evaluation import verification_risk_fnmr as verification


class VerifyWrapper(object):
    def __init__(self, preprocess_func):
        test_data_dir_mx = r'F:\data\metric-learning\face\ms1m-retinaface-t1'
        # test_1v1_target = 'calfw,cfp_fp,agedb_30'
        # test_1v1_target = 'cplfw,cfp_fp,agedb_30'
        test_1v1_target = 'cfp_fp'
        self.ver_list = []
        self.ver_name_list = []
        for name in test_1v1_target.split(','):
            path = os.path.join(test_data_dir_mx, name + ".bin")
            if os.path.exists(path):
                image_size = [112, 112]
                data_list, issame_list = verification.load_bin(path, image_size)
                data_list = data_list[0].asnumpy()
                images = preprocess_func(data_list, is_training=False)
                data_set = (images, issame_list)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)
                print('ver', name)

    def run(self, extract_feature_func, sigma_sizes=1):
        info_ver = ''
        ret_ver = {}
        for i in range(len(self.ver_list)):
            print('---', self.ver_name_list[i], '---')
            info_ver_, ret = verification.eval_images(
                self.ver_list[i][0], self.ver_list[i][1],
                extract_feature_func, 128, 10, sigma_sizes=sigma_sizes, only_mls=False)
            # print(info_ver_)
            info_ver += '---' + self.ver_name_list[i] + '\n'
            info_ver += info_ver_ + '\n'
            ret_ver[self.ver_name_list[i]] = ret
        return info_ver
