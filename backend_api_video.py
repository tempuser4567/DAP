# -*- coding:utf-8 -*-
import docker
import os, sys
import shutil
import time
import traceback
import json
import numpy as np
import psutil
from sklearn.metrics import accuracy_score
from PIL import Image
sys.path.append(".")
from utils import get_metrics, get_topdown, dict_append, read_result_vft
from metrics import AP, AR
from eval_box.module_map import module_map, module_map_reverse, type_map, type_map_reverse
from ipdb import set_trace


class BackEndVideo():
    def __init__(self):
        self.root = os.path.abspath('.')
        self.users = os.path.join(self.root, 'users', 'video')
        self.eval_box = os.path.join(self.root, 'eval_box')
        self.debug_box = os.path.join(self.root, 'debug_box')
        self.debug_data = os.path.join(self.debug_box, 'debug_data')
        self.debug_video_data = os.path.join(self.debug_box, 'debug_video_data')

        self.client = docker.from_env()
        self.containers = {}
        self.module_paths=module_map

    def run_debug(self, user_id, task_id, types, cuda_id='0,1', rm_img=True):
        
        os.makedirs(os.path.join(self.get_taskpath(user_id, task_id),"debug_outputs"), exist_ok=True)
        dockerfile_path = os.path.join(self.get_taskpath(user_id, task_id), 'dockerfile')
        image_name = self.get_imagename(user_id, task_id, 'debug')
        try:
            self.image_build(task_id=task_id, image_name=image_name, dockerfile_path=dockerfile_path)
        except (docker.errors.BuildError, docker.errors.APIError) as e:
            return 'docker build error\n---------------\n' + traceback.format_exc()
        for type in types:
            if type == 'video':
                data = self.debug_video_data
            elif type == 'frame':
                data = self.debug_frame_data
            elif type == 'face':
                data = self.debug_data
            command = "python debug/debug.py"# " python debug/debug.py"
            mount_list = [
                f'{os.path.join(self.get_taskpath(user_id, task_id), "scripts")}:/taskroot/scripts',
                f'{os.path.join(self.get_taskpath(user_id, task_id), "debug_outputs")}:/taskroot/outputs',
                f'{data}:/taskroot/data',
                f'{self.debug_box}:/taskroot/debug'
            ]

            container = self.client.containers.run(
                image=image_name,
                command=command,
                name=f'debug_{user_id}_{task_id}',
                volumes=mount_list,
                working_dir='/taskroot',
                device_requests=[docker.types.DeviceRequest(device_ids=[str(cuda_id)], capabilities=[['gpu']])],
                network_mode='bridge',
                detach=True,
                #user='1000:1000'
            )

            '''----------block mode-----------'''
            self.containers[f'{user_id}_{task_id}'] = {
                'container': container,
                'container_id': container.id,
                'task': 'debug'
            }

            container.wait()
            # set_trace()
            log = str(container.logs(), encoding='utf-8')

            container.remove()
            if rm_img:
                self.client.images.remove(image_name, force=True)
            self.containers.pop(f'{user_id}_{task_id}')

        # check if success
            if os.path.exists(os.path.join(self.get_taskpath(user_id, task_id), "debug_outputs", 'success')):
                # os.remove(os.path.join(self.get_taskpath(user_id, task_id), "outputs", 'success'))
                return 0
            else:
                return "debuging error\n---------------\n" + log

    def run_eval(self, user_id, task_id, modules, mem_limit='20g', cuda_id='0,1', rm_img=True):
        p = psutil.Process()
        data_list = self.create_output_dir(user_id, task_id, modules)
        taskroot = self.get_taskpath(user_id, task_id)
        dockerfile_path = os.path.join(taskroot, 'dockerfile')
        image_name = self.get_imagename(user_id, task_id, 'eval')

        self.image_build(task_id=task_id, image_name=image_name, dockerfile_path=dockerfile_path)

        command = "python eval_box/eval_v3.py"
        mount_list = [
            f'{os.path.join(taskroot, "scripts")}:/taskroot/scripts',
            f'{os.path.join(taskroot, "eval_outputs")}:/taskroot/outputs',
            f'{self.eval_box}:/taskroot/eval_box', 
        ]
        mount_list += data_list
        print('start')
        print(p.pid)
        startTime = time.time()
        container = self.client.containers.run(
            image=image_name,
            command=command,
            volumes=mount_list,
            working_dir='/taskroot',
            name=f'eval_{user_id}_{task_id}',
            mem_limit=mem_limit,
            network_mode='bridge',
            device_requests=[docker.types.DeviceRequest(device_ids=[str(cuda_id)], capabilities=[['gpu']])],
            detach=True,
            #user='1000:1000'
        )
        midTime1 = time.time()
        '''----------block mode-----------'''
        container.wait()
        midTime2 = time.time()
        log = str(container.logs(), encoding='utf-8')
        container.remove()
        endTime = time.time()
        if rm_img:
            self.client.images.remove(image_name, force=True)
        print('mid time:', midTime2 - midTime1)
        print('docker time:', endTime - startTime)
        # check if success
        for output in os.listdir(os.path.join(taskroot, "eval_outputs")):
            if os.path.isdir(os.path.join(taskroot, "eval_outputs", output)):
                print(os.path.join(taskroot, "eval_outputs", output))
                if not os.path.exists(os.path.join(taskroot, "eval_outputs", output, 'success')):
                    return f"{output} running error\n---------------\n" + log, user_id, task_id
        return f"running success\n---------------\n" + log, user_id, task_id

    def eval_result_process(self, user_id, task_id):
        taskroot = self.get_taskpath(user_id, task_id)
        root = os.path.join(taskroot, 'eval_outputs')
        final_report = {"result_data": {}}
        outputs = os.listdir(os.path.join(taskroot, 'eval_outputs'))
        outputs.remove('model.txt')
        outputs.sort() # outputs is the list of module name

        with open(os.path.join(taskroot, 'eval_outputs', 'model.txt'), 'r') as f:
            line = f.read()
            final_report['model'] = {'flops': line.split(',')[0], 'params': line.split(',')[1]}
        for output in outputs:
            if (output.split('_')[1] == 'effectiveness' or (output.split('_')[0] == 'face' and output.split('_')[1] == 'generalization' and output.split('_')[2] == 'domain')): # or (output == 'test'):
                datasets = os.listdir(os.path.join(taskroot, 'eval_outputs', output))
                datasets = list(x for x in datasets if os.path.isdir(os.path.join(taskroot, 'eval_outputs', output, x)))

                report = {}
                Acc_list, AUC_list, EER_list, Precision_list, Recall_list, F1_score_list, conf_list, dataset_list, roc_list, pr_list = get_metrics(self.eval_box, taskroot, datasets, output)
                
                #report['dataset'] = dataset_list
                report['roc_curve'] = roc_list
                report['pr_curve'] = pr_list
                report['Acc'] = format(sum(Acc_list) / len(Acc_list) * 100, '.2f')
                report['AUC'] = format(sum(AUC_list) / len(AUC_list), '.4f')
                report['EER'] = format(sum(EER_list) / len(EER_list), '.4f')
                report['Precision'] = format(sum(Precision_list) / len(Precision_list) * 100, '.2f')
                report['Recall'] = format(sum(Recall_list) / len(Recall_list) * 100, '.2f')
                report['F1_score'] = format(sum(F1_score_list) / len(F1_score_list), '.4f')
                report['conf_diff'] = format(sum(conf_list) / len(conf_list), '.4f')
                final_report["result_data"][module_map_reverse[output]] = report

            elif output.split('_')[0] == 'face' and (output.split('_')[1] == 'robustness'):
                types = output.split('_')[-1]
                datasets = os.listdir(os.path.join(taskroot, 'eval_outputs', output))
                datasets = list(x for x in datasets if os.path.isdir(os.path.join(taskroot, 'eval_outputs', output, x)))
                datasets.sort()

                report = {}
                dataset_dict = {}
                for dataset in datasets:
                    dataset_dict = dict_append(dataset_dict, dataset.split('_')[-1], dataset)

                for i, disturb in enumerate(dataset_dict.keys()):
                    Acc_list, AUC_list, EER_list, Precision_list, Recall_list, F1_score_list, conf_list, dataset_list, roc_list, pr_list = get_metrics(self.eval_box, taskroot, dataset_dict[disturb], output)
                    n = str(i)
                    report[n] = {}

                    """ disturb_name = translate_map['%s-%s' % (output.split('_')[2], disturb)]
                    report = dict_append(report, 'map', disturb_name) """
                    report = dict_append(report, 'map', disturb)
                    report[n]['roc_curve'] = roc_list
                    report[n]['pr_curve'] = pr_list
                    report[n]['Acc'] = format(sum(Acc_list) / len(Acc_list) * 100, '.2f')
                    report[n]['AUC'] = format(sum(AUC_list) / len(AUC_list), '.4f')
                    report[n]['EER'] = format(sum(EER_list) / len(EER_list), '.4f')
                    report[n]['Precision'] = format(sum(Precision_list) / len(Precision_list) * 100, '.2f')
                    report[n]['Recall'] = format(sum(Recall_list) / len(Recall_list) * 100, '.2f')
                    report[n]['F1_score'] = format(sum(F1_score_list) / len(F1_score_list), '.4f')
                    report[n]['conf_diff'] = format(sum(conf_list) / len(conf_list), '.4f')
                if module_map_reverse['face_robustness_' + output.split('_')[2]] not in final_report["result_data"]:
                    final_report["result_data"][module_map_reverse['face_robustness_' + output.split('_')[2]]] = {}
                final_report["result_data"][module_map_reverse['face_robustness_' + output.split('_')[2]]][type_map_reverse[types]] = report

            elif output.split('_')[0] == 'face' and output.split('_')[1] == 'security':
                datasets = os.listdir(os.path.join(taskroot, 'eval_outputs', output))
                datasets = list(x for x in datasets if os.path.isdir(os.path.join(taskroot, 'eval_outputs', output, x)))
                datasets.sort()

                report = {}
                dataset_dict = {}
                for dataset in datasets:
                    dataset_dict = dict_append(dataset_dict, dataset.split('_')[-1], dataset)

                for i, disturb in enumerate(dataset_dict.keys()):
                    Acc_list, AUC_list, EER_list, Precision_list, Recall_list, F1_score_list, conf_list, dataset_list, roc_list, pr_list = get_metrics(self.eval_box, taskroot, dataset_dict[disturb], output)
                    n = str(i)
                    report[n] = {}

                    report = dict_append(report, 'map', disturb)
                    report[n]['roc_curve'] = roc_list
                    report[n]['pr_curve'] = pr_list
                    report[n]['Acc'] = format(sum(Acc_list) / len(Acc_list) * 100, '.2f')
                    report[n]['AUC'] = format(sum(AUC_list) / len(AUC_list), '.4f')
                    report[n]['EER'] = format(sum(EER_list) / len(EER_list), '.4f')
                    report[n]['Precision'] = format(sum(Precision_list) / len(Precision_list) * 100, '.2f')
                    report[n]['Recall'] = format(sum(Recall_list) / len(Recall_list) * 100, '.2f')
                    report[n]['F1_score'] = format(sum(F1_score_list) / len(F1_score_list), '.4f')
                    report[n]['conf_diff'] = format(sum(conf_list) / len(conf_list), '.4f')
                final_report["result_data"][module_map_reverse[output]]= report

            elif output.split('_')[0] == 'face' and output.split('_')[1] == 'functionality' and output.split('_')[2] == "mask'localization":
                datasets = os.listdir(os.path.join(taskroot, 'eval_outputs', output))
                datasets = list(x for x in datasets if os.path.isdir(os.path.join(taskroot, 'eval_outputs', output, x)))
                datasets.sort()
                report = {}
                Acc_list, AUC_list, EER_list, Precision_list, Recall_list, F1_score_list, conf_list, dataset_list, roc_list, pr_list = get_metrics(self.eval_box, taskroot, datasets, output)
                report['roc_curve'] = roc_list
                report['pr_curve'] = pr_list
                report['Acc'] = format(sum(Acc_list) / len(Acc_list) * 100, '.2f')
                report['AUC'] = format(sum(AUC_list) / len(AUC_list), '.4f')
                report['EER'] = format(sum(EER_list) / len(EER_list), '.4f')
                report['Precision'] = format(sum(Precision_list) / len(Precision_list) * 100, '.2f')
                report['Recall'] = format(sum(Recall_list) / len(Recall_list) * 100, '.2f')
                report['F1_score'] = format(sum(F1_score_list) / len(F1_score_list), '.4f')
                report['conf_diff'] = format(sum(conf_list) / len(conf_list), '.4f')
                top_dict, down_dict = get_topdown(taskroot, datasets, output)
                os.makedirs(os.path.join(taskroot, 'eval_outputs', 'image_output'))

                for i, key in enumerate(top_dict.keys()):
                    dataset = key.split('_')[0]
                    index = key.split('_')[-1]
                    f_gt = open(os.path.join(self.eval_box, output, 'groundtruth', dataset + '.txt'), 'r')
                    gt_path = f_gt.readlines()[int(index)].split(',')[-1][:-1].replace('/taskroot/data', self.eval_box)
                    report = dict_append(report, 'top5_gt', gt_path)
                    gt_img = Image.open(gt_path)
                    gt_img.save(os.path.join(taskroot, 'eval_outputs', 'image_output', 'top%d_gt.png' % i))
                    gt_img = np.array(gt_img.convert('1'))
                    f_gt.close()
                    mask_path = os.path.join(taskroot, 'eval_outputs', output, dataset, 'output_image', index + '.png')
                    report = dict_append(report, 'top5_mask', mask_path)
                    mask_img = Image.open(mask_path)
                    mask_img.save(os.path.join(taskroot, 'eval_outputs', 'image_output', 'top%d_mask.png' % i))
                    mask_img = np.array(mask_img.convert('1'))
                    report = dict_append(report, 'top5_pred', top_dict[key])
                    report = dict_append(report, 'top5_Acc', accuracy_score(gt_img, mask_img))

                for i, key in enumerate(down_dict.keys()):
                    dataset = key.split('_')[0]
                    index = key.split('_')[-1]
                    f_gt = open(os.path.join(self.eval_box, output, 'groundtruth', dataset + '.txt'), 'r')
                    gt_path = f_gt.readlines()[int(index)].split(',')[-1][:-1].replace('/taskroot/data', self.eval_box)
                    report = dict_append(report, 'down5_gt', gt_path)
                    gt_img = Image.open(gt_path)
                    gt_img.save(os.path.join(taskroot, 'eval_outputs', 'image_output', 'down%d_gt.png' % i))
                    gt_img = np.array(gt_img.convert('1'))
                    f_gt.close()
                    mask_path = os.path.join(taskroot, 'eval_outputs', output, dataset, 'output_image', index + '.png')
                    report = dict_append(report, 'down5_mask', mask_path)
                    mask_img = Image.open(mask_path)
                    mask_img.save(os.path.join(taskroot, 'eval_outputs', 'image_output', 'down%d_mask.png' % i))
                    mask_img = np.array(mask_img.convert('1'))
                    report = dict_append(report, 'down5_pred', down_dict[key])
                    report = dict_append(report, 'down5_Acc', accuracy_score(gt_img, mask_img))
                final_report["result_data"][module_map_reverse[output]] = report

            elif (output.split('_')[0] == 'face' and output.split('_')[1] == 'functionality' and output.split('_')[2] in ["cam'angle", "expression", "lit'angle"]) or (output.split('_')[0] == 'face' and output.split('_')[1] == 'generalization' and output.split('_')[2] in ['gender', 'skin']):
                datasets = os.listdir(os.path.join(taskroot, 'eval_outputs', output))
                datasets = list(x for x in datasets if os.path.isdir(os.path.join(taskroot, 'eval_outputs', output, x)))

                report = {}
                dataset_dict = {}
                for dataset in datasets:
                    dataset_dict = dict_append(dataset_dict, dataset.split('_')[-1], dataset)
                
                for i, attribute in enumerate(dataset_dict.keys()):
                    Acc_list, AUC_list, EER_list, Precision_list, Recall_list, F1_score_list, conf_list, dataset_list, roc_list, pr_list = get_metrics(self.eval_box, taskroot, dataset_dict[attribute], output)
                    n = str(i)
                    report[n] = {}

                    report = dict_append(report, 'map', attribute)
                    report[n]['roc_curve'] = roc_list
                    report[n]['pr_curve'] = pr_list
                    report[n]['Acc'] = format(sum(Acc_list) / len(Acc_list) * 100, '.2f')
                    report[n]['AUC'] = format(sum(AUC_list) / len(AUC_list), '.4f')
                    report[n]['EER'] = format(sum(EER_list) / len(EER_list), '.4f')
                    report[n]['Precision'] = format(sum(Precision_list) / len(Precision_list) * 100, '.2f')
                    report[n]['Recall'] = format(sum(Recall_list) / len(Recall_list) * 100, '.2f')
                    report[n]['F1_score'] = format(sum(F1_score_list) / len(F1_score_list), '.4f')
                    report[n]['conf_diff'] = format(sum(conf_list) / len(conf_list), '.4f')
                final_report["result_data"][module_map_reverse[output]] = report

            elif output.split('_')[0] == 'video' and output.split('_')[1] == 'functionality' and output.split('_')[2] == "temporal'localization":
                datasets = os.listdir(os.path.join(taskroot, 'eval_outputs', output))
                datasets = list(x for x in datasets if os.path.isdir(os.path.join(taskroot, 'eval_outputs', output, x)))

                report = {}
                AP_list = []
                AR_list = []
                for dataset in datasets:
                    protocol_path = os.path.join(self.eval_box, output, 'groundtruth', '%s.txt' % dataset)
                    result_path = os.path.join(taskroot, 'eval_outputs', output, dataset, 'eval_result.txt')
                    gt, predict = read_result_vft(result_path, protocol_path) # gt:list, predict:dict

                    iou_thresholds = [0.5, 0.75, 0.95]
                    ap_score = AP(iou_thresholds=iou_thresholds)(gt, predict)
                    AP_list.append(ap_score)
                    
                    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
                    n_proposals_list = [100, 50, 20, 10]
                    ar_score = AR(n_proposals_list, iou_thresholds=iou_thresholds)(gt, predict)
                    AR_list.append(ar_score)
                report['AP'] = AP_list
                report['AR'] = AR_list
                final_report["result_data"][module_map_reverse[output]] = report

        with open(os.path.join(taskroot, 'eval_outputs', "final_report.json"), "w") as r:
            r.write(json.dumps(final_report, ensure_ascii=False))

    def image_build(self, task_id, image_name, dockerfile_path):
        image, _ = self.client.images.build(
            path=os.path.dirname(dockerfile_path),
            dockerfile=os.path.basename(dockerfile_path),
            tag=image_name,
            forcerm=True
        )

        return image

    def get_userpath(self, user_id):
        return os.path.join(self.users, str(user_id))

    def get_taskpath(self, user_id, task_id):
        return os.path.join(self.users, str(user_id), str(task_id))

    def get_imagename(self, user_id, task_id, stage):  # stage = eval or debug
        return f'{stage}/{user_id}_{task_id}:latest'

    def create_user(self, user_id):
        user_dir = self.get_userpath(user_id)
        os.makedirs(user_dir)
        return user_dir

    def delete_user(self, user_id):
        user_dir = self.get_userpath(user_id)
        shutil.rmtree(user_dir)
        return 0

    def create_task(self, user_id, task_id):
        task_dir = self.get_taskpath(user_id, task_id)
        os.makedirs(task_dir)
        return task_dir

    def delete_task(self, user_id, task_id):
        task_dir = self.get_taskpath(user_id, task_id)
        shutil.rmtree(task_dir)
        return 0

    def create_output_dir(self, user_id, task_id, modules):
        local_m = modules
        data_list = []

        modules0 = [x for x in local_m if x.split('_')[0] == '0' and x.split('_')[1] == '0']
        
        types = []
        for module in modules0:
            types.append(type_map[module.split('_')[3]])

        modules_r = [x for x in local_m if x.split('_')[0] == '0' and x.split('_')[1] == '3']
        for module in modules_r:
            for type in types:
                os.makedirs(os.path.join(self.get_taskpath(user_id, task_id), "eval_outputs", self.module_paths[module] + '_' + type))
                data_list.append(f'{os.path.join(self.eval_box, self.module_paths[module], type)}:/taskroot/data/{self.module_paths[module]}_{type}')
            local_m.remove(module)
        for module in local_m:
            os.makedirs(os.path.join(self.get_taskpath(user_id, task_id), "eval_outputs", self.module_paths[module]))
        data_list += [f'{os.path.join(self.eval_box, self.module_paths[module])}:/taskroot/data/{self.module_paths[module]}' for module in local_m]

        return data_list



if __name__ == '__main__':
    # flask run -p8888
    from concurrent.futures import ThreadPoolExecutor, as_completed
    # import time

    tasks = [
        # ['4', '2', ["2_1_0_0"], ['video']],
        ['2', '5', ["0_0_0_0", "0_0_0_1", "0_0_0_2", "0_0_0_3", "0_3_0_0", "0_3_1_0", "0_3_2_0", "0_3_3_0", "0_3_4_0", "0_3_5_0", "0_3_6_0", "0_3_7_0", "0_3_8_0", "0_2_0_0", "0_2_0_1", "0_2_0_2", "0_2_0_3", "0_2_0_4", "0_1_1_0", "0_1_2_0", "0_1_3_0", "0_2_1_0", "0_2_2_0", "0_4_0_0", "0_4_1_0",], ['face']],  # "0_0", "0_2"表示用户选择的评估模块名，根据不同的选择模块挂载不同的数据集
        # ['1', '35', ["1_0_0_0", "1_0_0_1", "1_0_0_2", "1_0_0_3"], ['frame']],  # "0_0", "0_2"表示用户选择的评估模块名，根据不同的选择模块挂载不同的数据集
        # ["0_0_0_0", "0_0_0_1", "0_0_0_2", "0_0_0_3",]
        # ["0_3_0_0", "0_3_1_0", "0_3_2_0", "0_3_3_0", "0_3_4_0", "0_3_5_0", "0_3_6_0", "0_3_7_0", "0_3_8_0",]
        # ["0_2_0_0", "0_2_0_1", "0_2_0_2", "0_2_0_3", "0_2_0_4",]
        # ["0_1_1_0", "0_1_2_0", "0_1_3_0", "0_2_1_0", "0_2_2_0",]
        # ["0_4_0_0", "0_4_1_0",]
        # ["0_1_0_0",]
        # ['test','task2', ["0_0", "0_1", "2_0"]],
        # ['test','task3']
    ]
    backend = BackEndVideo()
    res = []
    eval = []

    for i, (user_id, task_id, modules, types) in enumerate(tasks):
        print(backend.run_debug(user_id, task_id, types, cuda_id=1, rm_img=True))
        set_trace()
        print(backend.run_eval(user_id, task_id, modules, cuda_id=0, rm_img=True))
        set_trace()
        backend.eval_result_process(user_id, task_id)
        set_trace()
