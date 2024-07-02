import os

from typing import Optional
from transformers.trainer import Trainer

from logger_config import logger
from evaluation.metrics import accuracy, batch_mrr
from models import BiencoderOutput, BiencoderModel
from util import AverageMeter
import subprocess
from multiprocessing import Process
import json
import torch

class BiencoderTrainer(Trainer):
    def __init__(self, *pargs, **kwargs):
        super(BiencoderTrainer, self).__init__(*pargs, **kwargs)
        self.model: BiencoderModel

        self.acc1_meter = AverageMeter('Acc@1', round_digits=2)
        self.acc3_meter = AverageMeter('Acc@3', round_digits=2)
        self.mrr_meter = AverageMeter('mrr', round_digits=2)
        self.last_epoch = 0
        self.loss_dict = []

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to {}".format(output_dir))
        self.model.save(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        # loss_dict_name = '_'.join(output_dir.split('/')[-2:])
        # with open(f'loss_dict/{loss_dict_name}.json', 'w') as f: 
        #     json.dump(self.loss_dict, f, indent=4)
        # build index asynchronously
        # index_building_script_path = '/home/bingxing2/home/scx6964/workspace_wenhan/project/demorank/build_index/build_dense_index.sh'
        # print('###################### building the index ######################')
        # run_shell_script(index_building_script_path)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs: BiencoderOutput = model(inputs)
        loss = outputs.loss
        loss_dict = outputs.loss_dict
        loss_dict['global_step'] = self.state.global_step
        self.loss_dict.append(loss_dict)

        if self.model.training:
            step_acc1, step_acc3 = accuracy(output=outputs.scores.detach(), target=outputs.labels, topk=(1, 3))
            step_mrr = batch_mrr(output=outputs.scores.detach(), target=outputs.labels)

            self.acc1_meter.update(step_acc1)
            self.acc3_meter.update(step_acc3)
            self.mrr_meter.update(step_mrr)

            if self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
                logger.info('step: {}, {}, {}, {}'.format(self.state.global_step, self.mrr_meter, self.acc1_meter, self.acc3_meter))

            self._reset_meters_if_needed()

        return (loss, outputs) if return_outputs else loss

    def _reset_meters_if_needed(self):
        if int(self.state.epoch) != self.last_epoch:
            self.last_epoch = int(self.state.epoch)
            self.acc1_meter.reset()
            self.acc3_meter.reset()
            self.mrr_meter.reset()

def run_shell_script(script_path):
    # 获取脚本所在目录和脚本名
    script_dir, script_name = os.path.split(script_path)
    # 保存当前目录
    original_directory = os.getcwd()
    # 切换到脚本所在目录
    os.chdir(script_dir)
    # 运行脚本
    subprocess.run(["bash", script_name])
    # 切换回原始目录
    os.chdir(original_directory)

