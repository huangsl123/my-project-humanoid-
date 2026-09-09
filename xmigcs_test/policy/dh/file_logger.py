# Copyright (c) 2025 Xuxin @ 747302550. 保留所有权利. 未经许可，禁止复制、修改或分发
import datetime
import csv
import os
import os.path as osp

class FileLogger:
    def __init__(self, dt, save_dir, variable_name="observation"):
        self.dt = dt
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_name = "{}_{}.csv".format(current_time, variable_name)
        csv_path = os.path.join(save_dir, csv_name)
        self.file = open(csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.file)
        self.count = 0

    def data_log(self, observation_to_log):
        log_time = self.count * self.dt
        data1 = list(observation_to_log)
        data1.insert(0, log_time)
        self.csv_writer.writerow(data1)
        self.count += 1

    def data_log_no_time(self, observation_to_log):
        "无时间戳,所有数据写在一行上"
        data1 = list(observation_to_log)
        data_str = ",".join(map(str, data1))  # 将数据转换为字符串并用逗号分隔
        self.file.write(data_str + ",")  # 写入文件（追加模式，不换行）
        self.file.flush()
