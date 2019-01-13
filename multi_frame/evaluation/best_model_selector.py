import os
import math
from collections import defaultdict

def confstr2conf(conf_str):
    conf = {}
    conf_str_s = conf_str.split("_")
    conf["network"] = conf_str_s[0]
    conf["input_type"] = conf_str_s[1]
    conf["rnn_hidden"] = int(conf_str_s[2])
    conf["batch_size"] = int(conf_str_s[3])
    conf["window_size"] = int(conf_str_s[4])
    return conf

def select_best_model_conf(log_dir, log_files, num=1, verbose=False):
    conf2avescore = []

    for log_file in log_files:
        conf2scores = defaultdict(list)
        file_path = os.path.join(log_dir, log_file)
        lines = open(file_path, 'r').readlines()
        val_reports = [line for line in lines if line.split(',')[0] == "VALIDATION_REPORT"]
        for val_report in val_reports:
            val_report_s = [e.strip() for e in val_report.split(",")]
            conf_and_val_id = val_report_s[1]
            conf = "_".join(conf_and_val_id.split("_")[:5])
            score = float(val_report_s[3])
            if math.isnan(score):
                continue
            conf2scores[conf].append(score)

        for conf, scores in conf2scores.items():
            if len(scores) < 4:
                continue
            conf2avescore.append((conf, sum(scores) / len(scores)))

    conf2avescores = list(reversed(sorted(conf2avescore, key=lambda x: x[1])))
    best_conf_str = conf2avescores[0][0]
    
    if verbose:
        for conf, score in conf2avescores:
            print("%s -> %f" % (conf, score))
        print("best_conf (%f): %s" % (conf2avescores[0][1], best_conf_str))
        
    return best_conf_str


if __name__ == "__main__":
    log_dir = "./training/output/test_dialog_id_05/log"
#     # lstm
#     log_files = ["20181219205625.txt"]
#     # gru
#     log_files = ["20181219202934.txt"]
#     # rnn
#     log_files = ["20181223190430.txt"]
    
    # atlstm
    log_dir = "./training/output/attention_test_dialog_id_05/log"
    log_files = ["20181228014132.txt"]
    best_conf = select_best_model_conf(log_dir, log_files, verbose=True)
    
    
    