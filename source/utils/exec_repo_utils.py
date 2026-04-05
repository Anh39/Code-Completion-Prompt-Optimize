import multiprocessing as mp
from fuzzywuzzy import fuzz
import os, shutil
import numpy as np
import math

class MPLogExceptions(object):
    def __init__(self, callable):
        self.__callable = callable

    def error(msg, *args): #type:ignore
        return mp.get_logger().error(msg, *args) 

    def __call__(self, *args, **kwargs):
        import traceback
        try:
            result = self.__callable(*args, **kwargs)

        except Exception as e:
            # Here we add some debugging help. If multiprocessing's
            # debugging is on, it will arrange to log the traceback
            self.error(traceback.format_exc())
            # Re-raise the original exception so the Pool worker can``
            # clean up
            raise

        # It was fine, give a normal answer
        return result
def copy_src_to_dest(src_dir, tgt_dir, cur):
    source_dir = os.path.join(src_dir, cur)
    target_dir = os.path.join(tgt_dir, cur)
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
def remove_comments(code, language = "python", remove_blank_line = True):
    import re
    if language == "python":
        code = re.sub(r'(""".*?"""|\'\'\'.*?\'\'\')', '', code, flags=re.DOTALL)
        #code = re.sub(r'#.*', '', code)
    elif language == "java":
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        code = re.sub(r'//.*', '', code)
    elif language == "cpp":
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        code = re.sub(r'//.*', '', code, flags=re.DOTALL)
        # 匹配除了新行符之外的任何单个字符，现在匹配包括行结束符在内的任何单个字符
        # 匹配单行注释 //...
        # (?<!http:|https:) 避免删除URL中的双斜线
        #code = re.sub(r'(?<!http:|https:)\/\/.*', '', code)
    if remove_blank_line:
        code_lines = code.split("\n")
        code_lines = [c for c in code_lines if c.strip() != ""]
        code = "\n".join(code_lines)
    return code
def cal_edit_sim(references, hypotheses):
    total = len(references)
    edit_sim = 0.0
    for pred, gt in zip(hypotheses, references):
        pred = pred.strip()
        gt = gt.strip()
        edit_sim += fuzz.ratio(pred, gt)
    return edit_sim / total
def get_avg_score(samples, key):
    return float(np.average([obj[key] for obj in samples]))
def multi_tasks_from_objs(objs, workers = 64, task=None, chunk_size=None, args=None):
    p = mp.Pool(workers)
    if chunk_size:
        results = []
        job_num = math.ceil(len(objs) / chunk_size)
        print(f"job num: {job_num}")
        for worker_id in range(job_num):
            results.append(p.apply_async(MPLogExceptions(task), args=(objs[worker_id * chunk_size: (worker_id + 1) * chunk_size], worker_id, workers, args)))
    else:
        chunk_size = math.ceil(len(objs) / float(workers))
        results = []
        for worker_id in range(workers):
            results.append(p.apply_async(MPLogExceptions(task), args=(objs[worker_id * chunk_size: (worker_id + 1) * chunk_size], worker_id, workers, args)))
    p.close()
    p.join()
    output_objs = []
    for result in results:
        output_objs.extend(result.get())
    return output_objs