from utils.humaneval_pass.data import read_problems 

file_path = "data/HumanEval/SingleLineInfilling.jsonl"
problems = read_problems(file_path)
print(problems["SingleLineInfilling/HumanEval/0/L0"])