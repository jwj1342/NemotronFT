SYSTEM_PROMPT = """
Please write out the derivation process in detail step by step, ensuring that each step is logically sound. 
Make sure to include all necessary intermediate steps and justifications for each step in the derivation.
After reaching the conclusion, please fill in the final result in the \boxed{}.
likes:
"\\boxed{final result here}"
"""

DISTILLATION_COT_SYSTEM_PROMPT = """
You are a reasoning annotator for distillation data construction.
Given a question and the gold answer, produce a detailed, logically complete, step-by-step chain-of-thought that justifies the gold answer.
Your reasoning should be faithful, explicit, and suitable for training a student model to imitate the reasoning process.
Do not discuss alternative answers.
Return JSON only.
"""

DISTILLATION_COT_USER_PROMPT = """
Question:
[question]

Gold answer:
[gold_answer]

Requirements:
1. The reasoning must directly justify the given gold answer.
2. The reasoning must be detailed, logically complete, and step-by-step.
3. Do not skip important intermediate steps, derivations, or logical transitions.
4. Make implicit reasoning explicit whenever it is necessary for correctness or clarity.
5. Use only relevant reasoning; avoid filler, repetition, unsupported speculation, and irrelevant commentary.
6. The reasoning must remain faithful to the question and must end at the provided gold answer.
7. When calculations or formal deductions are involved, show the intermediate process clearly.
8. Prefer clear, self-contained sentences so that the reasoning can serve as distillation supervision.
9. The final answer must be formatted as \\\\boxed{...}.
10. The boxed content must exactly equal the provided gold answer.
11. Do not include any explanation outside the JSON object.

Output format:
{
  "cot": "detailed step-by-step reasoning process",
  "answer": "\\\\boxed{...}"
}
"""