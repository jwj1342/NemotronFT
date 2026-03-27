import pandas as pd
import re

df = pd.read_csv('data/train.csv')

def extract_core_question(prompt):
    # Remove "In Alice's Wonderland, " prefix
    if prompt.startswith("In Alice's Wonderland, "):
        prompt = prompt[len("In Alice's Wonderland, "):]

    # Detect category and extract key operation info
    if "bit manipulation rule transforms 8-bit binary numbers" in prompt:
        category = "Bit Manipulation"
        key_info = "bit shifts, rotations, XOR, AND, OR, NOT operations"
    elif "secret encryption rules are used on text" in prompt:
        category = "Text Cipher"
        key_info = "text encryption/decryption"
    elif "secret unit conversion is applied to measurements" in prompt:
        category = "Unit Conversion"
        key_info = "measurement conversion"
    elif "transformation rules is applied to equations" in prompt:
        # Check if numeric or symbolic by looking at examples
        if re.search(r'\d+[+\-*/\\|]\d+ =', prompt):
            category = "Equation Numeric"
            key_info = "numeric equation transformation"
        else:
            category = "Equation Symbolic"
            key_info = "symbolic equation transformation"
    elif "gravitational constant has been secretly changed" in prompt:
        category = "Gravity Constant"
        key_info = "d = 0.5*g*t^2"
    elif "numbers are secretly converted into a different numeral system" in prompt:
        category = "Numeral System"
        key_info = "numeral system conversion"
    else:
        category = "Unknown"
        key_info = ""

    # Extract examples and question parts
    examples_match = re.search(r'(Here are some examples?|Below are a few examples?|Some examples are given below|For example):?(.*?)(Now,)', prompt, re.DOTALL)
    if examples_match:
        examples = examples_match.group(2).strip()
        question = prompt[prompt.find("Now,"):].strip()

        if key_info:
            return f"Category: {category} ({key_info})\n\nExamples:{examples}\n\n{question}"
        else:
            return f"Category: {category}\n\nExamples:{examples}\n\n{question}"

    return f"Category: {category}\n\n{prompt}"

df['prompt_cleaned'] = df['prompt'].apply(extract_core_question)

# Show cleaning results
print("Original prompt sample:")
print(df['prompt'].iloc[0][:200])
print("\nCleaned prompt sample:")
print(df['prompt_cleaned'].iloc[0][:300])

# Save cleaned data
df_cleaned = df[['id', 'prompt_cleaned', 'answer']].copy()
df_cleaned.columns = ['id', 'prompt', 'answer']
df_cleaned.to_csv('data/train_cleaned.csv', index=False)
print(f"\nSaved {len(df_cleaned)} cleaned samples to train_cleaned.csv")
