import re 
import json 
import typer

def is_alternate_roles(messages: list[dict]) -> bool:
    for i, message in enumerate(messages):
        if i % 2 == 0 and message["role"] != "user":
            return False
        if i % 2 == 1 and message["role"] != "assistant":
            return False
    return True 


def extract_data_from_log(log_line: str) -> str:
    # First check if the log line contains "Received request chatcmpl"
    if "Received request chatcmpl" not in log_line:
        # print("no chatcmpl in log line: ", log_line)
        return None
    # extract the text inside: prompt:'<begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n and <|start_header_id|>assistant<|end_header_id|>\n\n', params: SamplingParams
    start_index = log_line.find("prompt: '")
    end_index = log_line.find("params: SamplingParams")
    if start_index == -1 or end_index == -1:
        print(f"no start_index or end_index in log line: {log_line}")
        return None
    if start_index > end_index:
        return None
    
    prompt = log_line[start_index + len("prompt: '"): end_index - 3].strip()
    prompt = prompt.replace("\\n", "\n")
    prompt = prompt.replace("\\t", "\t")
    prompt = prompt.replace("\\r", "\r")
    prompt = prompt.replace("\\\"", "\"")
    prompt = prompt.replace("\\'", "'")
    prompt = prompt.replace("\\", "")
    # now recover the list of messages and tools
    #print("--------------------")
    #print(prompt)
    result = []
    for match in re.finditer(r"<\|start_header_id\|>(?P<role>user|assistant)<\|end_header_id\|>(?P<content>.+?)<\|eot_id\|>", prompt, re.DOTALL):
        role = match.group("role")
        content = match.group("content").strip()
        result.append({"role": role, "content": content})
    
    # check roles are user, assistant alternately
    if not is_alternate_roles(result):
        print("wrong role sequences: ", result)
        return None
    return result


def main(input_file: str, output_file: str):
    result = []
    with open(input_file, "r") as f:
        for line in f:
            item = extract_data_from_log(line)
            if item:
                result.append(item)
                if len(item) > 1:
                    print("more than one message in one log line: ", item)
    
    print(f"total {len(result)} items")
    with open(output_file, "w") as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    typer.run(main)
    #main("test_lines.txt", "test_lines_output.json")