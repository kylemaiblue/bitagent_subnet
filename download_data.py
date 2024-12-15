from bitagent.tasks.tool_call_task import ToolCallTask
from simulate_validator import SimulateValidator
from simulate_validator import dump_task_to_json
import json 
import typer
import datetime 
import traceback
from gen_utils import LLMTask

def main(dataset: str = "bfcl", size: int = -1, thread_count: int = 5, from_scratch: bool = False, data_mode: str = "tool_call", port: int = 30000):
    assert dataset in ["bfcl", "glaive", "bitagent"]
    save_path = f"extracted_data/all_data/{dataset}.jsonl"
    validator = SimulateValidator(shuffle_data=False, base_url=f"http://localhost:{port}/v1")
    ds_size_dic = validator.tool_dataset.get_ds_size()
    print("ds_size_dic: ", ds_size_dic)
    bfcl_size = ds_size_dic[dataset]
    task = LLMTask(inputs = [i for i in range(bfcl_size)], save_path=save_path, size=size, from_scratch=from_scratch)
    def handle_item(index: int):
        result = {}
        try:
            task = ToolCallTask(validator=validator, name="Responds with correct function call", dname=dataset, ds_index=index, data_mode=data_mode)
            dumped_data = dump_task_to_json(task)
            result = dumped_data
        except Exception as e:
            traceback.print_exc()
            print(f"Error: {e}")
            result.update({
                "error": str(e),
                "dataset": dataset,
                "index": index,
                "traceback": traceback.format_exc()
            })
        return result, 0
    
    def metrics_func(data):
        count = 0
        for item in data:
            if "error" in item:
                count += 1
        print(f"error_count: {count} / len(data): {len(data)}")
    task.run(handle_item_func=handle_item, metrics_func=metrics_func, thread_count=thread_count)



def investigate_data(dataset: str = "glaive", index: int = 0):
    validator = SimulateValidator(shuffle_data=False)
    dataset = "glaive"
    size = validator.tool_dataset.get_ds_size_of_dname(dataset)
    for i in range(size):
        data = validator.tool_dataset.__next_ds__(dataset, i)
        if data is None:
            print(f"None data at index {i}")
            break
        
if __name__ == "__main__":
    #investigate_data()
    typer.run(main)
