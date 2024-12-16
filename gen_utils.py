import json , os , datetime
from typing import List
from concurrent.futures import ThreadPoolExecutor
import traceback


def split_item_into_batches_in_order(items, batch_size):
    result = []
    cur_batch = []
    for item in items:
        if len(cur_batch) < batch_size:
            cur_batch.append(item)
        else:
            result.append(cur_batch)
            cur_batch = [item]
    if cur_batch:
        result.append(cur_batch)
    assert sum([len(batch) for batch in result]) == len(items)
    return result


def handle_batch_items(input_items, handle_item_function):
    final_result = []
    indices = [index for index in range(len(input_items))]

    def monkey_patch_handle(new_input):
        index, input_item = new_input
        try:
            result = handle_item_function(input_item)
        except Exception as e:
            traceback.print_exc()
            print(f"Exception handling input: {str(e)}, input_item: {input_item}")
            return index, (None, 0)
        return index, result

    new_input_items = [(index, item) for index, item in zip(indices, input_items)]
    final_result = [None for _ in range(len(input_items))]
    total_cost = 0
    with ThreadPoolExecutor() as executor:
        results = executor.map(monkey_patch_handle, new_input_items)
        for result in results:
            index, item_result = result
            total_cost += item_result[1]
            final_result[index] = item_result[0]
    return final_result, total_cost

def save_data(data, save_path):
    with open(save_path, "w") as f:
        if "jsonl" in save_path:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        elif "json" in save_path:
            f.write(json.dumps(data, ensure_ascii=False, indent=4))
        else:
            raise ValueError(f"Unsupported file extension: {save_path}")


def read_data(data_path: str):
    with open(data_path, "r") as f:
        if "jsonl" in data_path:
            return [json.loads(line) for line in f]
        elif "json" in data_path:
            return json.loads(f.read())
        else:
            raise ValueError(f"Unsupported file extension: {data_path}")


class LLMTask(object):
    def __init__(
        self,
        input_path: str = "",
        save_path: str = "",
        inputs: List = [],
        size: int = -1,
        from_scratch: bool = False,
    ) -> None:
        self.inputs = []
        if input_path:
            with open(input_path, "r") as f:
                if input_path.endswith(".json"):
                    self.inputs = json.loads(f.read())
                else:
                    self.inputs = [json.loads(line) for line in f]

        if inputs:
            self.inputs = inputs

        self.size = size
        self.result = []
        self.save_path = save_path
        if save_path:
            if os.path.exists(save_path):
                self.result = read_data(save_path)
        if from_scratch:
            self.result = []

    def run(self, handle_item_func, metrics_func=None, thread_count: int = 5):
        p_items = self.inputs[len(self.result) :]

        if self.size > -1:
            if len(self.result) > self.size:
                print(f"stop because already had: {len(self.result)} > {self.size}")
                p_items = []
            else:
                remaining_size = self.size - len(self.result)
                p_items = p_items[:remaining_size]

        print(
            f"total number of inputs: {len(self.inputs)}, handled: {len(self.result)}, size={self.size}, remaining: {len(p_items)}"
        )

        total_remaining_size = len(p_items)
        total_cost = 0
        failed_count = 0
        item_handled_count = 0
        t1 = datetime.datetime.now()

        buckets = split_item_into_batches_in_order(p_items, thread_count)

        backup_path = self.save_path + "_backup"
        for index, bucket in enumerate(buckets):
            item_results, cost = handle_batch_items(bucket, handle_item_func)
            assert len(bucket) == len(item_results)

            for item_result in item_results:
                if item_result is None:
                    failed_count += 1

            self.result.extend(item_results)
            item_handled_count += len(bucket)

            total_cost += cost
            t2 = datetime.datetime.now()
            acc_time = (t2 - t1).total_seconds()
            avg_time = acc_time / item_handled_count
            avg_cost = total_cost / item_handled_count
            remaining_size = total_remaining_size - item_handled_count

            if metrics_func:
                metrics_func(self.result)
            save_data(self.result, self.save_path)
            if index % 5 == 0 and index > 1:
                save_data(self.result, backup_path)
            print(
                f"{item_handled_count}/{len(p_items)}, fail_count: {failed_count}, avg_time: {avg_time}, remaining time: {avg_time * remaining_size / 60} mins, avg_cost: {avg_cost}, estimated_cost: {avg_cost * total_remaining_size}"
            )
