import asyncio
from bitagent.tasks.task import get_random_task
from bitagent.datasources.tools import ToolDataset
from bitagent.tasks.tool_call_task import ToolCallTask
from langchain_openai import ChatOpenAI
from sentence_transformers import util
from bitagent.helpers.sbert import CachedSentenceTransformer
from bitagent.helpers.llms import llm
import asyncio
import traceback
import re
from typing import Dict
import typer
import json
from colorama import Fore, Style, init

init()


class ValidatorConfig:
    def __init__(self):
        self.validator_hf_server_port = 8000
        self.openai_api_key = "abc"


class SimulateValidator:
    def __init__(
        self,
        llm_name: str = "thesven/Mistral-7B-Instruct-v0.3-GPTQ",
        base_url: str = "http://localhost:30000/v1",
        validator_hf_server_port: int = 8000,
        shuffle_data: bool = True,
    ):
        self.tool_dataset = ToolDataset(shuffle=shuffle_data)
        # bt.logging.debug("Initializing Validator - this may take a while (downloading data and models) - loading model ...")
        self.sentence_transformer = CachedSentenceTransformer("BAAI/bge-small-en-v1.5")

        def llm(messages, max_new_tokens=160, temperature=0.7):
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            llm = ChatOpenAI(
                openai_api_key="abc",
                model_name=llm_name,
                base_url=base_url,
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
            return llm.invoke(messages).content.strip()

        self.llm = llm

        # bt.logging.debug("Initializing Validator - this may take a while (downloading data and models) - finished loading model")

        # code to measure the relevance of the response to the question
        def measure_relevance_of_texts(text1, text2):
            # Encode the texts to get the embeddings
            if type(text2) == list:
                embeddings = self.sentence_transformer.encode(
                    [text1, *text2], convert_to_tensor=True, show_progress_bar=False
                )
            else:
                embeddings = self.sentence_transformer.encode(
                    [text1, text2], convert_to_tensor=True, show_progress_bar=False
                )
            # Compute the cosine similarity between the embeddings
            if type(text2) == list:
                return util.pytorch_cos_sim(embeddings[0], embeddings[1:])[0]
            else:
                return float(util.pytorch_cos_sim(embeddings[0], embeddings[1:])[0][0])

        self.measure_relevance_of_texts = measure_relevance_of_texts
        self.config = ValidatorConfig()
        self.config.validator_hf_server_port = validator_hf_server_port


async def run_evaluation(
    save_path: str,
    eval_model_name: str,
    gen_model_name: str = "thesven/Mistral-7B-Instruct-v0.3-GPTQ",
    gen_base_url: str = "http://localhost:30000/v1",
    eval_model_port: int = 8000,
    num_tasks: int = 4,
    batch_size: int = 2,
    temperature: float = 0.7,
):

    validator = SimulateValidator(
        llm_name=gen_model_name,
        base_url=gen_base_url,
        validator_hf_server_port=eval_model_port,
    )
    sem = asyncio.Semaphore(5)

    async def call_llm_with_semaphore(task):
        async with sem:
            return await asyncio.to_thread(
                llm,
                validator,
                task.synapse.messages,
                task.synapse.tools,
                eval_model_name,
                hugging_face=True,
                temperature=temperature,
            )

    async def evaluate_task(validator, task, response):
        try:
            return [task.reward(validator, response)]
        except Exception as e:
            print(f"An exception calling task.reward: {e}")

    async def return_results(validator, task, miner_uid, reward, response):
        # means we got all of the information we need to score the miner and update wandb
        if len(reward) == 4:
            score, max_possible_score, task_results, correct_answer = reward
            # make sure the score is not None
            if score and max_possible_score:
                normalized_score = score / max_possible_score
                return task_results
            return None
        elif len(reward) == 2:  # skip it
            # bt.logging.debug(f"Skipping results for this task b/c Task API seems to have rebooted: {reward[1]}")
            # time.sleep(25)
            return None
        else:
            # bt.logging.debug(f"Skipping results for this task b/c not enough information")
            # time.sleep(25)
            return None

    tasks = []

    for i, _ in enumerate(range(0, num_tasks, batch_size)):
        for i in range(batch_size):
            tasks.append(get_random_task(validator, offline=True))

    log_results = [dump_task_to_json(task) for task in tasks]

    print(f"Generated {len(tasks)} tasks")

    llm_responses_and_finishes = await asyncio.gather(
        *[call_llm_with_semaphore(task) for task in tasks]
    )
    try:
        llm_responses = [r[0] for r in llm_responses_and_finishes]
        llm_finishes = [r[1] for r in llm_responses_and_finishes]
    except Exception as e:
        traceback.print_exc()

    responses = []
    for j, llm_response in enumerate(llm_responses):
        task = tasks[j]
        response = task.synapse.model_copy()
        response.response = llm_response.strip()
        response.dendrite.process_time = (
            5.0  # TODO may be useful to test performance of the model itself
        )
        response.dendrite.status_code = 200
        response.axon.status_code = 200
        response.competition_version = "1-2"
        responses.append(response)
        log_results[j]["response"] = llm_response.strip()

    rewards = await asyncio.gather(
        *[
            evaluate_task(validator, tasks[i], responses[i])
            for i in range(len(responses))
        ]
    )

    try:
        scores = []
        miner_tasks = []  # Collect tasks to execute in parallel for each miner
        for i, reward in enumerate(rewards):
            log_results[i]["reward"] = reward
            if (
                len(reward[0]) == 4
                and reward[0][0] is not None
                and reward[0][1] is not None
            ):
                scores.append(reward[0][0] / reward[0][1])

                async def process_miner_task(task_idx, miner_uid, reward, response):
                    # Get the result for this miner
                    result = await return_results(
                        validator, tasks[task_idx], miner_uid, reward[0], response
                    )
                    # print("--------------------")
                    # for item in result:
                    #     print(item)
                    return result

                miner_tasks.append(process_miner_task(i, 134, reward, responses[i]))
            else:
                # Bad reward, so 0 score
                print("bad reward ... ")
                scores.append(0.0)
            log_results[i]["final_score"] = scores[-1]
        # Await all miner-specific tasks concurrently
        await asyncio.gather(*miner_tasks)

    except Exception as e:
        traceback.print_exc


    print("scores: ", scores)
    avg_score = sum(scores) / len(scores)
    print("avg_score: ", avg_score)
    
    dump_data = {
        "eval_model_name": eval_model_name,
        "gen_model_name": gen_model_name,
        "temperature": temperature,
        "avg_score": sum(scores) / len(scores),
        "scores": scores,
        "log_results": log_results,
    }
    with open(save_path, "w") as f:
        json.dump(dump_data, f, ensure_ascii=False, indent=4)


def dump_task_to_json(task: ToolCallTask) -> Dict:
    tools = task.synapse.tools
    n_tools = [tool.to_dict() for tool in tools]
    messages = task.synapse.messages
    criteria = task.criteria

    ground_truth = None
    for criterion in criteria:
        if criterion.name == "Return correct function name":
            ground_truth = criterion.eval_args
            #print("ground_truth: ", ground_truth)
            break

    record = {
        "tools": n_tools,
        "messages": [message.to_dict() for message in messages],
        "original_user": task.original_user,
        "ground_truth": ground_truth,
        "source": task.source,
    }
    return record


def main(
    save_path: str,
    eval_model_name: str,
    gen_model_name: str = "thesven/Mistral-7B-Instruct-v0.3-GPTQ",
    gen_base_url: str = "http://localhost:30000/v1",
    eval_model_port: int = 8000,
    num_tasks: int = 4,
    batch_size: int = 2,
    temperature: float = 0.7,
):
    asyncio.run(
        run_evaluation(
            save_path,
            eval_model_name,
            gen_model_name=gen_model_name,
            gen_base_url=gen_base_url,
            eval_model_port=eval_model_port,
            num_tasks=num_tasks,
            batch_size=batch_size,
            temperature=temperature,
        )
    )


if __name__ == "__main__":
    typer.run(main)

# python simulate_validator.py internal_evaluation/num_task_4_bs_2.json jrp66K19Ex/DSB1lJvQL3 --num_tasks 1000 --batch_size 100
