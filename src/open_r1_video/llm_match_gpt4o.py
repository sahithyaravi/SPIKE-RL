from openai import OpenAI
import re
import json
import argparse
from tqdm import tqdm
import pandas as pd

client = OpenAI()


def _extract_score(text: str) -> float:
    """Extract numerical score from generated text"""
    match = re.search(r'(\d+\.?\d*)', text.strip())
    if match:
        score = float(match.group(1))
        return score if 0.0 <= score <= 1.0 else 0.5
    return 0.5
    

def llm_match(gold, predicted):
    prompt = f"""
Rate how closely the content of the prediction matches the content of the reference description in terms of meaning and how well it captures important details regarding events in the video.
Ignore the difference in length.
Score 0.0-1.0 where:

0.0-0.3: Poor match (key details in the reference are missing in the prediction)
0.4-0.6: Moderate match (a few key details in the reference are captured in the prediction)
0.7-0.9: Good match (most key details are present in the prediction)
1.0: Perfect match (all key details in the reference are accurately captured in the prediction)
Output only the numerical score (e.g., 0.75).

Reference: {gold}
Predicted: {predicted}

Score:
    """

    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "user",
                "content": [
                    { "type": "input_text", "text": prompt },
                
                ],
            }
        ],
        max_output_tokens=50,
        # temperature=0.5,
    )

    return _extract_score(response.output_text)


def load_jsonl(jsonl_path: str):
    data_dict = []
    with open(jsonl_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            data_dict.append(entry)
    return data_dict



def main():
    # args get a json file
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True, help="Path to json file with gold and predicted")
    parser.add_argument("--gold_key", type=str, default="reference", help="Key for gold in json")
    parser.add_argument("--pred_key", type=str, default="prediction", help="Key for predicted in json")
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples to process, -1 for all")
    args = parser.parse_args()
    # output path same folder as input file, named llm_scores.json
    output_path = args.json_path.replace(".json", "_gpt_scores.json")
    # average metrics path
    avg_path = args.json_path.replace(".json", "_gpt_avg.txt")

    # read jsonl
    data = load_jsonl(args.json_path)

    if args.num_samples > 0:
        data = data[: args.num_samples]
    scores = []
    for item in tqdm(data):
        gold = item[args.gold_key]
        pred = item[args.pred_key]
        score = llm_match(gold, pred)
        scores.append(score)
        item["gpt_score"] = score

        # write to json every 10 items
        if len(scores) % 10 == 0:
            with open(output_path, "w") as f:
                json.dump(data, f, indent=4)
        
    
    # save scores to json
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    # save average score to txt
    avg_score = sum(scores) / len(scores)
    with open(avg_path, "w") as f:
        f.write(f"Average GPT Score: {avg_score}\n")

    # gold = "The boy pushes the baby in the chair and an adult comes to intervene and slips on the floor"
    # predicted = "n the video, a young boy is playfully interacting with a high chair in a living room. he reaches up to grab something on the chair while his father, lying on the floor, attempts to assist him. the father accidentally ends up being pushed over as the boy continues his playful actions, creating a humorous and lighthearted scene. the room features wooden flooring and furniture,"
    # print(llm_match(gold, predicted))

if __name__ == "__main__":
    main()