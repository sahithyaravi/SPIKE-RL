from funqa import load_funqa_json_qa, get_funqa_video_path
from oops import load_oops_json, get_oops_video_path
from bean import load_bean_json, get_bean_video_path
from activitynet import load_activitynet_json, get_activitynet_video_path
from exfuntube import load_exfuntube_json, get_exfuntube_video_path



def process_json_and_create_data(json_path, video_root):
    if "funqa" in json_path.lower():
        print("Loading FunQA dataset...")
        dataset = load_funqa_json_qa(json_path)
        get_video_path = get_funqa_video_path
    elif "bean" in json_path.lower():
        print("Loading BEAN dataset...")
        dataset = load_bean_json(json_path)
        get_video_path = get_bean_video_path
    elif "activitynet" in json_path.lower():
        print("Loading ActivityNet dataset...")
        dataset = load_activitynet_json(json_path)
        get_video_path = get_activitynet_video_path
    elif "exfuntube" in json_path.lower():
        print("Loading ExFunTube dataset...")
        dataset = load_exfuntube_json(json_path)
        get_video_path = get_exfuntube_video_path
    else:
        print("Loading Oops dataset...")
        dataset = load_oops_json(json_path)
        get_video_path = get_oops_video_path

    final_dataset = []
    for idx, entry in enumerate(dataset):
        set_id = entry.get("set_id", "unknown_set")
        index = entry.get("index") if "index" in entry else None
        visual_input = entry.get("visual_input", None)
        video_file = get_video_path(video_root, set_id, index, visual_input)
        entry["video_path"] = video_file
        entry["caption"] = entry.get("caption", "") or entry.get("explanation", "")
        final_dataset.append(entry)

    return final_dataset
