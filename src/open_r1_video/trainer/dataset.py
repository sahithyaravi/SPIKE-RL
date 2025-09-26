def get_data(item):
    # dataset = wangxd
    video_path = item.get("video") or item.get("video_path")

    if "wangxd" in video_path:
        ground_truth = item.get("ground_truth")

    elif item.get("source") == "ActivityNet_Captions":
        ground_truth = item.get("caption")
        video_path = "data/ActivityNet_Captions/Activity_Videos/" + video_path
    elif item.get("source") == "oops":
        ground_truth = item.get("caption")
        video_path = "data/oops_dataset/oops_video/train/" + video_path

    return video_path, ground_truth
