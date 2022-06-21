import json
import gzip
import re
import csv

# hugging fact dataset: https://huggingface.co/datasets/sentence-transformers/reddit-title-body/tree/main
filename = 'reddit_title_text_2011.jsonl.gz'

json_content = []
can_be_used = total_posts = 0
with gzip.open(filename , 'rb') as gzip_file:
    for line in gzip_file:
        line = line.rstrip()
        if line:
            obj = json.loads(line)
            json_content.append(obj)
            # re.findall includes punctuation
            title_split = re.findall(r"[\w']+|[.,!?:;]", obj["title"])
            body_split = re.findall(r"[\w']+|[.,!?:;]", obj["body"])
            # titles between 3 to 15 words and body between 15 and 60 words (including punctuation)
            if 3 <= len(title_split) <= 15 and 15 <= len(body_split) <= 60:
                can_be_used += 1
            total_posts += 1

with open("hugging_face_cumulative_stats.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Can be used", can_be_used])
    writer.writerow(["Total posts", total_posts])
