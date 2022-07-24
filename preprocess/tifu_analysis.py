import json
import csv

with open("tifu_all_tokenized_and_filtered.json") as f:
    lines = f.readlines()
    with open("tifu_row_stats.csv", "w", newline="") as f2:
        writer = csv.writer(f2)
        i = 1
        # can_be_used requirements: title between 3 and 15 words, paragraph between 10 and 60 words
        has_tldr = can_be_used = 0
        title_length_sum = paragraph_length_sum = 0
        longest_paragraph = 0
        shortest_paragraph = 1000
        longest_title = 0
        shortest_title = 1000
        
        for line in lines:
            line = json.loads(line)
            if line["tldr"] is not None:
                title_len, para_len = len(line["trimmed_title_tokenized"]), len(line["tldr"].split())
                writer.writerow([title_len, para_len])
                has_tldr += 1

                title_length_sum += title_len
                paragraph_length_sum += para_len

                shortest_paragraph = min(shortest_paragraph, para_len)
                longest_paragraph = max(longest_paragraph, para_len)
                shortest_title = min(shortest_title, title_len)
                longest_title = max(longest_title, title_len)

                if 3 <= title_len <= 15 and 10 <= para_len < 60:
                    can_be_used += 1


            else:
                writer.writerow([len(line["trimmed_title_tokenized"]), "NO TLDR"])
            i += 1
            
            # if i == 1000:
            #     break
with open("tifu_cumulative_stats.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Can be used", can_be_used])
    writer.writerow(["Average paragraph length", paragraph_length_sum/has_tldr])
    writer.writerow(["Max paragraph length", longest_paragraph])
    writer.writerow(["Min paragraph length", shortest_paragraph])
    writer.writerow(["--------------------------"])
    writer.writerow(["Average title length", title_length_sum/has_tldr])
    writer.writerow(["Max title length", longest_title])
    writer.writerow(["Min title length", shortest_title])