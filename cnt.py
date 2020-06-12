import jsonlines

length = 0
number =0
word =set()

#
# with open("dataset/mnli/train.jsonl", "r+", encoding="utf8") as f:
#     for line in jsonlines.Reader(f):
#         for item in line:
#
#             s1 = item["sentence1"].split(" ")
#             s2 = item["sentence2"].split(" ")
#
#             for w in s1+s2:
#                 word.add(w)
#             length+=len(s1)+len(s2)
#             number+=2
#             if number%1000==0:
#                 print("processing %d samples"%number)
#
# 9.318120989326369
# 14564

import csv
with open("dataset/mnli/train.tsv", "r") as f:
    reader = csv.reader(f, delimiter="\t", quotechar=None)
    lines = []
    for line in reader:
        s1 = line[8].split(" ")
        s2 = line[9].split(" ")
        print(len(s1),len(s2))
        for w in s1+s2:
            word.add(w)

        length+=len(s1)+len(s2)
        number+=2
        if number%1000==0:
            print("processing %d samples"%number)

#
# 15.016812960430656
# 189762
print(length/number)
print(len(word))

