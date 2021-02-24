from nltk import word_tokenize
from nltk.translate.nist_score import sentence_nist
from util import parse_sts
import argparse
import numpy as np

def main(sts_data):
    """Calculate NIST metric for pairs of strings
    Data is formatted as in the STS benchmark"""

    # TODO 1: define a function to read the data in util
    texts, labels = parse_sts(sts_data)

    print(f"Found {len(texts)} STS pairs")

    # take a sample of sentences so the code runs fast for faster debugging
    # when you're done debugging, you may want to run this on more!
    sample_text = texts[120:140]
    sample_labels = labels[120:140]
    # zip them together to make tuples of text associated with labels
    sample_data = zip(sample_labels, sample_text)

    scores = []
    for label,text in sample_data:
        print(label)
        print(text)
        t1, t2 = text
        print(f"Sentences: {t1}\t{t2}")



        # TODO 2: Calculate NIST for each pair of sentences
        # calculate NIST(a,b) and NIST(b,a) and
        # catch any exceptions and assign 0.0 for that part of the score

        # input tokenized text
        t1_toks = word_tokenize(t1.lower())
        t2_toks = word_tokenize(t2.lower())

        # try / except for each side because of ZeroDivision Error
        # 0.0 is lowest score - give that if ZeroDivision Error
        try:
            nist_1 = sentence_nist([t1_toks, ], t2_toks)
        except ZeroDivisionError:
            # print(f"\n\n\nno NIST, {i}")
            nist_1 = 0.0

        try:
            nist_2 = sentence_nist([t2_toks, ], t1_toks)
        except ZeroDivisionError:
            # print(f"\n\n\nno NIST, {i}")
            nist_2 = 0.0

        # sum to produce one metric
        nist_total = nist_1 + nist_2
        print(f"Label: {label}, NIST: {nist_total:0.02f}\n")
        scores.append(nist_total)


    # TODO 3: find and print the sentences from the sample with the highest and lowest scores
    min_score = np.argmin(scores)
    print(f"Lowest score: {scores[min_score]}")
    print(sample_text[min_score])

    max_score = np.argmax(scores)
    print(f"Highest score: {scores[max_score]}")
    print(sample_text[max_score])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_data", type=str, default="stsbenchmark/sts-dev.csv",
                        help="sts data")
    args = parser.parse_args()

    main(args.sts_data)
