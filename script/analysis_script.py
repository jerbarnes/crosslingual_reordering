import argparse


def open_file(file):
    gold = []
    pred = []
    examples = []

    for line in open(file):
        try:
            if line.split()[0] in ["0", "1", "2", "3", "4"]:
                g, p, e = line.split()[0], line.split()[1], line.split()[2:]
                gold.append(g)
                pred.append(p)
                examples.append(e)
        except:
            print(line)

    return gold, pred, examples

def compare_files(file1, file2):

    gold1, preds1, examples1 = open_file(file1)
    gold2, preds2, examples2 = open_file(file2)

    assert len(gold1) == len(gold2)

    for i, (p1, p2) in enumerate(zip(preds1, preds2)):
        if p1 != p2 and p2 == gold2[i]:
            # print("{0}\t{1}\t{2}\t{3}  ||  {4}".format(gold2[i], p1, p2, " ".join(examples1[i]), " ".join(examples2[i])))
            print("{0}\t{1}\t{2}\t{3}".format(gold2[i], p1, p2, " ".join(examples1[i])))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("file1")
    parser.add_argument("file2")
    args = parser.parse_args()

    compare_files(args.file1, args.file2)
