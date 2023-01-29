import csv

NB_DOCUMENTS = 348
K = 5

def evaluate_antecedents():
    # correct antecedent = same start/end of anaphor match same start/end of antecedent
    with open("evaluation/eval_antecedents.md", "w") as out_file:
        out_file.write("# Antecedent Evaluation\n\n")
        for i in range(NB_DOCUMENTS):
            gold_file = open(f"evaluation/{i}-gold_antecedents.csv", "r")
            pred_file = open(f"evaluation/{i}-k_best_antecedents.csv", "r")
            gold_reader = list(csv.DictReader(gold_file))
            pred_reader = list(csv.DictReader(pred_file))
            correct_antecedent_at_rank = {1 : 0,
                                          2 : 0,
                                          3 : 0,
                                          4 : 0,
                                          5 : 0}
            doc_key = None
            nb_gold_antecedents = 0
            for gold_row in gold_reader:
                if doc_key is None:
                    doc_key = gold_row["doc_key"]
                nb_gold_antecedents += 1
                for pred_row in pred_reader:
                    gold_tuple = (gold_row["anaphor_start"], gold_row["anaphor_end"], gold_row["antecedent_start"], gold_row["antecedent_end"])
                    pred_tuple = (pred_row["span_start"], pred_row["span_end"], pred_row["antecedent_start"], pred_row["antecedent_end"])
                    if gold_tuple == pred_tuple:
                        correct_antecedent_at_rank[int(pred_row["antecedent_rank"])] += 1
            
            out_file.write(f"## Document {i} ({doc_key}) :\n")
            for rank in range(1, K+1):
                out_file.write(f"Number of correct antecedent(s) at rank {rank}  : {correct_antecedent_at_rank[rank]}/{nb_gold_antecedents}\n")
            out_file.write(f"Number of unfound anaphor(s)/antecedent(s) : {nb_gold_antecedents - sum(correct_antecedent_at_rank.values())}/{nb_gold_antecedents}\n\n")
            gold_file.close()
            pred_file.close()

if __name__ == '__main__':
    evaluate_antecedents()