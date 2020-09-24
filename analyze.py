from run import Runner
import util
import json
import pickle
from os.path import join
import os
from collections import defaultdict

singular_pronouns = ['i', 'me', 'my', 'mine', 'myself', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'yourself']
plural_pronouns = ['they', 'them', 'their', 'theirs', 'themselves', 'we', 'us', 'our', 'ours', 'ourselves', 'yourselves']
ambiguous_pronouns = ['you', 'your', 'yours']
valid_pronouns = singular_pronouns + plural_pronouns + ambiguous_pronouns


def get_prediction_path(config, config_name, saved_suffix, suffix=''):
    dir_analysis = join(config['data_dir'], 'analysis')
    os.makedirs(dir_analysis, exist_ok=True)

    name = f'pred_{config_name}_{saved_suffix}{suffix}.bin'
    path = join(dir_analysis, name)
    return path


def get_prediction(config_name, saved_suffix, gpu_id):
    runner = Runner(config_name, gpu_id)
    conf = runner.config

    path = get_prediction_path(conf, config_name, saved_suffix)
    if os.path.exists(path):
        # Load if saved
        with open(path, 'rb') as f:
            prediction = pickle.load(f)
        print('Loaded prediction from %s' % path)
    else:
        # Get prediction
        model = runner.initialize_model(saved_suffix)
        examples_train, examples_dev, examples_test = runner.data.get_tensor_examples()
        stored_info = runner.data.get_stored_info()

        samples_test = [example[1] for example in examples_test]
        predicted_clusters, predicted_spans, predicted_antecedents = runner.predict(model, samples_test)
        prediction = (predicted_clusters, predicted_spans, predicted_antecedents)

        # Save
        with open(path, 'wb') as f:
            pickle.dump(prediction, f)
        print('Prediction saved in %s' % path)

    return prediction


def get_prediction_wo_hoi(config_name, saved_suffix, gpu_id):
    runner = Runner(config_name, gpu_id)
    conf = runner.config

    suffix = '_noHOI'
    path = get_prediction_path(conf, config_name, saved_suffix, suffix)
    if os.path.exists(path):
        # Load if saved
        with open(path, 'rb') as f:
            prediction = pickle.load(f)
        print('Loaded prediction from %s' % path)
    else:
        # Get prediction
        model = runner.initialize_model(saved_suffix)
        examples_train, examples_dev, examples_test = runner.data.get_tensor_examples()
        stored_info = runner.data.get_stored_info()

        # Turn off HOI after model initialization
        if '_cm' in config_name:
            conf['coref_depth'] = 1
            conf['higher_order'] = 'attended_antecedent'
        elif '_d2' in config_name or '_sc' in config_name or '_ee' in config_name:
            conf['coref_depth'] = 1

        samples_test = [example[1] for example in examples_test]
        predicted_clusters, predicted_spans, predicted_antecedents = runner.predict(model, samples_test)
        prediction = (predicted_clusters, predicted_spans, predicted_antecedents)

        # Save
        with open(path, 'wb') as f:
            pickle.dump(prediction, f)
        print('Prediction saved in %s' % path)

    return prediction


def get_original_samples(config, split='tst'):
    samples = []
    paths = {
        'trn': join(config['data_dir'], f'train.english.{config["max_segment_len"]}.jsonlines'),
        'dev': join(config['data_dir'], f'dev.english.{config["max_segment_len"]}.jsonlines'),
        'tst': join(config['data_dir'], f'test.english.{config["max_segment_len"]}.jsonlines')
    }
    with open(paths[split]) as fin:
        for line in fin.readlines():
            data = json.loads(line)
            samples.append(data)
    return samples


def get_gold_to_cluster_id(example_list):
    gold_to_cluster_id = []  # 0 means not in cluster
    non_anaphoric = []  # Firstly appeared mention in a cluster
    for i, example in enumerate(example_list):
        gold_to_cluster_id.append(defaultdict(int))
        non_anaphoric.append(set())

        clusters = example['clusters']
        clusters = [sorted(cluster) for cluster in clusters]  # Sort mention
        for c_i, c in enumerate(clusters):
            non_anaphoric[i].add(tuple(c[0]))
            for m in c:
                gold_to_cluster_id[i][tuple(m)] = c_i + 1
    return gold_to_cluster_id, non_anaphoric


def check_singular_plural_cluster(cluster):
    """ Cluster with text """
    singular, plural, contain_ambiguous = False, False, False
    for m in cluster:
        if singular and plural:
            break
        m = m.lower()
        if not singular:
            singular = (m in singular_pronouns)
        if not plural:
            plural = (m in plural_pronouns)
    for m in cluster:
        m = m.lower()
        if m in ambiguous_pronouns:
            contain_ambiguous = True
            break
    return singular, plural, contain_ambiguous


def analyze(config_name, saved_suffix, gpu_id):
    runner = Runner(config_name, gpu_id)
    conf = runner.config

    # Get gold clusters
    example_list = get_original_samples(conf)
    gold_to_cluster_id, non_anaphoric = get_gold_to_cluster_id(example_list)

    # Get prediction
    predicted_clusters, predicted_spans, predicted_antecedents = get_prediction(config_name, saved_suffix, gpu_id)

    # Get cluster text
    cluster_list = []
    subtoken_list = []
    for i, example in enumerate(example_list):
        subtokens = util.flatten(example['sentences'])
        subtoken_list.append(subtokens)
        cluster_list.append([[' '.join(subtokens[m[0]: m[1] + 1]) for m in c] for c in predicted_clusters[i]])

    # Get cluster stats
    num_clusters, num_singular_clusters, num_plural_clusters, num_mixed_clusters, num_mixed_ambiguous = 0, 0, 0, 0, 0
    for clusters in cluster_list:
        # print(clusters)
        for c in clusters:
            singular, plural, contain_ambiguous = check_singular_plural_cluster(c)
            num_clusters += 1
            if singular and plural:
                num_mixed_clusters += 1
                if contain_ambiguous:
                    num_mixed_ambiguous += 1
            if singular:
                num_singular_clusters += 1
            if plural:
                num_plural_clusters += 1

    # Get antecedent stats
    fl, fn, wl, correct = 0, 0, 0, 0  # False Link, False New, Wrong Link
    s_to_p, p_to_s = 0, 0
    num_non_gold, num_total_spans = 0, 0
    for i, antecedents in enumerate(predicted_antecedents):
        antecedents = [(-1, -1) if a == -1 else predicted_spans[i][a] for a in antecedents]
        for j, antecedent in enumerate(antecedents):
            span = predicted_spans[i][j]
            span_cluster_id = gold_to_cluster_id[i][span]
            num_total_spans += 1

            if antecedent == (-1, -1):
                continue

            # Only look at stats of pronouns
            span_text = ' '.join(subtoken_list[i][span[0]: span[1] + 1]).lower()
            antecedent_text = ' '.join(subtoken_list[i][antecedent[0]: antecedent[1] + 1]).lower()
            if span_text not in valid_pronouns or antecedent_text not in valid_pronouns:
                continue

            if span_text in singular_pronouns and antecedent_text in plural_pronouns:
                s_to_p += 1
            elif span_text in plural_pronouns and antecedent_text in singular_pronouns:
                p_to_s += 1

            if span_cluster_id == 0:  # Non-gold span
                num_non_gold += 1
                if antecedent == (-1, -1):
                    correct += 1
                else:
                    fl += 1
            elif span in non_anaphoric[i]:  # Non-anaphoric span
                if antecedent == (-1, -1):
                    correct += 1
                else:
                    fl += 1
            else:
                if antecedent == (-1, -1):
                    fn += 1
                elif span_cluster_id != gold_to_cluster_id[i][antecedent]:
                    wl += 1
                else:
                    correct += 1

    return num_clusters, num_singular_clusters, num_plural_clusters, num_mixed_clusters, num_mixed_ambiguous, fl, fn, wl, correct, \
           num_non_gold, num_total_spans, s_to_p, p_to_s


def analyze2(config_name, saved_suffix, gpu_id):
    runner = Runner(config_name, gpu_id)
    conf = runner.config

    # Get gold clusters
    example_list = get_original_samples(conf)
    gold_to_cluster_id, non_anaphoric = get_gold_to_cluster_id(example_list)

    # Get info
    named_entities, pronouns = [], []
    for example in example_list:
        named_entities.append(util.flatten(example['named_entities']))
        pronouns.append(util.flatten(example['pronouns']))

    # Get normal prediction
    predicted_clusters, predicted_spans, predicted_antecedents = get_prediction(config_name, saved_suffix, gpu_id)
    # Get prediction turning off HOI
    predicted_clusters_nohoi, predicted_spans_nohoi, predicted_antecedents_nohoi = get_prediction_wo_hoi(config_name, saved_suffix, gpu_id)
    # predicted_spans and predicted_spans_nohoi should be almost identical

    # Check wrong->correct and correct->wrong links after turning off HOI
    f2t, t2f, t2t, f2f = [[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]]
    f2t_pct, t2f_pct, t2t_pct, f2f_pct = [], [], [], []
    link_status_wo_hoi = get_link_status(predicted_spans_nohoi, predicted_antecedents_nohoi, gold_to_cluster_id, non_anaphoric)
    link_status_w_hoi = get_link_status(predicted_spans, predicted_antecedents, gold_to_cluster_id, non_anaphoric)
    for doc_i in range(len(link_status_wo_hoi)):
        f2t_doc, t2f_doc, t2t_doc, f2f_doc = [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]
        status_dict_wo_hoi = link_status_wo_hoi[doc_i]
        status_dict_w_hoi = link_status_w_hoi[doc_i]
        for span, link_wo_hoi in status_dict_wo_hoi.items():
            link_w_hoi = status_dict_w_hoi.get(span, None)
            if link_w_hoi is None:
                continue  # Only look at gold mentions in both prediction

            span_type = identify_span_type(named_entities[doc_i], pronouns[doc_i], span)

            if link_wo_hoi:
                if link_w_hoi:
                    t2t_doc[span_type] += 1
                else:
                    t2f_doc[span_type] += 1
            else:
                if link_w_hoi:
                    f2t_doc[span_type] += 1
                else:
                    f2f_doc[span_type] += 1
        total_link = sum(f2t_doc) + sum(t2f_doc) + sum(t2t_doc) + sum(f2f_doc)
        if total_link == 0:
            print('Zero gold mention; should not happen often')
            continue
        for span_type in range(3):
            f2t[span_type].append(f2t_doc[span_type])
        for span_type in range(3):
            t2f[span_type].append(t2f_doc[span_type])
        for span_type in range(3):
            t2t[span_type].append(t2t_doc[span_type])
        for span_type in range(3):
            f2f[span_type].append(f2f_doc[span_type])
        f2t_pct.append(sum(f2t_doc) * 100 / total_link)
        t2f_pct.append(sum(t2f_doc) * 100 / total_link)
        t2t_pct.append(sum(t2t_doc) * 100 / total_link)
        f2f_pct.append(sum(f2f_doc) * 100 / total_link)

    f2t_total, t2f_total, t2t_total, f2f_total = 0, 0, 0, 0
    f2t_type_pct, t2f_type_pct, t2t_type_pct, f2f_type_pct = [[], [], []], [[], [], []], [[], [], []], [[], [], []]
    for doc_i in range(len(f2t[0])):
        f2t_doc_sum = f2t[0][doc_i] + f2t[1][doc_i] + f2t[2][doc_i]
        t2f_doc_sum = t2f[0][doc_i] + t2f[1][doc_i] + t2f[2][doc_i]
        t2t_doc_sum = t2t[0][doc_i] + t2t[1][doc_i] + t2t[2][doc_i]
        f2f_doc_sum = f2f[0][doc_i] + f2f[1][doc_i] + f2f[2][doc_i]
        if f2t_doc_sum > 0:
            for span_type in range(3):
                f2t_type_pct[span_type].append(f2t[span_type][doc_i] * 100 / f2t_doc_sum)
        if t2f_doc_sum > 0:
            for span_type in range(3):
                t2f_type_pct[span_type].append(t2f[span_type][doc_i] * 100 / t2f_doc_sum)
        if t2t_doc_sum > 0:
            for span_type in range(3):
                t2t_type_pct[span_type].append(t2t[span_type][doc_i] * 100 / t2t_doc_sum)
        if f2f_doc_sum > 0:
            for span_type in range(3):
                f2f_type_pct[span_type].append(f2f[span_type][doc_i] * 100 / f2f_doc_sum)
        f2t_total += f2t_doc_sum
        t2f_total += t2f_doc_sum
        t2t_total += t2t_doc_sum
        f2f_total += f2f_doc_sum

    return f2t_total, t2f_total, t2t_total, f2f_total,\
           sum(f2t_pct) / len(f2t_pct), sum(t2f_pct) / len(t2f_pct), sum(t2t_pct) / len(t2t_pct), sum(f2f_pct) / len(f2f_pct), \
           mean(f2t_type_pct[0]), mean(f2t_type_pct[1]), mean(f2t_type_pct[2]), \
           mean(t2f_type_pct[0]), mean(t2f_type_pct[1]), mean(t2f_type_pct[2]), \
           mean(t2t_type_pct[0]), mean(t2t_type_pct[1]), mean(t2t_type_pct[2]), \
           mean(f2f_type_pct[0]), mean(f2f_type_pct[1]), mean(f2f_type_pct[2])


def mean(l):
    return sum(l) / len(l)


def identify_span_type(named_entities_doc, pronouns_doc, span):
    """ 1: pronoun; 2: named entity; 0: other(nominal nouns) """
    # Check pronoun
    if pronouns_doc[span[0]: span[1] + 1] == ([True] * (span[1] - span[0] + 1)):
        return 1
    # Check named entity
    entity_text = ''.join(named_entities_doc[span[0]: span[1] + 1])
    if entity_text.count('(') == 1 and entity_text.count(')') == 1:
        return 2
    return 0


def get_link_status(predicted_spans, predicted_antecedents, gold_to_cluster_id, non_anaphoric):
    """
    :param predicted_spans: from get_prediction()
    :param predicted_antecedents:
    :param gold_to_cluster_id, non_anaphoric: from get_gold_to_cluster_id()
    :return: dict of gold spans indicating wrong(False) or correct(True) link
    """
    link_status = []
    for doc_i in range(len(predicted_spans)):
        status_dict = {}  # Only for gold mentions
        spans = predicted_spans[doc_i]
        for span_i, antecedent_i in enumerate(predicted_antecedents[doc_i]):
            span_cluster_id = gold_to_cluster_id[doc_i][spans[span_i]]
            if span_cluster_id == 0:
                continue
            if antecedent_i == -1:
                status_dict[spans[span_i]] = (spans[span_i] in non_anaphoric[doc_i])
            else:
                antecedent_cluster_id = gold_to_cluster_id[doc_i][spans[antecedent_i]]
                status_dict[spans[span_i]] = (span_cluster_id == antecedent_cluster_id)
        link_status.append(status_dict)
    return link_status


if __name__ == '__main__':
    gpu_id = 6

    experiments = [('train_bert_large_ml0_d1', 'May20_10-25-13_65000'),
                   ('train_bert_large_ml0_d1', 'May21_00-29-00_66000'),
                   ('train_bert_large_ml0_d1', 'May21_17-04-35_50000'),
                   ('train_bert_large_ml0_d1', 'May24_03-33-55_58000')]

    results_final = None
    for experiment in experiments:
        # results = analyze(*experiment, gpu_id=gpu_id)
        results = analyze2(*experiment, gpu_id=gpu_id)
        if results is None:
            continue

        if results_final is None:
            results_final = results
        else:
            results_final = [r + results[i] for i, r in enumerate(results_final)]

        # print('%s_%s: # clusters: %d; # singular clusters: %d; # plural clusters: %d; # mixed clusters: %d; '
        #       'FL %d; FN: %d; WL: %d; CORRECT %d; # gold spans: %d; # total spans: %d' % (*experiment, *results))

    results_final = [r / len(experiments) for r in results_final]

    # Analyze
    # print('Avg: # clusters: %.3f; # singular clusters: %.3f; # plural clusters: %.3f; # mixed clusters: %.3f; # mixed with ambiguous: %.3f; '
    #       'FL %.3f; FN: %.3f; WL: %.3f; CORRECT %.3f; # gold spans: %.3f; # total spans: %.3f; # S to P: %.3f; # P to S: %.3f' % (*results_final,))

    # Analyze2
    print('f2t, t2f, t2t, f2f: %.2f, %.2f, %.2f, %.2f;\t%.2f%%, %.2f%%, %.2f%%, %.2f%%;\n%.2f%%, %.2f%%, %.2f%%\n%.2f%%, %.2f%%, %.2f%%\n%.2f%%, %.2f%%, %.2f%%\n%.2f%%, %.2f%%, %.2f%%' % (*results_final,))
