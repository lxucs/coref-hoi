import json
from spacy.lang.en import English
from preprocess import get_document
import argparse
import util
from tensorize import CorefDataProcessor
from run import Runner
import logging
logging.getLogger().setLevel(logging.CRITICAL)


def create_spacy_tokenizer():
    nlp = English()
    sentencizer = nlp.create_pipe('sentencizer')
    nlp.add_pipe(sentencizer)


def get_document_from_string(string, seg_len, bert_tokenizer, spacy_tokenizer, genre='nw'):
    doc_key = genre  # See genres in experiment config
    doc_lines = []

    # Build doc_lines
    for token in spacy_tokenizer(string):
        cols = [genre] + ['-'] * 11
        cols[3] = token.text
        doc_lines.append('\t'.join(cols))
        if token.is_sent_end:
            doc_lines.append('\n')

    doc = get_document(doc_key, doc_lines, 'english', seg_len, bert_tokenizer)
    return doc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, required=True,
                        help='Configuration name in experiments.conf')
    parser.add_argument('--model_identifier', type=str, required=True,
                        help='Model identifier to load')
    parser.add_argument('--gpu_id', type=int, default=None,
                        help='GPU id; CPU by default')
    parser.add_argument('--seg_len', type=int, default=512)
    parser.add_argument('--jsonlines_path', type=str, default=None,
                        help='Path to custom input from file; input from console by default')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save output')
    args = parser.parse_args()

    runner = Runner(args.config_name, args.gpu_id)
    model = runner.initialize_model(args.model_identifier)
    data_processor = CorefDataProcessor(runner.config)

    if args.jsonlines_path:
        # Input from file
        with open(args.jsonlines_path, 'r') as f:
            lines = f.readlines()
        docs = [json.loads(line) for line in lines]
        tensor_examples, stored_info = data_processor.get_tensor_examples_from_custom_input(docs)
        predicted_clusters, _, _ = runner.predict(model, tensor_examples)

        if args.output_path:
            with open(args.output_path, 'w') as f:
                for i, doc in enumerate(docs):
                    doc['predicted_clusters'] = predicted_clusters[i]
                    f.write(json.dumps(doc))
            print(f'Saved prediction in {args.output_path}')
    else:
        # Interactive input
        model.to(model.device)
        nlp = English()
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        while True:
            input_str = str(input('Input document:'))
            bert_tokenizer, spacy_tokenizer = data_processor.tokenizer, nlp
            doc = get_document_from_string(input_str, args.seg_len, bert_tokenizer, nlp)
            tensor_examples, stored_info = data_processor.get_tensor_examples_from_custom_input([doc])
            predicted_clusters, _, _ = runner.predict(model, tensor_examples)

            subtokens = util.flatten(doc['sentences'])
            print('---Predicted clusters:')
            for cluster in predicted_clusters[0]:
                mentions_str = [' '.join(subtokens[m[0]:m[1]+1]) for m in cluster]
                mentions_str = [m.replace(' ##', '') for m in mentions_str]
                mentions_str = [m.replace('##', '') for m in mentions_str]
                print(mentions_str)  # Print out strings
                # print(cluster)  # Print out indices
