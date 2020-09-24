from run import Runner


def evaluate(config_name, gpu_id, saved_suffix):
    runner = Runner(config_name, gpu_id)
    model = runner.initialize_model(saved_suffix)

    examples_train, examples_dev, examples_test = runner.data.get_tensor_examples()
    stored_info = runner.data.get_stored_info()

    runner.evaluate(model, examples_dev, stored_info, 0, official=True, conll_path=runner.config['conll_eval_path'])  # Eval dev
    print('=================================')
    runner.evaluate(model, examples_test, stored_info, 0, official=True, conll_path=runner.config['conll_test_path'])  # Eval test


if __name__ == '__main__':
    config_name = 'train_spanbert_large_ml0_cm_fn1000_max_dloss'
    saved_suffix = 'XXX'
    gpu_id = 7
    evaluate(config_name, gpu_id, saved_suffix)
