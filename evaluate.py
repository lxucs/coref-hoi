from run import Runner
import sys


def evaluate(config_name, gpu_id, saved_suffix):
    runner = Runner(config_name, gpu_id)
    model = runner.initialize_model(saved_suffix)

    examples_train, examples_dev, examples_test = runner.data.get_tensor_examples()
    stored_info = runner.data.get_stored_info()

    # runner.evaluate(model, examples_dev, stored_info, 0, official=True, conll_path=runner.config['conll_eval_path'])  # Eval dev
    # print('=================================')
    runner.evaluate(model, examples_test, stored_info, 0, official=True, conll_path=runner.config['conll_test_path'])  # Eval test


if __name__ == '__main__':
    config_name, saved_suffix = sys.argv[1], sys.argv[2]
    if len(sys.argv) < 4:
        gpu_id = None
    else:
        gpu_id = int(sys.argv[3])
        
    evaluate(config_name, gpu_id, saved_suffix)
