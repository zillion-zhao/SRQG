from get_bing_results import get_eval_bing_results, get_srqg_gen_bing_results
from lists_texts_extractor import extract_eval_lists_texts
from lists_texts_extractor_srqg_gen import extract_srqg_gen_lists_texts
from candidates_extractor_ranker import extract_rank_evaluation_candidates, extract_rank_srqg_ltr_candidates, extract_rank_srqg_gen_candidates

from baselines import run_baselines
from srqg_rule import run_srqg_rule
from evaluation import evaluate


if __name__ == '__main__':
    # 1. get top search results from Bing (for evaluation data, SRQG-LTR, and SRQG-Gen)
    # get_eval_bing_results()
    # get_srqg_gen_bing_results()

    # 2. run lists and texts extractor (for evaluation data, SRQG-LTR, and SRQG-Gen)
    # extract_eval_lists_texts()
    # extract_srqg_gen_lists_texts()

    # 3. run candidates extractor and ranker (for evaluation data, SRQG-LTR, and SRQG-Gen)
    # extract_rank_evaluation_candidates(full_feature=False)
    # extract_rank_srqg_ltr_candidates(full_feature=False)
    extract_rank_srqg_gen_candidates(full_feature=False)

    # 4. run baselines
    # run_baselines(['RTC', 'QLM'], mode='multiple')

    # 5. run our models
    # run_srqg_rule(mode='single')

    # 6. evaluation
    # evaluate()
    pass
