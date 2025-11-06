poetry run train --reduce-method extra-trees

poetry run train --reduce-method all

poetry run train --reduce-method minfo



poetry run evaluate --model-name xgb_select_20_by_minfo__stop_auc_tune_hyperopt --model-version 1

poetry run evaluate --model-name xgb_select_20_by_extra-trees__stop_auc_tune_hyperopt --model-version 1

poetry run evaluate --model-name xgb_all__stop_auc_tune_hyperopt --model-version 2