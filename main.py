from src.Hyperparams import HparamsDefiner
from src.AbstractTrainer import AbstractModelTrainer
from src.BaseObjects import PredictionModelType, DetectionModelType, DBConnectorType

from matplotlib import pyplot as plt

if __name__ == '__main__':

    alg_list = [PredictionModelType.DSANET, PredictionModelType.LSTM, PredictionModelType.MARIMA]
    mae_list, smape_list, mape_list, probs_list, preds_list = [], [], [], [], []
    original_preds_list = []

    for alg_type in alg_list:
        print(f'ALG_TYPE = {alg_type}')
        parser = HparamsDefiner(data_name="HAI").get_parser()
        hyperparams = parser.parse_args()
        print("hyperparams", hyperparams)

        print(f'RUNNING ON CPU')
        abs_trainer = AbstractModelTrainer(hyperparams, alg_type, DetectionModelType.DeepSVDD, DBConnectorType.POSTGRES)
        abs_trainer.fit()
        abs_trainer.calculate_metrics()

        num_epochs, mae, smape, mape, trainig_time, pred_time, detection_time, num_parameters, cpu_count, cpu_times, disk_usage, used_memory = abs_trainer.get_metrics_report()
        mae_list.append(mae)
        smape_list.append(smape)
        mape_list.append(mape)

        preds = abs_trainer.get_predictions()
        preds_list.append(preds)

        original_preds = abs_trainer.get_original_predictions()
        original_preds_list.append(original_preds)

        probs = abs_trainer.get_proba_predictions()
        probs_list.append(probs)

        abs_trainer.save_report()

    for i in range(len(mae_list)):
        plt.plot(mae_list[i], label=alg_list[i])
    plt.title("MAE")
    plt.legend()
    plt.xlabel("Номер датчика")
    plt.ylabel("Величина ошибки")
    plt.show()

    for i in range(len(smape_list)):
        plt.plot(smape_list[i], label=alg_list[i])
    plt.title("SMAPE")
    plt.legend()
    plt.xlabel("Номер датчика")
    plt.ylabel("Величина ошибки")
    plt.show()

    for i in range(len(mape_list)):
        plt.plot(mape_list[i], label=alg_list[i])
    plt.title("MAPE")
    plt.legend()
    plt.xlabel("Номер датчика")
    plt.ylabel("Величина ошибки")
    plt.show()

    for i in range(len(probs_list)):
        plt.plot(probs_list[i], label=alg_list[i])
    plt.title("PROBS")
    plt.legend()
    plt.xlabel("Номер датчика")
    plt.ylabel("Вероятность наличия аномалии")
    plt.show()

    for i in range(len(preds_list)):
        for j in range(len(preds_list[i])):
            plt.plot(preds_list[i][j], label=alg_list[i])
            plt.plot(original_preds_list[i][j], label="original")
            plt.title("PREDICTIONS")
            plt.legend()
            plt.xlabel("Номер временного ряда")
            plt.ylabel("Значение временного ряда")
            plt.show()
