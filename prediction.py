import json
from tqdm import tqdm
from eval_methods import *
from utils import *


class Predictor:
    def __init__(self, use_cuda, model, window_size, n_features, batch_size, pred_args, summary_file_name="summary.txt"):
        self.model = model
        self.window_size = window_size
        self.n_features = n_features
        self.dataset = pred_args["dataset"]
        self.target_dims = pred_args["target_dims"]
        self.scale_scores = pred_args["scale_scores"]
        self.q = pred_args["q"]
        self.level = pred_args["level"]
        self.dynamic_pot = pred_args["dynamic_pot"]
        self.use_mov_av = pred_args["use_mov_av"]
        self.gamma = pred_args["gamma"]
        self.reg_level = pred_args["reg_level"]
        self.save_path = pred_args["save_path"]
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.pred_args = pred_args
        self.summary_file_name = summary_file_name
        self.device = device = "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"


    def get_score(self, values):   
        print("Predicting and calculating anomaly scores..")
        data = SlidingWindowDataset(values, self.window_size, self.target_dims)
        loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        pred_err = []
        recon_err = []
        count = 0
        last = 1
        for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                x = x.permute(0, 2, 1)
                y = y.permute(0, 2, 1)

                preds, recons = self.model(x, y) 
                forecast_loss = torch.sqrt((y.squeeze(-1) - preds.squeeze(-1)) ** 2)  
                recon_loss = torch.sqrt((x.squeeze(-1) - recons.squeeze(-1)) ** 2)     
                
                pred_err.append(forecast_loss.detach().cpu().numpy())
                recon_err.append(recon_loss.detach().cpu().numpy())

        pred_b_err = np.concatenate(pred_err, axis=0) 
        recon_b_err = np.concatenate(recon_err, axis=0)  
        recon_b_err = recon_b_err.mean(axis=-1)
        
        anomaly_scores = pred_b_err + self.gamma * recon_b_err

        if self.scale_scores:
            q75, q25 = np.percentile(anomaly_scores, [75, 25])
            iqr = q75 - q25
            median = np.median(anomaly_scores)
            anomaly_scores = (anomaly_scores - median) / (1+iqr)

        df_dict = {}
        for i in range(preds.shape[1]):
            df_dict[f"True_{i}"] = values[self.window_size:, i]
            df_dict[f"A_Score_{i}"] = anomaly_scores[:, i]

        df = pd.DataFrame(df_dict)

        return df, anomaly_scores

    def predict_anomalies(self, train, test, true_anomalies, load_scores=False, save_output=True,
                          scale_scores=False):

        if load_scores:
            print("Loading anomaly scores")

            train_pred_df = pd.read_pickle(f"{self.save_path}/train_output.pkl")
            test_pred_df = pd.read_pickle(f"{self.save_path}/test_output.pkl")

            train_anomaly_scores = train_pred_df['A_Score_Global'].values
            test_anomaly_scores = test_pred_df['A_Score_Global'].values

        else:
            train_pred_df, train_anomaly_scores = self.get_score(train)  
            test_pred_df, test_anomaly_scores = self.get_score(test)

            global root_cause_score
            root_cause_score = torch.topk(torch.tensor(test_anomaly_scores), 5)
            test_anomaly_scores = np.max(test_anomaly_scores, axis=1)

            train_anomaly_scores = normalize_anomaly_scores(train_anomaly_scores)
            test_anomaly_scores = normalize_anomaly_scores(test_anomaly_scores)


        if self.use_mov_av:
            smoothing_window = int(self.batch_size * self.window_size * 0.05)
            train_anomaly_scores = pd.DataFrame(train_anomaly_scores).ewm(span=smoothing_window).mean().values.flatten()
            test_anomaly_scores = pd.DataFrame(test_anomaly_scores).ewm(span=smoothing_window).mean().values.flatten()

        out_dim = self.n_features if self.target_dims is None else len(self.target_dims)
        all_preds = np.zeros((len(test_pred_df), out_dim))


        true_anomalies = np.array(true_anomalies).reshape((-1))
        bf_eval = bf_search(test_anomaly_scores, true_anomalies, start=0, end=2, step_num=100, verbose=False)
        for k, v in bf_eval.items():
            bf_eval[k] = v 

        test_pred_df["A_Score_Global"] = test_anomaly_scores
        test_pred_df["A_True_Global"] = true_anomalies
        test_pred_df["Thresh_Global"] = bf_eval["threshold"]

        train_pred_df.to_pickle(f"{self.save_path}/train_output.pkl")
        test_pred_df.to_pickle(f"{self.save_path}/test_output.pkl")

        summary = {"bf_result": bf_eval}
        with open(f"{self.save_path}/{self.summary_file_name}", "w") as f:
            json.dump(str(summary), f, indent=2)

        print("-- Done.")
