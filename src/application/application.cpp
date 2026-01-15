/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/application.h>

#include <LightGBM/boosting.h>
#include <LightGBM/dataset.h>
#include <LightGBM/dataset_loader.h>
#include <LightGBM/metric.h>
#include <LightGBM/network.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/prediction_early_stop.h>
#include <LightGBM/cuda/vector_cudahost.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/openmp_wrapper.h>
#include <LightGBM/utils/text_reader.h>

#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "predictor.hpp"

namespace LightGBM {

// ============================================================================
// Applicationコンストラクタ：初期化処理
// ============================================================================
Application::Application(int argc, char** argv) {
  // 【ステップ1】コマンドライン引数からパラメータを読み込む
  LoadParameters(argc, argv);
  
  // 【ステップ2】OpenMPのスレッド数を設定（並列処理のための設定）
  OMP_SET_NUM_THREADS(config_.num_threads);
  
  // 【ステップ3】データファイルが指定されているかチェック
  if (config_.data.size() == 0 && config_.task != TaskType::kConvertModel) {
    Log::Fatal("No training/prediction data, application quit");
  }

  // 【ステップ4】GPU（CUDA）を使用する場合の設定
  if (config_.device_type == std::string("cuda")) {
      LGBM_config_::current_device = lgbm_device_cuda;
  }
}

Application::~Application() {
  if (config_.is_parallel) {
    Network::Dispose();
  }
}

// ============================================================================
// LoadParameters：コマンドライン引数と設定ファイルからパラメータを読み込む
// ============================================================================
void Application::LoadParameters(int argc, char** argv) {
  std::unordered_map<std::string, std::vector<std::string>> all_params;
  std::unordered_map<std::string, std::string> params;
  
  // 【ステップ1】コマンドライン引数を解析してパラメータマップに追加
  // 例: "objective=regression" → {"objective": ["regression"]}
  for (int i = 1; i < argc; ++i) {
    Config::KV2Map(&all_params, argv[i]);
  }
  
  // 【ステップ2】設定ファイルからパラメータを読み込む（configオプションが指定されている場合）
  bool config_file_ok = true;
  if (all_params.count("config") > 0) {
    TextReader<size_t> config_reader(all_params["config"][0].c_str(), false);
    config_reader.ReadAllLines();
    if (!config_reader.Lines().empty()) {
      for (auto& line : config_reader.Lines()) {
        // "#"以降のコメントを削除
        if (line.size() > 0 && std::string::npos != line.find_first_of("#")) {
          line.erase(line.find_first_of("#"));
        }
        line = Common::Trim(line);  // 前後の空白を削除
        if (line.size() == 0) {
          continue;  // 空行はスキップ
        }
        Config::KV2Map(&all_params, line.c_str());
      }
    } else {
      config_file_ok = false;
    }
  }
  
  // 【ステップ3】ログの詳細度を設定
  Config::SetVerbosity(all_params);
  
  // 【ステップ4】重複パラメータを削除（最初の値のみを保持）
  Config::KeepFirstValues(all_params, &params);
  
  if (!config_file_ok) {
    Log::Warning("Config file %s doesn't exist, will ignore", params["config"].c_str());
  }
  
  // 【ステップ5】パラメータのエイリアスを変換（例: "num_iterations" → "num_iteration"）
  ParameterAlias::KeyAliasTransform(&params);
  
  // 【ステップ6】設定オブジェクトにパラメータを設定
  config_.Set(params);
  Log::Info("Finished loading parameters");
}

// ============================================================================
// LoadData：訓練データと検証データを読み込む
// ============================================================================
void Application::LoadData() {
  auto start_time = std::chrono::high_resolution_clock::now();
  std::unique_ptr<Predictor> predictor;
  
  // 【ステップ1】継続訓練の場合、既存モデルから予測関数を取得
  // （初期モデルから予測値を計算して、それを初期スコアとして使用するため）
  PredictFunction predict_fun = nullptr;
  if (boosting_->NumberOfTotalModel() > 0 && config_.task != TaskType::KRefitTree) {
    predictor.reset(new Predictor(boosting_.get(), 0, -1, true, false, false, false, -1, -1));
    predict_fun = predictor->GetPredictFunction();
  }

  // 【ステップ2】分散学習の場合、ランダムシードを同期
  if (config_.is_data_based_parallel) {
    config_.data_random_seed = Network::GlobalSyncUpByMin(config_.data_random_seed);
  }

  // 【ステップ3】データセットローダーを作成
  Log::Debug("Loading train file...");
  DatasetLoader dataset_loader(config_, predict_fun,
                               config_.num_class, config_.data.c_str());
  
  // 【ステップ4】訓練データを読み込む
  if (config_.is_data_based_parallel) {
    // 分散学習の場合：各マシンが自分の担当分のデータを読み込む
    train_data_.reset(dataset_loader.LoadFromFile(config_.data.c_str(),
                                                  Network::rank(), Network::num_machines()));
  } else {
    // 単一マシンの場合：全データを読み込む
    train_data_.reset(dataset_loader.LoadFromFile(config_.data.c_str(), 0, 1));
  }
  
  // 【ステップ5】バイナリファイルとして保存する場合
  if (config_.save_binary) {
    train_data_->SaveBinaryFile(nullptr);
  }
  
  // 【ステップ6】訓練データ用の評価指標を作成
  // 例: "rmse", "auc", "multi_logloss" など
  if (config_.is_provide_training_metric) {
    for (auto metric_type : config_.metric) {
      auto metric = std::unique_ptr<Metric>(Metric::CreateMetric(metric_type, config_));
      if (metric == nullptr) {
        continue;
      }
      metric->Init(train_data_->metadata(), train_data_->num_data());
      train_metric_.push_back(std::move(metric));
    }
  }
  train_metric_.shrink_to_fit();

  if (!config_.metric.empty()) {
    // only when have metrics then need to construct validation data

    // Add validation data, if it exists
    for (size_t i = 0; i < config_.valid.size(); ++i) {
      Log::Debug("Loading validation file #%zu...", (i + 1));
      // add
      auto new_dataset = std::unique_ptr<Dataset>(
        dataset_loader.LoadFromFileAlignWithOtherDataset(
          config_.valid[i].c_str(),
          train_data_.get()));
      valid_datas_.push_back(std::move(new_dataset));
      // need save binary file
      if (config_.save_binary) {
        valid_datas_.back()->SaveBinaryFile(nullptr);
      }

      // add metric for validation data
      valid_metrics_.emplace_back();
      for (auto metric_type : config_.metric) {
        auto metric = std::unique_ptr<Metric>(Metric::CreateMetric(metric_type, config_));
        if (metric == nullptr) {
          continue;
        }
        metric->Init(valid_datas_.back()->metadata(),
                     valid_datas_.back()->num_data());
        valid_metrics_.back().push_back(std::move(metric));
      }
      valid_metrics_.back().shrink_to_fit();
    }
    valid_datas_.shrink_to_fit();
    valid_metrics_.shrink_to_fit();
  }
  auto end_time = std::chrono::high_resolution_clock::now();
  // output used time on each iteration
  Log::Info("Finished loading data in %f seconds",
            std::chrono::duration<double, std::milli>(end_time - start_time) * 1e-3);
}

// ============================================================================
// InitTrain：訓練前の初期化処理
// ============================================================================
void Application::InitTrain() {
  // 【ステップ1】分散学習の場合、ネットワークを初期化
  if (config_.is_parallel) {
    Network::Init(config_);
    Log::Info("Finished initializing network");
    // 分散学習で使用するランダムシードを同期
    config_.feature_fraction_seed =
      Network::GlobalSyncUpByMin(config_.feature_fraction_seed);
    config_.feature_fraction =
      Network::GlobalSyncUpByMin(config_.feature_fraction);
    config_.drop_seed =
      Network::GlobalSyncUpByMin(config_.drop_seed);
  }

  // 【ステップ2】ブースティングオブジェクトを作成
  // "gbdt", "dart", "goss", "rf" などのタイプを指定可能
  boosting_.reset(
    Boosting::CreateBoosting(config_.boosting,
                             config_.input_model.c_str()));
  
  // 【ステップ3】目的関数を作成
  // "regression", "binary", "multiclass", "lambdarank" など
  objective_fun_.reset(
    ObjectiveFunction::CreateObjectiveFunction(config_.objective,
                                               config_));
  
  // 【ステップ4】訓練データを読み込む
  LoadData();
  
  // 【ステップ5】バイナリファイル保存のみのタスクの場合は終了
  if (config_.task == TaskType::kSaveBinary) {
    Log::Info("Save data as binary finished, exit");
    exit(0);
  }
  
  // 【ステップ6】目的関数を初期化
  objective_fun_->Init(train_data_->metadata(), train_data_->num_data());
  
  // 【ステップ7】ブースティングオブジェクトを初期化
  // ここで、データセット、目的関数、評価指標を設定
  boosting_->Init(&config_, train_data_.get(), objective_fun_.get(),
                  Common::ConstPtrInVectorWrapper<Metric>(train_metric_));
  
  // 【ステップ8】検証データをブースティングオブジェクトに追加
  for (size_t i = 0; i < valid_datas_.size(); ++i) {
    boosting_->AddValidDataset(valid_datas_[i].get(),
                               Common::ConstPtrInVectorWrapper<Metric>(valid_metrics_[i]));
    Log::Debug("Number of data points in validation set #%zu: %d", i + 1, valid_datas_[i]->num_data());
  }
  Log::Info("Finished initializing training");
}

// ============================================================================
// Train：メインの訓練処理
// ============================================================================
void Application::Train() {
  Log::Info("Started training...");
  
  // 【ステップ1】ブースティングオブジェクトのTrain()メソッドを呼び出し
  // この中で、指定された回数（num_iterations）だけツリーを追加していきます
  // snapshot_freq: 定期的にモデルを保存する頻度
  boosting_->Train(config_.snapshot_freq, config_.output_model);
  
  // 【ステップ2】訓練が完了したら、最終的なモデルをファイルに保存
  boosting_->SaveModelToFile(0, -1, config_.saved_feature_importance_type,
                             config_.output_model.c_str());
  
  // 【ステップ3】モデルをC++のif-else文のコードに変換する場合
  if (config_.convert_model_language == std::string("cpp")) {
    boosting_->SaveModelToIfElse(-1, config_.convert_model.c_str());
  }
  Log::Info("Finished training");
}

void Application::Predict() {
  if (config_.task == TaskType::KRefitTree) {
    // create predictor
    Predictor predictor(boosting_.get(), 0, -1, false, true, false, false, 1, 1);
    predictor.Predict(config_.data.c_str(), config_.output_result.c_str(), config_.header, config_.predict_disable_shape_check,
                      config_.precise_float_parser);
    TextReader<int> result_reader(config_.output_result.c_str(), false);
    result_reader.ReadAllLines();

    size_t nrow = result_reader.Lines().size();
    size_t ncol = 0;
    if (nrow > 0) {
      ncol = Common::StringToArray<int>(result_reader.Lines()[0], '\t').size();
    }
    std::vector<int> pred_leaf;
    pred_leaf.resize(nrow * ncol);

    #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
    for (int irow = 0; irow < static_cast<int>(nrow); ++irow) {
      auto line_vec = Common::StringToArray<int>(result_reader.Lines()[irow], '\t');
      CHECK_EQ(line_vec.size(), ncol);
      for (int i_row_item = 0; i_row_item < static_cast<int>(ncol); ++i_row_item) {
        pred_leaf[irow * ncol + i_row_item] = line_vec[i_row_item];
      }
      // Free memory
      result_reader.Lines()[irow].clear();
    }
    DatasetLoader dataset_loader(config_, nullptr,
                                 config_.num_class, config_.data.c_str());
    train_data_.reset(dataset_loader.LoadFromFile(config_.data.c_str(), 0, 1));
    train_metric_.clear();
    objective_fun_.reset(ObjectiveFunction::CreateObjectiveFunction(config_.objective,
                                                                    config_));
    objective_fun_->Init(train_data_->metadata(), train_data_->num_data());
    boosting_->Init(&config_, train_data_.get(), objective_fun_.get(),
                    Common::ConstPtrInVectorWrapper<Metric>(train_metric_));

    boosting_->RefitTree(pred_leaf.data(), nrow, ncol);
    boosting_->SaveModelToFile(0, -1, config_.saved_feature_importance_type,
                               config_.output_model.c_str());
    Log::Info("Finished RefitTree");
  } else {
    // create predictor
    Predictor predictor(boosting_.get(), config_.start_iteration_predict, config_.num_iteration_predict, config_.predict_raw_score,
                        config_.predict_leaf_index, config_.predict_contrib,
                        config_.pred_early_stop, config_.pred_early_stop_freq,
                        config_.pred_early_stop_margin);
    predictor.Predict(config_.data.c_str(),
                      config_.output_result.c_str(), config_.header, config_.predict_disable_shape_check,
                      config_.precise_float_parser);
    Log::Info("Finished prediction");
  }
}

void Application::InitPredict() {
  boosting_.reset(
    Boosting::CreateBoosting("gbdt", config_.input_model.c_str()));
  Log::Info("Finished initializing prediction, total used %d iterations", boosting_->GetCurrentIteration());
}

void Application::ConvertModel() {
  boosting_.reset(
    Boosting::CreateBoosting(config_.boosting, config_.input_model.c_str()));
  boosting_->SaveModelToIfElse(-1, config_.convert_model.c_str());
}


}  // namespace LightGBM
