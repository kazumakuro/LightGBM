/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_INCLUDE_LIGHTGBM_APPLICATION_H_
#define LIGHTGBM_INCLUDE_LIGHTGBM_APPLICATION_H_

#include <LightGBM/config.h>
#include <LightGBM/meta.h>

#include <memory>
#include <vector>

namespace LightGBM {

class DatasetLoader;
class Dataset;
class Boosting;
class ObjectiveFunction;
class Metric;

/*!
* \brief LightGBMのメインアプリケーションクラス
* 
* このクラスは、LightGBMの訓練と予測の両方を管理します。
* - Train: 新しいモデルを訓練する
* - Predict: 既存のモデルを使ってテストデータのスコアを予測し、ファイルに保存する
*/
class Application {
 public:
  // コンストラクタ：コマンドライン引数からパラメータを読み込む
  Application(int argc, char** argv);

  /*! \brief デストラクタ */
  ~Application();

  /*! \brief アプリケーションを実行するメイン関数 */
  inline void Run();

 private:
  /*! \brief コマンドライン引数と設定ファイルからパラメータを読み込む */
  void LoadParameters(int argc, char** argv);

  /*! \brief 訓練データと検証データを読み込む */
  void LoadData();

  /*! \brief 訓練前の初期化処理 */
  void InitTrain();

  /*! \brief メインの訓練ロジック */
  void Train();

  /*! \brief 予測前の初期化処理 */
  void InitPredict();

  /*! \brief メインの予測ロジック */
  void Predict();

  /*! \brief モデル変換のロジック */
  void ConvertModel();

  /*! \brief すべての設定パラメータ */
  Config config_;
  /*! \brief 訓練データ */
  std::unique_ptr<Dataset> train_data_;
  /*! \brief 検証データ（複数の検証セットに対応） */
  std::vector<std::unique_ptr<Dataset>> valid_datas_;
  /*! \brief 訓練データ用の評価指標 */
  std::vector<std::unique_ptr<Metric>> train_metric_;
  /*! \brief 検証データ用の評価指標（各検証セットごと） */
  std::vector<std::vector<std::unique_ptr<Metric>>> valid_metrics_;
  /*! \brief ブースティングオブジェクト（GBDT, DART, GOSS, RFなど） */
  std::unique_ptr<Boosting> boosting_;
  /*! \brief 訓練用の目的関数（回帰、分類、ランキングなど） */
  std::unique_ptr<ObjectiveFunction> objective_fun_;
};


// ============================================================================
// Run()メソッド：タスクタイプに応じて適切な処理を実行
// ============================================================================
inline void Application::Run() {
  // タスクタイプに応じて分岐
  if (config_.task == TaskType::kPredict || config_.task == TaskType::KRefitTree) {
    // 【予測タスク】既存のモデルを使って予測を行う
    InitPredict();  // モデルを読み込んで初期化
    Predict();      // 予測を実行
  } else if (config_.task == TaskType::kConvertModel) {
    // 【モデル変換タスク】モデルをif-else文のコードに変換
    ConvertModel();
  } else {
    // 【訓練タスク】新しいモデルを訓練する
    InitTrain();  // データ読み込み、ブースティングオブジェクトの初期化
    Train();      // 訓練を実行
  }
}

}  // namespace LightGBM

#endif   // LIGHTGBM_INCLUDE_LIGHTGBM_APPLICATION_H_
