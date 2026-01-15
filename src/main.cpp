/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/application.h>

#include <iostream>
#include <string>

#ifdef USE_MPI
  #include "network/linkers.h"
#endif

// ============================================================================
// LightGBMのメインエントリーポイント
// ============================================================================
// この関数は、コマンドラインからLightGBMを実行した際の最初の処理です。
// Pythonからlightgbmを使う場合は、C API経由で呼ばれます。
int main(int argc, char** argv) {
  bool success = false;
  try {
    // 【ステップ1】Applicationオブジェクトを作成
    // コンストラクタでコマンドライン引数からパラメータを読み込みます
    LightGBM::Application app(argc, argv);
    
    // 【ステップ2】アプリケーションを実行
    // Run()メソッド内で、タスクタイプ（訓練/予測/モデル変換）に応じて
    // 適切な処理を呼び出します
    app.Run();

#ifdef USE_MPI
    // MPI（分散学習）を使用している場合の後処理
    LightGBM::Linkers::MpiFinalizeIfIsParallel();
#endif

    success = true;
  }
  catch (const std::exception& ex) {
    // エラーハンドリング：標準例外をキャッチ
    std::cerr << "Met Exceptions:" << std::endl;
    std::cerr << ex.what() << std::endl;
  }
  catch (const std::string& ex) {
    // エラーハンドリング：文字列例外をキャッチ
    std::cerr << "Met Exceptions:" << std::endl;
    std::cerr << ex << std::endl;
  }
  catch (...) {
    // エラーハンドリング：その他の例外をキャッチ
    std::cerr << "Unknown Exceptions" << std::endl;
  }

  if (!success) {
#ifdef USE_MPI
    // エラー時、MPIを使用している場合は異常終了
    LightGBM::Linkers::MpiAbortIfIsParallel();
#endif

    exit(-1);
  }
}
