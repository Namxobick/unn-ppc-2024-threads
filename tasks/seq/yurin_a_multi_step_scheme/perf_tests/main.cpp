// Copyright 2024 Yurin Andrey
#include <cmath>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "gtest/gtest.h"
#include "seq/yurin_a_multi_step_scheme/include/ops_seq.hpp"

TEST(Yurin_A_Multi_Step_Scheme_Seq, test_pipeline_run) {
  // Create data
  std::vector<double> equation{1, 1, 2, 2, 1, 1, 0, 0};
  std::vector<double> boundaryConditions{0, 1, 2, -2, 1, 1};
  double h{0.00000018};
  double end{1};

  uint32_t size = (end - boundaryConditions[0]) / h + 1;
  std::vector<double> out(size, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(equation.data()));
  taskDataSeq->inputs_count.emplace_back(equation.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(boundaryConditions.data()));
  taskDataSeq->inputs_count.emplace_back(boundaryConditions.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&h));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&end));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto multiStepSchemeSequential = std::make_shared<MultiStepSchemeSequential>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(multiStepSchemeSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  for (uint32_t i = 0; i < size; i++) {
    double x = i * h;
    EXPECT_NEAR(out[i], 0.5 * ((-1) * exp(-x) + 5 * sin(x) + (3 - 2 * x) * cos(x)), 1e-8);
  }
}

TEST(Yurin_A_Multi_Step_Scheme_Seq, test_task_run) {
  // Create data
  std::vector<double> equation{1, 1, 2, 2, 1, 1, 0, 0};
  std::vector<double> boundaryConditions{0, 1, 2, -2, 1, 1};
  double h{0.00000018};
  double end{1};

  uint32_t size = (end - boundaryConditions[0]) / h + 1;
  std::vector<double> out(size, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(equation.data()));
  taskDataSeq->inputs_count.emplace_back(equation.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(boundaryConditions.data()));
  taskDataSeq->inputs_count.emplace_back(boundaryConditions.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&h));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&end));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto multiStepSchemeSequential = std::make_shared<MultiStepSchemeSequential>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(multiStepSchemeSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  for (uint32_t i = 0; i < size; i++) {
    double x = i * h;
    EXPECT_NEAR(out[i], 0.5 * ((-1) * exp(-x) + 5 * sin(x) + (3 - 2 * x) * cos(x)), 1e-8);
  }
}
