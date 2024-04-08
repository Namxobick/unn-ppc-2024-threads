// Copyright 2024 Yurin Andrey
#include <omp.h>

#include <cmath>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "gtest/gtest.h"
#include "omp/yurin_a_multi_step_scheme/include/ops_omp.hpp"

TEST(Yurin_A_Multi_Step_Scheme_OMP, test_pipeline_run) {
  // Create data
  uint32_t inputSize = 5000000;
  std::vector<double> equation(inputSize);
  std::vector<double> boundaryConditions(inputSize - 3);

  for (uint32_t i = 0; i < inputSize; ++i) {
    equation[i] = cos(i);
    if (i < boundaryConditions.size()) boundaryConditions[i] = sin(i);
  }

  double h{0.25};
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
  auto multiStepSchemeOMP = std::make_shared<MultiStepSchemeOMP>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  perfAttr->current_timer = [&] { return omp_get_wtime(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(multiStepSchemeOMP);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(Yurin_A_Multi_Step_Scheme_OMP, test_task_run) {
  // Create data
  uint32_t inputSize = 5000000;
  std::vector<double> equation(inputSize);
  std::vector<double> boundaryConditions(inputSize - 3);

  for (uint32_t i = 0; i < inputSize; ++i) {
    equation[i] = cos(i);
    if (i < boundaryConditions.size()) boundaryConditions[i] = sin(i);
  }

  double h{0.25};
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

  auto multiStepSchemeOMP = std::make_shared<MultiStepSchemeOMP>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  perfAttr->current_timer = [&] { return omp_get_wtime(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(multiStepSchemeOMP);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}
