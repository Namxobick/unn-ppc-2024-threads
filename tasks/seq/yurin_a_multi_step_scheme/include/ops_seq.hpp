// Copyright 2024 Yurin Andrey
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

class MultiStepSchemeSequential : public ppc::core::Task {
 public:
  explicit MultiStepSchemeSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  void RungeKuttaMethod();
  void AdamsMethod();

  const std::vector<double> _coefficients = {1, 0.5, 0.42, 0.38};
  const uint32_t _numberOfSteps = 4;

  std::vector<double> equation{};
  std::vector<double> boundaryConditions{};
  double h{};
  double end{};

  std::vector<std::vector<double>> res{};
};
