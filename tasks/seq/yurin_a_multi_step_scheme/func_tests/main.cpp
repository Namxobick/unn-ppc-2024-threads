// Copyright 2024 Yurin Andrey
#include <cmath>
#include <vector>

#include "gtest/gtest.h"
#include "seq/yurin_a_multi_step_scheme/include/ops_seq.hpp"

TEST(Yurin_A_Multi_Step_Scheme_Seq, incorrect_data) {
  // Create data
  std::vector<double> equation{-2, -3, 1, -2, 0, 0};
  std::vector<double> boundaryConditions{0, 1, 2, -4, 6};
  double h{0.2};
  double end{1};

  std::vector<double> ans{1, 1.2798, 1.457366035277778, 1.5478079006794228, 1.5907511599041377, 1.5482917405772707};

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
  MultiStepSchemeSequential multiStepSchemeSequential(taskDataSeq);

  ASSERT_FALSE(multiStepSchemeSequential.validation());
}

TEST(Yurin_A_Multi_Step_Scheme_Seq, simple_test) {
  // Create data
  std::vector<double> equation{1, 0, 0, 0};
  std::vector<double> boundaryConditions{0, 1};
  double h{0.0001};
  double end{0.01};

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
  MultiStepSchemeSequential multiStepSchemeSequential(taskDataSeq);

  ASSERT_TRUE(multiStepSchemeSequential.validation());
  ASSERT_TRUE(multiStepSchemeSequential.pre_processing());
  ASSERT_TRUE(multiStepSchemeSequential.run());
  ASSERT_TRUE(multiStepSchemeSequential.post_processing());

  for (uint32_t i = 0; i < size; i++) {
    EXPECT_DOUBLE_EQ(out[i], 1);
  }
}

TEST(Yurin_A_Multi_Step_Scheme_Seq, first_test) {
  // Create data
  std::vector<double> equation{-2, -3, 1, -2, 0, 0};
  std::vector<double> boundaryConditions{0, 1, 2, -4};
  double h{0.2};
  double end{1};

  std::vector<double> ans{1, 1.2798, 1.457366035277778, 1.5478079006794228, 1.5907511599041377, 1.5482917405772707};

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
  MultiStepSchemeSequential multiStepSchemeSequential(taskDataSeq);

  ASSERT_TRUE(multiStepSchemeSequential.validation());
  ASSERT_TRUE(multiStepSchemeSequential.pre_processing());
  ASSERT_TRUE(multiStepSchemeSequential.run());
  ASSERT_TRUE(multiStepSchemeSequential.post_processing());

  for (uint32_t i = 0; i < ans.size(); i++) {
    EXPECT_DOUBLE_EQ(out[i], ans[i]);
  }
}

TEST(Yurin_A_Multi_Step_Scheme_Seq, second_test) {
  // Create data
  std::vector<double> equation{1, 1, 0, 0, 0, 0};
  std::vector<double> boundaryConditions{0, 0, 1, -1};
  double h{0.0001};
  double end{0.01};

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
  MultiStepSchemeSequential multiStepSchemeSequential(taskDataSeq);

  ASSERT_TRUE(multiStepSchemeSequential.validation());
  ASSERT_TRUE(multiStepSchemeSequential.pre_processing());
  ASSERT_TRUE(multiStepSchemeSequential.run());
  ASSERT_TRUE(multiStepSchemeSequential.post_processing());

  for (uint32_t i = 0; i < size; i++) {
    double x = i * h;
    EXPECT_NEAR(out[i], (1 - exp((-1) * x)), 1e-6);
  }
}

TEST(Yurin_A_Multi_Step_Scheme_Seq, fird_test) {
  // Create data
  std::vector<double> equation{1, -1, 2, 0};
  std::vector<double> boundaryConditions{0, 0.1};
  double h{0.0001};
  double end{0.01};

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
  MultiStepSchemeSequential multiStepSchemeSequential(taskDataSeq);

  ASSERT_TRUE(multiStepSchemeSequential.validation());
  ASSERT_TRUE(multiStepSchemeSequential.pre_processing());
  ASSERT_TRUE(multiStepSchemeSequential.run());
  ASSERT_TRUE(multiStepSchemeSequential.post_processing());

  for (uint32_t i = 0; i < size; i++) {
    double x = i * h;
    EXPECT_NEAR(out[i], (-2 * x + 2.1 * exp(x) - 2), 1e-6);
  }
}
