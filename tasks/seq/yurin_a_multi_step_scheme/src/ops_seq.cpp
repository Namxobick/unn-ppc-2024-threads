// Copyright 2024 Yurin Andrey
#include "seq/yurin_a_multi_step_scheme/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool MultiStepSchemeSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  auto* tempEquation = reinterpret_cast<double*>(taskData->inputs[0]);
  equation = std::vector<double>(tempEquation, tempEquation + taskData->inputs_count[0]);

  auto* tempBoundaryConditions = reinterpret_cast<double*>(taskData->inputs[1]);
  boundaryConditions = std::vector<double>(tempBoundaryConditions, tempBoundaryConditions + taskData->inputs_count[1]);

  h = reinterpret_cast<double*>(taskData->inputs[2])[0];
  end = reinterpret_cast<double*>(taskData->inputs[3])[0];

  return true;
}

bool MultiStepSchemeSequential::validation() {
  internal_order_test();
  // Check count elements of output
  auto tend = reinterpret_cast<double*>(taskData->inputs[3])[0];
  auto tstart = reinterpret_cast<double*>(taskData->inputs[1])[0];
  auto th = reinterpret_cast<double*>(taskData->inputs[2])[0];

  return taskData->inputs_count[0] == taskData->inputs_count[1] + 2 && taskData->inputs_count[2] == 1 &&
         taskData->inputs_count[3] == 1 && taskData->outputs_count[0] == (tend - tstart) / th + 1;
}

bool MultiStepSchemeSequential::run() {
  internal_order_test();
  res.clear();
  res.reserve(static_cast<uint32_t>((end - boundaryConditions[0]) / h) + 2);
  res.push_back(boundaryConditions);
  RungeKuttaMethod();
  AdamsMethod();

  return true;
}

bool MultiStepSchemeSequential::post_processing() {
  internal_order_test();
  auto* out_ptr = reinterpret_cast<double*>(taskData->outputs[0]);

  for (uint32_t i = 0; i < res.size(); ++i) {
    out_ptr[i] = res[i][1];
  }
  return true;
}

void MultiStepSchemeSequential::RungeKuttaMethod() {
  uint32_t tempSize = 2 * (equation.size() - 3);
  for (uint32_t i = 0; i < _numberOfSteps - 1; ++i) {
    std::vector<std::vector<double>> tempAns(4);
    tempAns[0] = res[i];
    tempAns[0].resize(tempSize + 1);

    for (uint32_t j = 1; j < 4; ++j) {
      tempAns[j].resize(tempSize + 1);
      if (j != 3) {
        tempAns[j][0] = tempAns[0][0] + h / 2;
      } else {
        tempAns[j][0] = tempAns[0][0] + h;
      }
    }

    for (uint32_t j = 0; j < 4; ++j) {
      for (uint32_t k = 1; k < tempSize / 2 + 1; ++k) {
        if (k != tempSize / 2) {
          tempAns[j][k + tempSize / 2] = h * tempAns[j][k + 1];
        } else {
          for (uint32_t l = 1; l < equation.size(); ++l) {
            double summand{};
            if (l < equation.size() - 2) {
              summand = (-1) * equation[l] * tempAns[j][tempSize / 2 - l + 1];
            } else if (l == equation.size() - 2) {
              summand = equation[l] * tempAns[j][0];
            } else {
              summand = equation[l];
            }
            tempAns[j][k + tempSize / 2] += summand;
          }
          tempAns[j][k + tempSize / 2] *= h / equation[0];
        }

        if (j < 2) {
          tempAns[j + 1][k] = tempAns[j][k] + tempAns[j][k + tempSize / 2] / 2;
        } else if (j < 3) {
          tempAns[j + 1][k] = tempAns[j][k] + tempAns[j][k + tempSize / 2];
        }
      }
    }

    std::vector<double> deltaSum(equation.size() - 3);

    for (uint32_t j = 1; j < tempSize / 2 + 1; ++j) {
      for (int k = 0; k < 4; ++k) {
        if (k != 1 and k != 2) {
          deltaSum[j - 1] += tempAns[k][j + tempSize / 2];
        } else {
          deltaSum[j - 1] += 2 * tempAns[k][j + tempSize / 2];
        }
      }
      deltaSum[j - 1] /= 6;
    }

    std::vector<double> temp(res[i].size());
    temp[0] = res[i][0] + h;

    for (uint32_t j = 1; j < res[i].size(); ++j) {
      temp[j] = res[i][j] + deltaSum[j - 1];
    }

    res.push_back(temp);
  }
}

void MultiStepSchemeSequential::AdamsMethod() {
  std::vector<std::vector<double>> tempAns(4);
  if (end - res[0][0] < 0) {
    return;
  }

  uint32_t stepCount{4};
  uint32_t offset{_numberOfSteps + 3};

  for (uint32_t i = 0; i < _numberOfSteps; ++i) {
    uint32_t ind = _numberOfSteps - i - 1;
    tempAns[ind].resize((equation.size() - 3) * offset + 1);
    tempAns[ind][0] = res[ind][0];

    for (uint32_t j = 0; j < res[0].size() - 1; ++j) {
      for (uint32_t k = 0; k < stepCount; ++k) {
        if (k == 0) {
          tempAns[ind][j * offset + k + 1] = res[ind][j + 1];
        } else if (k == 1 or k > 3) {
          if (i == 0) continue;
          auto diminutive = tempAns[ind + 1][j * offset + k];
          auto deductible = tempAns[ind][j * offset + k];
          tempAns[ind][j * offset + k + 1] = diminutive - deductible;
        } else if (k == 2) {
          if (j != res[0].size() - 2)
            tempAns[ind][j * offset + k + 1] = res[ind][j + 2];
          else {
            for (uint32_t l = 1; l < equation.size(); ++l) {
              double summand = 0;
              if (l < equation.size() - 2) {
                summand = (-1) * equation[equation.size() - l - 2] * tempAns[ind][(l - 1) * offset + k - 1];
              } else if (l == equation.size() - 2) {
                summand = equation[l] * res[ind][0];
              } else {
                summand = equation[l];
              }
              tempAns[ind][j * offset + k + 1] += summand;
            }
            tempAns[ind][j * offset + k + 1] /= equation[0];
          }
        } else {
          tempAns[ind][j * offset + k + 1] = h * tempAns[ind][j * offset + k];
        }
      }
    }
    stepCount++;
  }
  uint32_t ind = _numberOfSteps;
  for (uint32_t i = ind; i < (end - res[0][0]) / h + 1; ++i) {
    tempAns.emplace_back((equation.size() - 3) * offset + 1);
    tempAns[ind][0] = tempAns[ind - 1][0] + h;

    std::vector<double> newStrInAns;
    newStrInAns.reserve(res[0].size());
    newStrInAns.push_back(tempAns[ind - 1][0] + h);

    for (uint32_t j = 0; j < res[0].size() - 1; ++j) {
      double tempDelta{};
      for (uint32_t k = 0; k < _numberOfSteps; ++k) {
        tempDelta += _coefficients[k] * tempAns[ind - k - 1][j * offset + 4 + k];
      }

      tempAns[ind - 1][j * offset + 2] = tempDelta;
      tempAns[ind][j * offset + 1] = tempDelta + tempAns[ind - 1][j * offset + 1];
      newStrInAns.push_back(tempAns[ind][j * offset + 1]);
    }
    res.push_back(newStrInAns);
    newStrInAns.clear();

    for (uint32_t j = 0; j < res[0].size() - 1; ++j) {
      if (j != res[0].size() - 2) {
        tempAns[ind][j * offset + 3] = res[i][j + 2];
        tempAns[ind][j * offset + 4] = res[i][j + 2] * h;
      } else {
        for (uint32_t l = 1; l < equation.size(); ++l) {
          double summand{};
          if (l < equation.size() - 2) {
            summand = (-1) * equation[equation.size() - l - 2] * res[i][l];
          } else if (l == equation.size() - 2) {
            summand = equation[l] * res[i][0];
          } else {
            summand = equation[l];
          }
          tempAns[ind][j * offset + 3] += summand;
        }
        tempAns[ind][j * offset + 3] /= equation[0];
        tempAns[ind][j * offset + 4] = tempAns[ind][j * offset + 3] * h;
      }
    }

    for (uint32_t j = 0; j < res[0].size() - 1; ++j) {
      for (uint32_t k = 0; k < _numberOfSteps - 1; ++k) {
        auto diminutive = tempAns[ind - k][j * offset + 4 + k];
        auto deductible = tempAns[ind - 1 - k][j * offset + 4 + k];
        tempAns[ind - k - 1][j * offset + 5 + k] = diminutive - deductible;
      }
    }
    tempAns.erase(tempAns.begin());
  }
}
