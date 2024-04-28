// Copyright 2024 Yurin Andrey
#include "tbb/yurin_a_multi_step_scheme/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <thread>

using namespace std::chrono_literals;
using namespace yurin_tbb;

bool MultiStepSchemeTBB::pre_processing() {
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

bool MultiStepSchemeTBB::validation() {
  internal_order_test();
  // Check count elements of output
  auto tend = reinterpret_cast<double*>(taskData->inputs[3])[0];
  auto tstart = reinterpret_cast<double*>(taskData->inputs[1])[0];
  auto th = reinterpret_cast<double*>(taskData->inputs[2])[0];

  return taskData->inputs_count[0] == taskData->inputs_count[1] + 2 && taskData->inputs_count[2] == 1 &&
         taskData->inputs_count[3] == 1 && taskData->outputs_count[0] == (tend - tstart) / th + 1;
}

bool MultiStepSchemeTBB::run() {
  internal_order_test();
  res.clear();
  res.reserve(static_cast<uint32_t>((end - boundaryConditions[0]) / h) + 2);
  res.push_back(boundaryConditions);
  RungeKuttaMethod();
  AdamsMethod();

  return true;
}

bool MultiStepSchemeTBB::post_processing() {
  internal_order_test();
  auto* out_ptr = reinterpret_cast<double*>(taskData->outputs[0]);

  for (uint32_t i = 0; i < res.size(); ++i) {
    out_ptr[i] = res[i][1];
  }
  return true;
}

void MultiStepSchemeTBB::RungeKuttaMethod() {
  uint32_t tempSize = 2 * (equation.size() - 3);

  for (int16_t i = 0; i < _numberOfSteps - 1; ++i) {
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
      tbb::parallel_for(tbb::blocked_range<size_t>(1, tempSize / 2 + 1), [&](const tbb::blocked_range<size_t>& r) {
        for (size_t k = r.begin(); k != r.end(); ++k) {
          if (k != tempSize / 2) {
            tempAns[j][k + tempSize / 2] = h * tempAns[j][k + 1];
          } else {
            double summand = 0;
            for (uint32_t l = 1; l < equation.size(); ++l) {
              if (l < equation.size() - 2) {
                summand += (-1) * equation[l] * tempAns[j][tempSize / 2 - l + 1];
              } else if (l == equation.size() - 2) {
                summand += equation[l] * tempAns[j][0];
              } else {
                summand += equation[l];
              }
            }
            tempAns[j][k + tempSize / 2] = summand * h / equation[0];
          }

          if (j < 2) {
            tempAns[j + 1][k] = tempAns[j][k] + tempAns[j][k + tempSize / 2] / 2;
          } else if (j < 3) {
            tempAns[j + 1][k] = tempAns[j][k] + tempAns[j][k + tempSize / 2];
          }
        }
      });
    }

    std::vector<double> deltaSum(equation.size() - 3);
    tbb::parallel_for(tbb::blocked_range<size_t>(1, tempSize / 2 + 1), [&](const tbb::blocked_range<size_t>& r) {
      for (size_t j = r.begin(); j != r.end(); ++j) {
        double sum = 0;
        for (int k = 0; k < 4; ++k) {
          if (k != 1 && k != 2) {
            sum += tempAns[k][j + tempSize / 2];
          } else {
            sum += 2 * tempAns[k][j + tempSize / 2];
          }
        }
        deltaSum[j - 1] = sum / 6;
      }
    });

    std::vector<double> temp(res[i].size());
    temp[0] = res[i][0] + h;

    for (uint32_t j = 1; j < res[i].size(); ++j) {
      temp[j] = res[i][j] + deltaSum[j - 1];
    }

    res.push_back(temp);
  }
}

void MultiStepSchemeTBB::AdamsMethod() {
  int32_t resSize = res[0].size();
  std::vector<std::vector<double>> tempAns(4);
  if (end - res[0][0] < 0) {
    return;
  }

  int16_t stepCount{4};
  int16_t offset = _numberOfSteps + 3;

  for (int16_t i = 0; i < _numberOfSteps; ++i) {
    uint32_t ind = _numberOfSteps - i - 1;
    tempAns[ind].resize((equation.size() - 3) * offset + 1);
    tempAns[ind][0] = res[ind][0];
    tbb::parallel_for(tbb::blocked_range<int32_t>(0, resSize - 1), [&](const tbb::blocked_range<int32_t>& r) {
      for (int32_t j = r.begin(); j != r.end(); ++j) {
        for (int16_t k = 0; k < stepCount; ++k) {
          if (k == 0) {
            tempAns[ind][j * offset + k + 1] = res[ind][j + 1];
          } else if (k == 1 || k > 3) {
            if (i == 0) continue;
            tempAns[ind][j * offset + k + 1] = tempAns[ind + 1][j * offset + k] - tempAns[ind][j * offset + k];
          } else if (k == 2) {
            if (j != resSize - 2) {
              tempAns[ind][j * offset + k + 1] = res[ind][j + 2];
            } else {
              double summand = 0;
              for (uint32_t l = 1; l < equation.size(); ++l) {
                if (l < equation.size() - 2) {
                  summand += (-1) * equation[equation.size() - l - 2] * tempAns[ind][(l - 1) * offset + k - 1];
                } else if (l == equation.size() - 2) {
                  summand += equation[l] * res[ind][0];
                } else {
                  summand += equation[l];
                }
              }
              tempAns[ind][j * offset + k + 1] = summand / equation[0];
            }
          } else {
            tempAns[ind][j * offset + k + 1] = h * tempAns[ind][j * offset + k];
          }
        }
      }
    });
    stepCount++;
  }

  int16_t ind = _numberOfSteps;

  for (uint32_t i = ind; i < (end - res[0][0]) / h + 1; ++i) {
    std::vector<double> newStrInAns;
    tempAns.emplace_back((equation.size() - 3) * offset + 1);
    tempAns[ind][0] = tempAns[ind - 1][0] + h;

    newStrInAns.reserve(res[0].size());
    newStrInAns.push_back(tempAns[ind - 1][0] + h);
    for (uint32_t j = 0; j < res[0].size() - 1; ++j) {
      double tempDelta{};
      for (int16_t k = 0; k < _numberOfSteps; ++k) {
        tempDelta += _coefficients[k] * tempAns[ind - k - 1][j * offset + 4 + k];
      }

      tempAns[ind - 1][j * offset + 2] = tempDelta;
      tempAns[ind][j * offset + 1] = tempDelta + tempAns[ind - 1][j * offset + 1];
      newStrInAns.push_back(tempAns[ind][j * offset + 1]);
    }

    res.push_back(newStrInAns);
    auto resI0 = res[i][0];
    auto equationSize = equation.size();
    tbb::parallel_for(tbb::blocked_range<int32_t>(0, resSize - 1), [&](const tbb::blocked_range<int32_t>& r) {
      for (int32_t j = r.begin(); j != r.end(); ++j) {
        if (j != resSize - 2) {
          auto x = res[i][j + 2];
          tempAns[ind][j * offset + 3] = x;
          tempAns[ind][j * offset + 4] = x * h;
        } else {
          double summand = 0;
          for (uint32_t l = 1; l < equationSize; ++l) {
            if (l < equationSize - 2) {
              summand += (-1) * equation[equationSize - l - 2] * res[i][l];
            } else if (l == equationSize - 2) {
              summand += equation[l] * resI0;
            } else {
              summand += equation[l];
            }
          }
          tempAns[ind][j * offset + 3] = summand / equation[0];
          tempAns[ind][j * offset + 4] = summand / equation[0] * h;
        }
      }
    });
    tbb::parallel_for(tbb::blocked_range<int32_t>(0, resSize - 1), [&](const tbb::blocked_range<int32_t>& r) {
      for (int32_t j = r.begin(); j != r.end(); ++j) {
        for (int32_t k = 0; k < _numberOfSteps - 1; ++k) {
          tempAns[ind - k - 1][j * offset + 5 + k] =
              tempAns[ind - k][j * offset + 4 + k] - tempAns[ind - 1 - k][j * offset + 4 + k];
        }
      }
    });
    tempAns.erase(tempAns.begin());
  }
}
