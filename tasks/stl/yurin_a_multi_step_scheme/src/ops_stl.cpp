// Copyright 2024 Yurin Andrey
#include "stl/yurin_a_multi_step_scheme/include/ops_stl.hpp"

#include <thread>

using namespace std::chrono_literals;
using namespace yurin_stl;

void CalculateTempAns(uint32_t start, uint32_t end, uint32_t tempSize, uint32_t j, double h,
                      const std::vector<double>& equation, std::vector<std::vector<double>>& tempAns) {
  if (start >= end) {
    return;
  }

  for (int64_t k = start; k < end; ++k) {
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
}

void CalculateDeltaSum(uint32_t start, uint32_t end, uint32_t tempSize, const std::vector<std::vector<double>>& tempAns,
                       std::vector<double>& deltaSum) {
  if (start >= end) {
    return;
  }

  for (uint32_t j = start; j < end; ++j) {
    for (int k = 0; k < 4; ++k) {
      if (k != 1 and k != 2) {
        deltaSum[j - 1] += tempAns[k][j + tempSize / 2];
      } else {
        deltaSum[j - 1] += 2 * tempAns[k][j + tempSize / 2];
      }
    }
    deltaSum[j - 1] /= 6;
  }
}

void CalculateTempAnsAdams(uint32_t start, uint32_t end, int16_t stepCount, uint32_t ind, uint32_t offset, double h,
                           uint32_t i, uint32_t resSize, const std::vector<std::vector<double>>& res,
                           const std::vector<double>& equation, std::vector<std::vector<double>>& tempAns) {
  if (start >= end) {
    return;
  }

  for (uint32_t j = start; j < end; ++j) {
    for (int16_t k = 0; k < stepCount; ++k) {
      if (k == 0) {
        tempAns[ind][j * offset + k + 1] = res[ind][j + 1];
      } else if (k == 1 or k > 3) {
        if (i == 0) continue;
        tempAns[ind][j * offset + k + 1] = tempAns[ind + 1][j * offset + k] - tempAns[ind][j * offset + k];
      } else if (k == 2) {
        if (j != resSize - 2)
          tempAns[ind][j * offset + k + 1] = res[ind][j + 2];
        else {
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
}

void CalculateAdams(uint32_t start, uint32_t end, uint32_t resSize, double h, uint32_t i, uint32_t ind, uint32_t offset,
                    uint16_t numberOfSteps, const std::vector<double>& equation,
                    const std::vector<std::vector<double>> res, std::vector<std::vector<double>>& tempAns) {
  if (start >= end) {
    return;
  }

  auto resI0 = res[i][0];
  for (uint32_t j = start; j < end; ++j) {
    if (j != resSize - 2) {
      tempAns[ind][j * offset + 3] = res[i][j + 2];
      tempAns[ind][j * offset + 4] = res[i][j + 2] * h;
    } else {
      for (uint32_t l = 1; l < equation.size(); ++l) {
        double summand;
        if (l < equation.size() - 2) {
          summand = (-1) * equation[equation.size() - l - 2] * res[i][l];
        } else if (l == equation.size() - 2) {
          summand = equation[l] * resI0;
        } else {
          summand = equation[l];
        }
        tempAns[ind][j * offset + 3] += summand;
      }
      tempAns[ind][j * offset + 3] /= equation[0];
      tempAns[ind][j * offset + 4] = tempAns[ind][j * offset + 3] * h;
    }
  }

  for (uint32_t j = start; j < end; ++j) {
    for (uint16_t k = 0; k < numberOfSteps - 1; ++k) {
      auto diminutive = tempAns[ind - k][j * offset + 4 + k];
      auto deductible = tempAns[ind - 1 - k][j * offset + 4 + k];
      tempAns[ind - k - 1][j * offset + 5 + k] = diminutive - deductible;
    }
  }
}

bool MultiStepSchemeSTL::pre_processing() {
  internal_order_test();
  // Init value for input and output
  auto* tempEquation = reinterpret_cast<double*>(taskData->inputs[0]);
  equation = std::vector<double>(tempEquation, tempEquation + taskData->inputs_count[0]);

  auto* tempBoundaryConditions = reinterpret_cast<double*>(taskData->inputs[1]);
  boundaryConditions = std::vector<double>(tempBoundaryConditions, tempBoundaryConditions + taskData->inputs_count[1]);

  h = reinterpret_cast<double*>(taskData->inputs[2])[0];
  end = reinterpret_cast<double*>(taskData->inputs[3])[0];
  numThreads = std::thread::hardware_concurrency();

  return true;
}

bool MultiStepSchemeSTL::validation() {
  internal_order_test();
  // Check count elements of output
  auto tend = reinterpret_cast<double*>(taskData->inputs[3])[0];
  auto tstart = reinterpret_cast<double*>(taskData->inputs[1])[0];
  auto th = reinterpret_cast<double*>(taskData->inputs[2])[0];

  return taskData->inputs_count[0] == taskData->inputs_count[1] + 2 && taskData->inputs_count[2] == 1 &&
         taskData->inputs_count[3] == 1 && taskData->outputs_count[0] == (tend - tstart) / th + 1;
}

bool MultiStepSchemeSTL::run() {
  internal_order_test();
  res.clear();
  res.reserve(static_cast<uint32_t>((end - boundaryConditions[0]) / h) + 2);
  res.push_back(boundaryConditions);
  RungeKuttaMethod();
  AdamsMethod();
  return true;
}

bool MultiStepSchemeSTL::post_processing() {
  internal_order_test();
  auto* out_ptr = reinterpret_cast<double*>(taskData->outputs[0]);

  for (uint32_t i = 0; i < res.size(); ++i) {
    out_ptr[i] = res[i][1];
  }

  return true;
}

void MultiStepSchemeSTL::RungeKuttaMethod() {
  std::vector<std::thread> threads(numThreads);
  uint32_t tempSize = 2 * (equation.size() - 3);
  const uint32_t blockSize = (tempSize / 2 + 1) / numThreads;

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
      uint32_t threadsCount = 0;
      for (uint32_t p = 0; p < numThreads; ++p) {
        uint32_t tstart = p * blockSize + 1;
        uint32_t tend = (p == numThreads - 1) ? (tempSize / 2 + 1) : (tstart + blockSize);
        if (tstart == tend) continue;
        threads[threadsCount++] =
            std::thread(CalculateTempAns, tstart, tend, tempSize, j, h, std::ref(equation), std::ref(tempAns));
      }
      for (uint32_t p = 0; p < threadsCount; ++p) {
        threads[p].join();
      }
    }

    std::vector<double> deltaSum(equation.size() - 3);

    uint32_t threadsCount = 0;
    for (uint32_t p = 0; p < numThreads; ++p) {
      uint32_t tstart = p * blockSize + 1;
      uint32_t tend = (p == numThreads - 1) ? (tempSize / 2 + 1) : (tstart + blockSize);
      if (tstart == tend) continue;
      threads[threadsCount++] =
          std::thread(CalculateDeltaSum, tstart, tend, tempSize, std::ref(tempAns), std::ref(deltaSum));
    }
    for (uint32_t p = 0; p < threadsCount; ++p) {
      threads[p].join();
    }

    std::vector<double> temp(res[i].size());
    temp[0] = res[i][0] + h;

    for (uint32_t j = 1; j < res[i].size(); ++j) {
      temp[j] = res[i][j] + deltaSum[j - 1];
    }

    res.push_back(temp);
  }
}

void MultiStepSchemeSTL::AdamsMethod() {
  uint32_t resSize = res[0].size();
  const uint32_t blockSize = (resSize - 1) / numThreads;
  std::vector<std::thread> threads(numThreads);
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
    uint32_t threadsCount = 0;
    for (uint32_t p = 0; p < numThreads; ++p) {
      uint32_t tstart = p * blockSize;
      uint32_t tend = (p == numThreads - 1) ? (resSize - 1) : (tstart + blockSize);
      if (tstart == tend) continue;
      threads[threadsCount++] = std::thread(CalculateTempAnsAdams, tstart, tend, stepCount, ind, offset, h, i, resSize,
                                            std::ref(res), std::ref(equation), std::ref(tempAns));
    }
    for (uint32_t p = 0; p < threadsCount; ++p) {
      threads[p].join();
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

    uint32_t threadsCount = 0;
    for (uint32_t p = 0; p < numThreads; ++p) {
      uint32_t tstart = p * blockSize;
      uint32_t tend = (p == numThreads - 1) ? (resSize - 1) : (tstart + blockSize);
      if (tstart == tend) continue;
      threads[threadsCount++] = std::thread(CalculateAdams, tstart, tend, resSize, h, i, ind, offset, _numberOfSteps,
                                            std::ref(equation), std::ref(res), std::ref(tempAns));
    }
    for (uint32_t p = 0; p < threadsCount; ++p) {
      threads[p].join();
    }

    tempAns.erase(tempAns.begin());
  }
}
