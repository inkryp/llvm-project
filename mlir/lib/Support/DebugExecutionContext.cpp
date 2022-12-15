//===- DebugExecutionContext.cpp - Debug Execution Context Support --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/DebugExecutionContext.h"
#include "mlir/Support/BreakpointManagers/FileLineColLocBreakpointManager.h"
#include "mlir/Support/BreakpointManagers/RewritePatternBreakpointManager.h"
#include "mlir/Support/BreakpointManagers/SimpleBreakpointManager.h"

llvm::SmallVector<mlir::BreakpointManager *> &
getGlobalInstancesOfBreakpointManagers() {
  static llvm::SmallVector<mlir::BreakpointManager *> breakpointManagers = {
      &mlir::FileLineColLocBreakpointManager::getGlobalInstance(),
      &mlir::RewritePatternBreakpointManager::getGlobalInstance(),
      &mlir::SimpleBreakpointManager::getGlobalInstance()};
  return breakpointManagers;
}

using namespace mlir;

//===----------------------------------------------------------------------===//
// DebugExecutionContext
//===----------------------------------------------------------------------===//

DebugExecutionContext::DebugExecutionContext(
    llvm::function_ref<DebugExecutionControl(
        ArrayRef<IRUnit>, ArrayRef<StringRef>, StringRef, StringRef,
        const int &, const DebugActionInformation *)>
        callback)
    : onBreakpointControlExecutionCallback(callback), daiHead(nullptr) {}

void DebugExecutionContext::registerObserver(Observer *observer) {
  observers.push_back(observer);
}

FailureOr<bool>
DebugExecutionContext::execute(ArrayRef<IRUnit> units,
                               ArrayRef<StringRef> instanceTags,
                               llvm::function_ref<ActionResult()> transform,
                               const DebugActionBase &action) {
  DebugActionInformation info{daiHead, action};
  daiHead = &info;
  ++depth;
  auto handleUserInput = [&]() -> bool {
    auto todoNext = onBreakpointControlExecutionCallback(
        units, instanceTags, action.tag, action.desc, depth, daiHead);
    switch (todoNext) {
    case DebugExecutionControl::Apply:
      depthToBreak = std::nullopt;
      return true;
    case DebugExecutionControl::Skip:
      depthToBreak = std::nullopt;
      return false;
    case DebugExecutionControl::Step:
      depthToBreak = depth + 1;
      return true;
    case DebugExecutionControl::Next:
      depthToBreak = depth;
      return true;
    case DebugExecutionControl::Finish:
      depthToBreak = depth - 1;
      return true;
    }
  };
  llvm::Optional<Breakpoint *> breakpoint;
  for (auto *breakpointManager : getGlobalInstancesOfBreakpointManagers()) {
    auto cur = breakpointManager->match(action, instanceTags, units);
    if (cur) {
      breakpoint = cur;
      // TODO(inkryp): Most of the times we only want to check if there is one.
      // However that is not always the case. How should we handle the existance
      // of multiple breakpoints?
      break;
    }
  }
  // For now implement it with breakpoint
  for (auto *observer : observers) {
    observer->onCallbackBeforeExecution(units, instanceTags, daiHead, depth,
                                        breakpoint);
  }
  bool apply = true;
  if (breakpoint || (depthToBreak && depth <= depthToBreak)) {
    apply = handleUserInput();
  }

  if (apply) {
    transform();
  }

  if (depthToBreak && depth <= depthToBreak) {
    handleUserInput();
  }
  --depth;
  daiHead = info.prev;
  return apply;
}
