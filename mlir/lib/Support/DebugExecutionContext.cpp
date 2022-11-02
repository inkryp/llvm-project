//===- DebugExecutionContext.cpp - Debug Execution Context Support --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/DebugExecutionContext.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// DebugExecutionContext
//===----------------------------------------------------------------------===//

FailureOr<bool>
DebugExecutionContext::execute(ArrayRef<IRUnit> units,
                               ArrayRef<StringRef> instanceTags,
                               llvm::function_ref<ActionResult()> transform,
                               const DebugActionBase &action) {
  DebugActionInformation info{daiHead, action, 0};
  if (daiHead) {
    info.depth = daiHead->depth;
  }
  info.depth++;
  daiHead = &info;
  auto &depth = daiHead->depth;
  auto handleUserInput = [&]() -> bool {
    auto todoNext =
        OnBreakpoint(units, instanceTags, action.tag, action.desc, daiHead);
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
  auto breakpoint = sbm.match(action.tag);
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
  info.depth--;
  daiHead = info.prev;
  return apply;
}
