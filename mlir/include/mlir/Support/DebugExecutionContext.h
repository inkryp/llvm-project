//===- DebugExecutionContext.h - Debug Execution Context Support *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TODO: Write a proper description for the service
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_DEBUGEXECUTIONCONTEXT_H
#define MLIR_SUPPORT_DEBUGEXECUTIONCONTEXT_H

#include "mlir/Support/BreakpointManagers/SimpleBreakpointManager.h"
#include "mlir/Support/DebugAction.h"

namespace mlir {

enum DebugExecutionControl {
  Apply = 1,
  Skip = 2,
  Step = 3,
  Next = 4,
  Finish = 5
};

struct DebugActionInformation {
  const DebugActionInformation *prev;
  const DebugActionBase &action;
};

class DebugExecutionContext : public DebugActionManager::GenericHandler {
public:
  DebugExecutionContext(
      llvm::function_ref<DebugExecutionControl(
          ArrayRef<IRUnit>, ArrayRef<StringRef>, StringRef, StringRef,
          const int &, const DebugActionInformation *)>
          callback)
      : OnBreakpoint(callback), sbm(SimpleBreakpointManager::getGlobalSbm()),
        daiHead(nullptr) {}
  FailureOr<bool> execute(ArrayRef<IRUnit> units,
                          ArrayRef<StringRef> instanceTags,
                          llvm::function_ref<ActionResult()> transform,
                          const DebugActionBase &action) final;
  SimpleBreakpoint *addSimpleBreakpoint(StringRef tag) {
    return sbm.addBreakpoint(tag);
  }
  void disableSimpleBreakpoint(SimpleBreakpoint *breakpoint) {
    sbm.disableBreakpoint(breakpoint);
  }
  void enableSimpleBreakpoint(SimpleBreakpoint *breakpoint) {
    sbm.enableBreakpoint(breakpoint);
  }
  void deleteSimpleBreakpoint(SimpleBreakpoint *breakpoint) {
    sbm.deleteBreakpoint(breakpoint);
  }

private:
  llvm::function_ref<DebugExecutionControl(
      ArrayRef<IRUnit>, ArrayRef<StringRef>, StringRef, StringRef,
      const int &depth, const DebugActionInformation *)>
      OnBreakpoint;
  int depth = 0;
  SimpleBreakpointManager &sbm;
  const DebugActionInformation *daiHead;
  Optional<int> depthToBreak;
};

} // namespace mlir

#endif // MLIR_SUPPORT_DEBUGEXECUTIONCONTEXT_H
