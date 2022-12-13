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

#include "mlir/Support/BreakpointManagers/BreakpointManager.h"
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
          callback);

  struct Observer {
    Observer(
        llvm::function_ref<void(ArrayRef<IRUnit>, ArrayRef<StringRef>,
                                const DebugActionInformation *, const int &,
                                llvm::Optional<Breakpoint *>)>
            onCallbackBeforeExecution =
                [](ArrayRef<IRUnit>, ArrayRef<StringRef>,
                   const DebugActionInformation *, const int &,
                   llvm::Optional<Breakpoint *>) {},
        llvm::function_ref<void(ActionResult)> onCallbackAfterExecution =
            [](ActionResult) {})
        : onCallbackBeforeExecution(onCallbackBeforeExecution),
          onCallbackAfterExecution(onCallbackAfterExecution) {}
    llvm::function_ref<void(ArrayRef<IRUnit>, ArrayRef<StringRef>,
                            const DebugActionInformation *, const int &,
                            llvm::Optional<Breakpoint *>)>
        onCallbackBeforeExecution;
    llvm::function_ref<void(ActionResult)> onCallbackAfterExecution;
  };

  void registerObserver(Observer *);

  FailureOr<bool> execute(ArrayRef<IRUnit> units,
                          ArrayRef<StringRef> instanceTags,
                          llvm::function_ref<ActionResult()> transform,
                          const DebugActionBase &action) final;

private:
  llvm::function_ref<DebugExecutionControl(
      ArrayRef<IRUnit>, ArrayRef<StringRef>, StringRef, StringRef, const int &,
      const DebugActionInformation *)>
      onBreakpointControlExecutionCallback;

  int depth = 0;

  const DebugActionInformation *daiHead;

  Optional<int> depthToBreak;

  SmallVector<BreakpointManager *> breakpointManagers;

  SmallVector<Observer *> observers;
};

} // namespace mlir

#endif // MLIR_SUPPORT_DEBUGEXECUTIONCONTEXT_H
